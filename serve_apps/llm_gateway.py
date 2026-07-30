"""Ray Serve LLM Gateway — resilient HTTP endpoint over self-hosted Ollama.

Two backends: a PRIMARY (e.g. the GPU 70B) and a FALLBACK (e.g. the CPU 3B).
- GET  /   → real health check: actually probes both backends for the model
- POST /   → generate; tries primary, falls back to secondary, else 503

Deployed by the RayService in components/ray-serve/. Import path: serve_apps.llm_gateway:app
"""

import os
import requests
from starlette.requests import Request
from starlette.responses import JSONResponse
from ray import serve

PRIMARY_URL = os.environ.get("OLLAMA_URL", "http://ollama-gpu.ollama:11434")
PRIMARY_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.3:70b")
FALLBACK_URL = os.environ.get("FALLBACK_URL", "http://ollama.ollama:11434")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "llama3.2")

# (connect timeout, read timeout): fail fast on an unreachable backend so the
# fallback kicks in quickly instead of hanging on the full read timeout.
TIMEOUT = (3, 120)


@serve.deployment(
    autoscaling_config={"min_replicas": 2, "max_replicas": 4, "target_ongoing_requests": 2},
    ray_actor_options={"num_cpus": 0.5},
)
class LLMGateway:
    def __init__(self):
        self.primary = (PRIMARY_URL, PRIMARY_MODEL)
        self.fallback = (FALLBACK_URL, FALLBACK_MODEL)

    def _ready(self, url: str, model: str) -> bool:
        """Is this backend reachable AND does it have the model loaded?"""
        try:
            r = requests.get(f"{url}/api/tags", timeout=3)
            r.raise_for_status()
            family = model.split(":")[0]
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(n == model or n.startswith(family) for n in names)
        except Exception:
            return False

    def _generate(self, url: str, model: str, prompt: str) -> str:
        r = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    async def __call__(self, request: Request):
        # GET → real health: report the TRUTH about each backend.
        if request.method == "GET":
            p_ok = self._ready(*self.primary)
            f_ok = self._ready(*self.fallback)
            status = "ok" if p_ok else ("degraded" if f_ok else "down")
            return {
                "status": status,  # ok = primary live · degraded = only fallback · down = neither
                "service": "ray-serve llm-gateway",
                "primary": {"model": self.primary[1], "ready": p_ok},
                "fallback": {"model": self.fallback[1], "ready": f_ok},
                "usage": 'POST JSON {"prompt": "..."}',
            }

        # POST → generate, primary first then fallback.
        try:
            body = await request.json()
        except Exception:
            body = {}
        prompt = body.get("prompt", "Say hello in one short sentence.")

        last_err = None
        for label, (url, model) in (("primary", self.primary), ("fallback", self.fallback)):
            try:
                response = self._generate(url, model, prompt)
                return {"served_by": label, "model": model, "prompt": prompt, "response": response}
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue

        # Both backends failed — return a clean 503, not a raw 500 / hang.
        return JSONResponse(
            status_code=503,
            content={
                "error": "all backends unavailable",
                "tried": [self.primary[1], self.fallback[1]],
                "detail": last_err,
            },
        )


app = LLMGateway.bind()
