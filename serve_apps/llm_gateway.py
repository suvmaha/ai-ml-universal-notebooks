"""Ray Serve LLM Gateway — a scalable HTTP endpoint in front of in-cluster Ollama.

Demonstrates the core Ray Serve mechanics on the AI-ML Playground:
  - HTTP ingress (Serve's built-in Starlette handling)
  - multiple replicas + fractional CPU packing
  - autoscaling config
  - composition: Serve (serving layer) -> Ollama (model backend)

Deployed by the RayService in components/ray-serve/.
Import path: serve_apps.llm_gateway:app
"""

import os
import requests
from starlette.requests import Request
from ray import serve

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama.ollama:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


@serve.deployment(
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 4,
        "target_ongoing_requests": 2,
    },
    ray_actor_options={"num_cpus": 0.5},
)
class LLMGateway:
    """One HTTP endpoint; each replica forwards prompts to Ollama."""

    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = MODEL):
        self.ollama_url = ollama_url
        self.model = model

    def _generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    async def __call__(self, request: Request) -> dict:
        # GET /serve/  -> health/info;  POST {"prompt": "..."} -> generation
        if request.method == "GET":
            return {
                "status": "ok",
                "service": "ray-serve llm-gateway",
                "model": self.model,
                "usage": 'POST JSON {"prompt": "..."}',
            }
        try:
            body = await request.json()
        except Exception:
            body = {}
        prompt = body.get("prompt", "Say hello in one short sentence.")
        return {"model": self.model, "prompt": prompt, "response": self._generate(prompt)}


# Bound application object referenced by the RayService import_path.
app = LLMGateway.bind()
