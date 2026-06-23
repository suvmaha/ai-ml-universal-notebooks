# AI-ML Universal Notebooks

Notebooks for learning and experimenting with AI/ML frameworks — LLMs, RAG, agents, MLOps.

Run them anywhere: JupyterHub, Colab, local Jupyter. Built to be backend-agnostic via `playground_config.py`.

---

## Quick Start

```bash
git clone https://github.com/suvmaha/ai-ml-universal-notebooks.git
```

Open any notebook and run it.

---

## Backend Configuration

`playground_config.py` is included in this repo. It lets notebooks switch between LLM backends without changing notebook code.

```python
from playground_config import get_llm, get_embeddings

llm = get_llm()  # defaults to Ollama (in-cluster or local)
# llm = get_llm(backend="anthropic")
# llm = get_llm(backend="bedrock")
# llm = get_llm(backend="openai")
```

---

## Notebooks

### LLM Backends
| Notebook | What it covers |
|----------|---------------|
| [llm-backends/test-ollama.ipynb](llm-backends/test-ollama.ipynb) | Verify Ollama, run llama3.2, test embeddings |

### LangChain
| Notebook | What it covers |
|----------|---------------|
| [langchain/hello-langchain.ipynb](langchain/hello-langchain.ipynb) | Build a chain, prompt template, streaming |

### RAG
| Notebook | What it covers |
|----------|---------------|
| coming | |

### Agents
| Notebook | What it covers |
|----------|---------------|
| coming | |

### MLOps
| Notebook | What it covers |
|----------|---------------|
| coming | |
