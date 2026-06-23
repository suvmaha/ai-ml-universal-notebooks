"""
playground_config.py — Backend switcher for AI-ML Playground notebooks.

Usage (first cell of any notebook):
    from playground_config import get_llm, get_embeddings
    llm = get_llm()
    embeddings = get_embeddings()

Switch backend by setting LLM_BACKEND env var before starting JupyterHub,
or override inline:
    llm = get_llm(backend="anthropic")

Supported backends:
    ollama      — in-cluster Ollama (default, no API key needed)
    anthropic   — Anthropic API (requires ANTHROPIC_API_KEY)
    bedrock     — AWS Bedrock via IRSA (requires IAM role, no key needed)
    openai      — OpenAI API (requires OPENAI_API_KEY)
"""

import os

# Backend selection — set LLM_BACKEND env var or pass backend= argument
DEFAULT_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# Ollama defaults — in-cluster service
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama.ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Anthropic defaults
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Bedrock defaults
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "anthropic.claude-sonnet-4-5")
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")

# OpenAI defaults
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_llm(backend: str = None, **kwargs):
    """Return a LangChain chat model for the selected backend."""
    backend = backend or DEFAULT_BACKEND

    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            **kwargs,
        )

    elif backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            **kwargs,
        )

    elif backend == "bedrock":
        from langchain_aws import ChatBedrock
        return ChatBedrock(
            model_id=BEDROCK_MODEL,
            region_name=BEDROCK_REGION,
            **kwargs,
        )

    elif backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_MODEL,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}. Choose: ollama, anthropic, bedrock, openai")


def get_embeddings(backend: str = None, **kwargs):
    """Return a LangChain embeddings model for the selected backend."""
    backend = backend or DEFAULT_BACKEND

    if backend == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
            **kwargs,
        )

    elif backend == "anthropic":
        # Anthropic doesn't provide embeddings — fall back to Ollama
        print("Anthropic does not provide embeddings. Using Ollama nomic-embed-text.")
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    elif backend == "bedrock":
        from langchain_aws import BedrockEmbeddings
        return BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=BEDROCK_REGION,
            **kwargs,
        )

    elif backend == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(**kwargs)

    else:
        raise ValueError(f"Unknown backend: {backend}. Choose: ollama, anthropic, bedrock, openai")


def status():
    """Print current backend configuration."""
    print(f"LLM Backend    : {DEFAULT_BACKEND}")
    print(f"Ollama URL     : {OLLAMA_BASE_URL}")
    print(f"Ollama model   : {OLLAMA_MODEL}")
    print(f"Embed model    : {OLLAMA_EMBED_MODEL}")
    print(f"Anthropic model: {ANTHROPIC_MODEL}")
    print(f"Bedrock model  : {BEDROCK_MODEL}")
