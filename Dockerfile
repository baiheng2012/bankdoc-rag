FROM python:3.12-slim

# PYTHONPATH lets "import bankdoc" work without installing the package.
# HF_HOME is where the embedding model is cached (baked into the image below).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only PyTorch (the default wheel bundles CUDA libraries and is several GB larger).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies come BEFORE the source code: Docker caches each layer, so editing
# code does not re-install every package.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the container starts offline.
# The model name must match EMBEDDING_MODEL in core/config.py.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY src ./src

# Do not run as root: a compromised process then has far fewer privileges.
RUN useradd --create-home --uid 10001 app
USER app

# From here on, never contact the Hugging Face hub at runtime.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

# /health needs no authentication, no database and no LLM.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "bankdoc.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
