# Helping Eyes web version for a Hugging Face Space (Docker SDK, free "CPU basic" hardware).
# Built from the folder assembled by cloud/deploy_space.sh.
FROM python:3.11-slim

# System packages: curl for the Ollama installer, zstd to unpack it, libgl/libglib for OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates zstd libgl1 libglib2.0-0 procps \
    && rm -rf /var/lib/apt/lists/*

# Ollama serves the language model on CPU
RUN curl -fsSL https://ollama.com/install.sh | sh

# Hugging Face runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_MODELS=/home/user/.ollama/models \
    LLM_MODEL=qwen2.5:3b-instruct \
    OCR_ENGINE=rapidocr \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download the OCR models and the language model at build time, so the Space starts quickly
RUN python -c "from rapidocr import RapidOCR; RapidOCR()"
RUN (ollama serve > /tmp/ollama-build.log 2>&1 &) && sleep 8 && ollama pull "$LLM_MODEL"

COPY --chown=user . .
EXPOSE 7860
CMD ["bash", "start.sh"]
