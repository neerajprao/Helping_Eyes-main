# Helping Eyes in a container: runs on any host that runs Docker (Hugging Face Spaces, Render, a VPS, ...).
# Built from the folder assembled by cloud/deploy_space.sh. Set LLM_API_KEY as a secret on the host.
FROM python:3.11-slim

# libgl / libglib are needed by OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hosts such as Hugging Face run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download the OCR models at build time, so the app starts quickly
RUN python -c "from rapidocr import RapidOCR; RapidOCR()"

COPY --chown=user . .
EXPOSE 7860
CMD ["bash", "start.sh"]
