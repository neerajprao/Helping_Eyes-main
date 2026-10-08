# Helping Eyes in a container: runs on any host that runs Docker (Hugging Face Spaces, Render, a VPS, ...).
# Build context: the project's top folder (a checkout of the repository, or what cloud/deploy_space.sh
# assembles). Set LLM_API_KEY as a secret on the host.
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
# NOTE: the app now reads text with Apple Vision (macOS only), so this image cannot run it on Linux.
# The line that downloaded the RapidOCR models is gone. Host the app from your Mac instead (python run.py --share).

COPY --chown=user helping_eyes ./helping_eyes
WORKDIR /home/user/app/helping_eyes
EXPOSE 7860
CMD ["bash", "cloud/start.sh"]
