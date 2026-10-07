#!/usr/bin/env bash
# Start Ollama, make sure the model is present, then start the web server on port 7860.
set -e
ollama serve > /tmp/ollama.log 2>&1 &
for i in $(seq 1 60); do
    curl -s http://localhost:11434/api/version > /dev/null && break
    sleep 1
done
ollama pull "${LLM_MODEL:-qwen2.5:3b-instruct}"   # no download if already in the image
exec uvicorn server:app --host 0.0.0.0 --port 7860
