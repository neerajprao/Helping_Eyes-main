#!/usr/bin/env bash
# Start the web server (port 7860, or $PORT when the host sets one).
set -e
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-7860}"
