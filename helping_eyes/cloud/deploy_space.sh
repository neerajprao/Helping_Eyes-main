#!/usr/bin/env bash
# Deploy the web version to a Hugging Face Space (free CPU basic hardware).
#
# Usage:
#   HF_TOKEN=hf_xxx ./cloud/deploy_space.sh <hf-username> [space-name]
#   ./cloud/deploy_space.sh --build-only <output-dir>     # assemble the Space folder only
#
# Create the Space once on huggingface.co (New Space -> SDK: Docker -> hardware: CPU basic),
# and a write token under Settings -> Access Tokens.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

assemble() {
    local out="$1"
    rm -rf "$out" && mkdir -p "$out"
    cp "$ROOT"/{server.py,commands.py,vision.py,doc_assistant.py,text_vision.py,requirements.txt} "$out/"
    cp -R "$ROOT/web" "$out/web"
    cp "$ROOT"/cloud/{Dockerfile,start.sh,README.md} "$out/"
    find "$out" -name "__pycache__" -prune -exec rm -rf {} +
}

if [[ "${1:-}" == "--build-only" ]]; then
    assemble "${2:?output directory}"
    echo "Space folder assembled in $2"
    exit 0
fi

USER_NAME="${1:?Hugging Face username}"
SPACE="${2:-helping-eyes}"
: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face write token}"

BUILD="$(mktemp -d)"
assemble "$BUILD"
cd "$BUILD"
git init -q -b main
git add .
git -c user.name="$USER_NAME" -c user.email="$USER_NAME@users.noreply.huggingface.co" commit -q -m "Deploy Helping Eyes web version"
git push --force "https://$USER_NAME:$HF_TOKEN@huggingface.co/spaces/$USER_NAME/$SPACE" main
echo "Pushed. The Space builds in a few minutes: https://huggingface.co/spaces/$USER_NAME/$SPACE"
echo "Public URL: https://$USER_NAME-$SPACE.hf.space"
