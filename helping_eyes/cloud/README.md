---
title: Helping Eyes
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Camera reading assistant - capture text, ask about it
---

# Helping Eyes

Point the camera at a label, letter or book page, capture it, then ask about it by voice or by typing.
Answers come only from the captured text; web lookups happen only after you agree and are marked
"According to the web".

- OCR: RapidOCR (CPU) · language model: Gemini Flash-Lite through an OpenAI-compatible API (set the `LLM_API_KEY` secret) · web search: DuckDuckGo
- Speech recognition and speech output run in the browser (Chrome or Edge recommended).
- Live spoken guidance, automatic capture and a book mode that reads page by page and highlights the word being read.
- Source code: see the project repository. The same app runs on a laptop.
