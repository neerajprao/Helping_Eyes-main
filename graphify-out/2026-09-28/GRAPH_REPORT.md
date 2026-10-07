# Graph Report - Helping_Eyes-main  (2026-09-28)

## Corpus Check
- 14 files · ~35,000 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 4, .example 1, .pptx 1)

## Summary
- 415 nodes · 801 edges · 14 communities (12 shown, 2 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 21 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `50ee550e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- smart_reader.py
- Helping Eyes: Presentation Script (CS3235 review)
- _read_flat_page
- text_vision.py
- DocAssistant
- search
- PageTurnDetector
- Speaker
- app.js
- BackgroundFrameReader
- page_enhance.py
- deploy_space.sh
- start.sh
- main

## God Nodes (most connected - your core abstractions)
1. `main()` - 23 edges
2. `Helping Eyes: Presentation Script (CS3235 review)` - 23 edges
3. `DocAssistant` - 21 edges
4. `Helping Eyes — project overview` - 17 edges
5. `assess_quality()` - 16 edges
6. `Speaker` - 15 edges
7. `_read_flat_page()` - 14 edges
8. `make_page()` - 13 edges
9. `PageTurnDetector` - 12 edges
10. `read_page()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `VizWiz (Bigham et al., UIST 2010)` --semantically_similar_to--> `Helping Eyes — project overview`  [INFERRED] [semantically similar]
  Helping_Eyes_Presentation_Script.pdf → README.md
- `Smith 2007 — Tesseract OCR engine` --conceptually_related_to--> `OCR engine selection: Apple Vision vs RapidOCR`  [INFERRED]
  Helping_Eyes_Presentation_Script.pdf → README.md
- `Problem statement (presentation slide 7)` --conceptually_related_to--> `Problem Statement (README)`  [INFERRED]
  Helping_Eyes_Presentation_Script.pdf → README.md
- `Lewis et al. — retrieval-augmented generation` --conceptually_related_to--> `Optional consent-based web lookup (DuckDuckGo)`  [INFERRED]
  Helping_Eyes_Presentation_Script.pdf → README.md
- `Du et al. — PP-OCR lightweight OCR system` --conceptually_related_to--> `OCR engine selection: Apple Vision vs RapidOCR`  [EXTRACTED]
  Helping_Eyes_Presentation_Script.pdf → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Book mode reading pipeline stages** — readme_page_turn_detection, readme_spine_detection, readme_reading_order_xycut, readme_reading_position_highlight, readme_continuing_after_move [EXTRACTED 0.95]
- **Helping Eyes core innovations** — readme_question_driven_reading, readme_text_detection_vs_object_detection, readme_grounded_ondevice_answers, readme_handsfree_book_reading [EXTRACTED 0.95]
- **Shared literature citations between README and presentation script** — readme_ref_gawande2023, readme_ref_smartreader_apsit2025, readme_ref_sharma2014_ocr_tts, helping_eyes_presentation_script_pdf_gawande2023, helping_eyes_presentation_script_pdf_smartreader_apsit2025, helping_eyes_presentation_script_pdf_sharma2014 [INFERRED 0.85]

## Communities (14 total, 2 thin omitted)

### Community 0 - "smart_reader.py"
Cohesion: 0.06
Nodes (56): analyze_image(), extract_new_text(), main(), normalize_text(), speak(), text_similarity(), continue_from(), Book reading mode: the computer vision that decides WHEN to read a page, WHERE… (+48 more)

### Community 1 - "Helping Eyes: Presentation Script (CS3235 review)"
Cohesion: 0.05
Nodes (52): Hugging Face Space README (Helping Eyes web version), Book mode reading highlight screenshot (line/word tracking), Book mode overlay screenshot (spine, page numbers, paragraph order), CI health-check workflow (web version), Cutter & Manduchi — spoken guidance for document capture, Ezaki, Bulacu & Schomaker 2004 — scene text detection, FingerReader (Shilkrot et al., CHI 2015), Gawande et al. 2023 — ML-based TTS device (+44 more)

### Community 2 - "_read_flat_page"
Cohesion: 0.07
Nodes (34): LineSpan, _median(), PageInfo, PageLayout, parse_page_number(), Box, What a printed page says about itself., Split a header/footer line into (page number, remaining title text). Handles… (+26 more)

### Community 3 - "text_vision.py"
Cohesion: 0.15
Nodes (24): dataclasses, foundation, objc, quartz, find_text_region(), Box, ndarray, _quad_to_box() (+16 more)

### Community 4 - "DocAssistant"
Cohesion: 0.07
Nodes (23): date, DocAssistant, expiry_answer(), expiry_checks(), expiry_dates(), (printed text, last valid day) for every expiry date found in the text., One line per expiry date found, saying whether it has passed., Direct spoken answer to an expiry question, or None when no expiry date is… (+15 more)

### Community 5 - "search"
Cohesion: 0.20
Nodes (16): BaseModel, post, Request, ask(), capture(), _get_session(), Question, _rate_limit() (+8 more)

### Community 6 - "PageTurnDetector"
Cohesion: 0.13
Nodes (14): find_book(), PageTurnDetector, ndarray, True (and remembered) when the page layout differs from the last page read., Start/end (exclusive) of each run of True values., The open book = the bright (paper) regions: Otsu threshold on a blurred…, How "busy" each x column is: the local standard deviation of brightness,…, Find the gutter (spine) of an open book and return its x position, or None for… (+6 more)

### Community 7 - "Speaker"
Cohesion: 0.11
Nodes (11): Popen, _build_command(), Queue text to be spoken (non-blocking). track_from: for text taken from a…, Silence current speech and discard everything queued., Block until nothing is queued or playing. Returns False on timeout., Start `say --interactive` on a pseudo-terminal; returns (process, pty fd)., Read say's redraws until it finishes, updating `position` word by word., Return the TTS command for this OS. Text is always fed via stdin. (+3 more)

### Community 8 - "app.js"
Cohesion: 0.20
Nodes (19): answerEl, ask(), cameraUnavailable(), canvas, capture(), escapeHtml(), handleSpoken(), lastSentences (+11 more)

### Community 9 - "BackgroundFrameReader"
Cohesion: 0.23
Nodes (6): BackgroundFrameReader, ndarray, Continuously reads frames in background to prevent buffer overflow Stores…, Continuously read frames in background, Get latest frame and its number (0, None when nothing yet), Stop the background reader

### Community 10 - "page_enhance.py"
Cohesion: 0.06
Nodes (62): _analysis_gray(), _apply_shift(), assess_quality(), _best_lag(), binarize(), capture_candidates(), dewarp(), DewarpModel (+54 more)

### Community 13 - "main"
Cohesion: 0.09
Nodes (29): capture_document(), handle_request(), main(), coach_quality(), handle_key(), Search the web for the question and speak the answer, Switch between normal capture and book reading mode, (start, chunk) pieces of the page for tracked speech, beginning at character… (+21 more)

## Knowledge Gaps
- **24 isolated node(s):** `start.sh script`, `video`, `canvas`, `statusEl`, `answerEl` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 172 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DocAssistant` connect `DocAssistant` to `smart_reader.py`, `search`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `Speaker` connect `Speaker` to `smart_reader.py`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `main()` connect `main` to `smart_reader.py`, `_read_flat_page`, `text_vision.py`, `PageTurnDetector`, `BackgroundFrameReader`, `page_enhance.py`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `main()` (e.g. with `request_worker()` and `typed_listener()`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `DocAssistant` (e.g. with `search()` and `Session`) actually correct?**
  _`DocAssistant` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `start.sh script`, `video`, `canvas` to the rest of the system?**
  _24 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `smart_reader.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06292966684294024 - nodes in this community are weakly interconnected._