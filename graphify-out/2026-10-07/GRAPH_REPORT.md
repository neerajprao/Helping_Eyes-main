# Graph Report - Helping_Eyes-main  (2026-09-28)

## Corpus Check
- 19 files · ~45,130 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 4, .pptx 1, .example 1)

## Summary
- 620 nodes · 1129 edges · 26 communities (23 shown, 3 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 14 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `50ee550e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- assitant.py
- Helping Eyes: Presentation Script (CS3235 review)
- laptop_version/page_enhance.py
- laptop_version/text_vision.py
- web_version/page_enhance.py
- server.py
- smart_reader.py
- Speaker
- DocAssistant
- DocAssistant
- app.js
- laptop_version/book_mode.py
- web_version/book_mode.py
- ndarray
- ndarray
- split_page_parts
- split_page_parts
- _read_flat_page
- _read_flat_page
- PageLayout
- PageLayout
- find_book
- find_book
- deploy_space.sh
- README.md
- start.sh

## God Nodes (most connected - your core abstractions)
1. `Helping Eyes: Presentation Script (CS3235 review)` - 23 edges
2. `main()` - 17 edges
3. `Helping Eyes — project overview` - 16 edges
4. `Speaker` - 15 edges
5. `DocAssistant` - 13 edges
6. `make_page()` - 13 edges
7. `DocAssistant` - 13 edges
8. `make_page()` - 13 edges
9. `assess_quality()` - 12 edges
10. `estimate_dewarp()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `VizWiz (Bigham et al., UIST 2010)` --semantically_similar_to--> `Helping Eyes — project overview`  [INFERRED] [semantically similar]
  Helping_Eyes_Presentation_Script.pdf → README.md
- `Problem statement (presentation slide 7)` --conceptually_related_to--> `Problem Statement (README)`  [INFERRED]
  Helping_Eyes_Presentation_Script.pdf → README.md
- `Book Reading Mode (computer vision pipeline)` --references--> `Book mode overlay screenshot (spine, page numbers, paragraph order)`  [EXTRACTED]
  README.md → docs/book_mode_overlay.jpg
- `Lewis et al. — retrieval-augmented generation` --conceptually_related_to--> `Optional consent-based web lookup (DuckDuckGo)`  [INFERRED]
  Helping_Eyes_Presentation_Script.pdf → README.md
- `Smith 2007 — Tesseract OCR engine` --conceptually_related_to--> `OCR engine selection: Apple Vision vs RapidOCR`  [INFERRED]
  Helping_Eyes_Presentation_Script.pdf → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Book mode reading pipeline stages** — readme_page_turn_detection, readme_spine_detection, readme_reading_order_xycut, readme_reading_position_highlight, readme_continuing_after_move [EXTRACTED 0.95]
- **Helping Eyes core innovations** — readme_question_driven_reading, readme_text_detection_vs_object_detection, readme_grounded_ondevice_answers, readme_handsfree_book_reading [EXTRACTED 0.95]
- **Shared literature citations between README and presentation script** — readme_ref_gawande2023, readme_ref_smartreader_apsit2025, readme_ref_sharma2014_ocr_tts, helping_eyes_presentation_script_pdf_gawande2023, helping_eyes_presentation_script_pdf_smartreader_apsit2025, helping_eyes_presentation_script_pdf_sharma2014 [INFERRED 0.85]

## Communities (26 total, 3 thin omitted)

### Community 0 - "assitant.py"
Cohesion: 0.06
Nodes (51): ambiguous_python_import_a9379122aa73, calendar, datetime, dotenv, google_generativeai, json, analyze_image(), extract_new_text() (+43 more)

### Community 1 - "Helping Eyes: Presentation Script (CS3235 review)"
Cohesion: 0.05
Nodes (47): Book mode reading highlight screenshot (line/word tracking), Book mode overlay screenshot (spine, page numbers, paragraph order), CI health-check workflow (web version), Cutter & Manduchi — spoken guidance for document capture, Ezaki, Bulacu & Schomaker 2004 — scene text detection, FingerReader (Shilkrot et al., CHI 2015), Gawande et al. 2023 — ML-based TTS device, Gonzalez Penuela et al., CHI 2024 — diary study of MLLM reading app (+39 more)

### Community 2 - "laptop_version/page_enhance.py"
Cohesion: 0.06
Nodes (62): _analysis_gray(), _apply_shift(), assess_quality(), _best_lag(), binarize(), capture_candidates(), dewarp(), DewarpModel (+54 more)

### Community 3 - "laptop_version/text_vision.py"
Cohesion: 0.08
Nodes (46): cv2, dataclasses, foundation, find_text_region(), Box, ndarray, _quad_to_box(), Text finding and reading. Two OCR engines with the same interface: apple… (+38 more)

### Community 4 - "web_version/page_enhance.py"
Cohesion: 0.06
Nodes (62): _analysis_gray(), _apply_shift(), assess_quality(), _best_lag(), binarize(), capture_candidates(), dewarp(), DewarpModel (+54 more)

### Community 5 - "server.py"
Cohesion: 0.08
Nodes (37): ambiguous_python_import_3a41de599dc0, ambiguous_python_import_7b0d469555d8, ambiguous_python_import_8a1e897c40b2, BaseModel, fastapi, fastapi_responses, fastapi_staticfiles, FileResponse (+29 more)

### Community 6 - "smart_reader.py"
Cohesion: 0.06
Nodes (41): ambiguous_python_import_0d57e9c79602, ambiguous_python_import_15fab9ae6755, ambiguous_python_import_bc0bdfed81b6, ambiguous_python_import_eadc95eef474, BackgroundFrameReader, capture_document(), handle_request(), main() (+33 more)

### Community 7 - "Speaker"
Cohesion: 0.11
Nodes (11): _build_command(), Queue text to be spoken (non-blocking). track_from: for text taken from a…, Silence current speech and discard everything queued., Block until nothing is queued or playing. Returns False on timeout., Start `say --interactive` on a pseudo-terminal; returns (process, pty fd)., Read say's redraws until it finishes, updating `position` word by word., Return the TTS command for this OS. Text is always fed via stdin., Queued, interruptible speech. Utterances are spoken one at a time, in order. (+3 more)

### Community 8 - "DocAssistant"
Cohesion: 0.11
Nodes (12): DocAssistant, DuckDuckGo text search: [{title, href, body}, ...]. Only the query leaves the…, Keeps the captured text plus the conversation about it., New capture: replace the text and forget the previous conversation. facts: page…, Same item, slightly different view: replace the text but keep the conversation., Stop any answer currently being generated., Load the model into memory so the first question isn't slow., Answer from the captured text, streamed sentence by sentence. Yields READ_ALL… (+4 more)

### Community 9 - "DocAssistant"
Cohesion: 0.11
Nodes (12): DocAssistant, DuckDuckGo text search: [{title, href, body}, ...]. Only the query leaves the…, Keeps the captured text plus the conversation about it., New capture: replace the text and forget the previous conversation. facts: page…, Same item, slightly different view: replace the text but keep the conversation., Stop any answer currently being generated., Load the model into memory so the first question isn't slow., Answer from the captured text, streamed sentence by sentence. Yields READ_ALL… (+4 more)

### Community 10 - "app.js"
Cohesion: 0.20
Nodes (19): answerEl, ask(), cameraUnavailable(), canvas, capture(), escapeHtml(), handleSpoken(), lastSentences (+11 more)

### Community 11 - "laptop_version/book_mode.py"
Cohesion: 0.14
Nodes (16): ambiguous_python_import_12748383589a, ambiguous_python_import_e619e6356f81, collections, continue_from(), PageInfo, Book reading mode: the computer vision that decides WHEN to read a page, WHERE…, Start/end (exclusive) of each run of True values., How "busy" each x column is: the local standard deviation of brightness,… (+8 more)

### Community 12 - "web_version/book_mode.py"
Cohesion: 0.14
Nodes (16): ambiguous_python_import_c4903fb5325b, ambiguous_python_import_e493a3900412, difflib, continue_from(), PageInfo, Book reading mode: the computer vision that decides WHEN to read a page, WHERE…, Start/end (exclusive) of each run of True values., How "busy" each x column is: the local standard deviation of brightness,… (+8 more)

### Community 13 - "ndarray"
Cohesion: 0.22
Nodes (6): PageTurnDetector, ndarray, True (and remembered) when the page layout differs from the last page read., Watches the camera for a page turn. TURNING something is moving (a hand, a page…, Start over: the first steady page will be read., A small binary 'fingerprint' of the page layout: text lines appear as dark…

### Community 14 - "ndarray"
Cohesion: 0.22
Nodes (6): PageTurnDetector, ndarray, True (and remembered) when the page layout differs from the last page read., Watches the camera for a page turn. TURNING something is moving (a hand, a page…, Start over: the first steady page will be read., A small binary 'fingerprint' of the page layout: text lines appear as dark…

### Community 15 - "split_page_parts"
Cohesion: 0.24
Nodes (13): _median(), parse_page_number(), TextLine, Split a header/footer line into (page number, remaining title text). Handles…, Group lines that sit side by side (overlapping in y) into rows, top to bottom., Separate the page's header and footer from its body. A header (footer) is a row…, Recursive XY-cut, a classic document layout algorithm, run on the boxes of the…, _rows() (+5 more)

### Community 16 - "split_page_parts"
Cohesion: 0.24
Nodes (13): _median(), parse_page_number(), TextLine, Split a header/footer line into (page number, remaining title text). Handles…, Group lines that sit side by side (overlapping in y) into rows, top to bottom., Separate the page's header and footer from its body. A header (footer) is a row…, Recursive XY-cut, a classic document layout algorithm, run on the boxes of the…, _rows() (+5 more)

### Community 17 - "_read_flat_page"
Cohesion: 0.18
Nodes (12): LineSpan, Where one printed line ended up in the page text, and where it is on screen., The same line with its boxes moved dx pixels to the right., Boxes found on a dewarped page image, mapped back onto the camera frame., read_page() on one image: split the spread, OCR each page, order the text (see…, Map a layout found on the flattened page back onto the camera frame (in place)., Split the spread, read each page with Apple Vision, separate headers and…, _read_flat_page() (+4 more)

### Community 18 - "_read_flat_page"
Cohesion: 0.18
Nodes (12): LineSpan, Where one printed line ended up in the page text, and where it is on screen., The same line with its boxes moved dx pixels to the right., Boxes found on a dewarped page image, mapped back onto the camera frame., read_page() on one image: split the spread, OCR each page, order the text (see…, Map a layout found on the flattened page back onto the camera frame (in place)., Split the spread, read each page with Apple Vision, separate headers and…, _read_flat_page() (+4 more)

### Community 19 - "PageLayout"
Cohesion: 0.28
Nodes (5): PageLayout, Everything found in one frame, in full-frame pixel coordinates (for drawing)., Printed page numbers, left to right. A missing number on one side of a spread…, Page 47', 'Pages 46 and 47', or '' when no number is printed., Page facts for the language model, so it can answer 'what page is this?'.

### Community 20 - "PageLayout"
Cohesion: 0.28
Nodes (5): PageLayout, Everything found in one frame, in full-frame pixel coordinates (for drawing)., Printed page numbers, left to right. A missing number on one side of a spread…, Page 47', 'Pages 46 and 47', or '' when no number is printed., Page facts for the language model, so it can answer 'what page is this?'.

### Community 21 - "find_book"
Cohesion: 0.29
Nodes (5): find_book(), Box, The open book = the bright (paper) regions: Otsu threshold on a blurred…, The printed line that contains character `pos` of the page text., Screen box of the word at character `pos` (or the next word on that line).

### Community 22 - "find_book"
Cohesion: 0.29
Nodes (5): find_book(), Box, The open book = the bright (paper) regions: Otsu threshold on a blurred…, The printed line that contains character `pos` of the page text., Screen box of the word at character `pos` (or the next word on that line).

## Knowledge Gaps
- **24 isolated node(s):** `start.sh script`, `video`, `canvas`, `statusEl`, `answerEl` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 264 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Speaker` connect `Speaker` to `assitant.py`, `smart_reader.py`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `DocAssistant` connect `DocAssistant` to `assitant.py`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `DocAssistant` connect `DocAssistant` to `assitant.py`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `main()` (e.g. with `request_worker()` and `typed_listener()`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `start.sh script`, `video`, `canvas` to the rest of the system?**
  _24 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `assitant.py` be split into smaller, more focused modules?**
  _Cohesion score 0.057912457912457915 - nodes in this community are weakly interconnected._
- **Should `Helping Eyes: Presentation Script (CS3235 review)` be split into smaller, more focused modules?**
  _Cohesion score 0.0545790934320074 - nodes in this community are weakly interconnected._