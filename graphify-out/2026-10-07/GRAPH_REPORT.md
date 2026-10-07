# Graph Report - Helping_Eyes-main  (2026-10-07)

## Corpus Check
- 16 files · ~48,838 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 8 file(s) not represented in the graph (top: (none) 5, .example 1, .css 1)

## Summary
- 445 nodes · 857 edges · 32 communities (10 shown, 22 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 21 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `50ee550e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- live.py
- Helping Eyes — project overview
- page_enhance.py
- text_vision.py
- test_live.py
- server.py
- ambiguous_python_import_0d57e9c79602
- app.js
- classify
- test_server.py
- deploy_space.sh
- book_mode.py
- ambiguous_python_import_c4903fb5325b
- README.md
- start.sh
- ambiguous_python_import_12748383589a
- ambiguous_python_import_15fab9ae6755
- ambiguous_python_import_3a41de599dc0
- ambiguous_python_import_7b0d469555d8
- ambiguous_python_import_8a1e897c40b2
- ambiguous_python_import_a9379122aa73
- ambiguous_python_import_bc0bdfed81b6
- ambiguous_python_import_e493a3900412
- ambiguous_python_import_e619e6356f81
- ambiguous_python_import_eadc95eef474
- google_generativeai
- pil
- queue
- select
- shutil
- speech_recognition
- subprocess

## God Nodes (most connected - your core abstractions)
1. `DocAssistant` - 25 edges
2. `PageLayout` - 17 edges
3. `guide_with()` - 17 edges
4. `LiveGuide` - 15 edges
5. `assess_quality()` - 15 edges
6. `_read_flat_page()` - 14 edges
7. `QualityReport` - 14 edges
8. `TextLine` - 14 edges
9. `Helping Eyes — project overview` - 14 edges
10. `PageTurnDetector` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Book Reading Mode (computer vision pipeline)` --references--> `Book mode overlay screenshot (spine, page numbers, paragraph order)`  [EXTRACTED]
  README.md → docs/book_mode_overlay.jpg
- `Reading position highlight (word-level TTS tracking)` --references--> `Book mode reading highlight screenshot (line/word tracking)`  [EXTRACTED]
  README.md → docs/book_mode_highlight.jpg
- `BookWatcher` --uses--> `PageTurnDetector`  [INFERRED]
  helping_eyes/live.py → helping_eyes/book_mode.py
- `_rows()` --uses--> `TextLine`  [INFERRED]
  helping_eyes/book_mode.py → helping_eyes/text_vision.py
- `split_page_parts()` --uses--> `TextLine`  [INFERRED]
  helping_eyes/book_mode.py → helping_eyes/text_vision.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Book mode reading pipeline stages** — readme_page_turn_detection, readme_spine_detection, readme_reading_order_xycut, readme_reading_position_highlight, readme_continuing_after_move [EXTRACTED 0.95]
- **Helping Eyes core innovations** — readme_question_driven_reading, readme_text_detection_vs_object_detection, readme_grounded_ondevice_answers, readme_handsfree_book_reading [EXTRACTED 0.95]

## Communities (32 total, 22 thin omitted)

### Community 0 - "live.py"
Cohesion: 0.06
Nodes (26): continue_from(), (start, end, normalised word) for each word, ignoring punctuation-only tokens., Compare the page being read with a new view of it. old_pos is how far reading…, _words(), BookReader, BookWatcher, layout_json(), LiveGuide (+18 more)

### Community 1 - "Helping Eyes — project overview"
Cohesion: 0.09
Nodes (24): Book mode reading highlight screenshot (line/word tracking), Book mode overlay screenshot (spine, page numbers, paragraph order), CI health-check workflow (web version), Book Reading Mode (computer vision pipeline), Continuing after the view moves (continue_from), Optional consent-based web lookup (DuckDuckGo), Grounded, on-device answers (innovation), Hands-free book reading (innovation) (+16 more)

### Community 2 - "page_enhance.py"
Cohesion: 0.07
Nodes (60): _analysis_gray(), _apply_shift(), assess_quality(), _best_lag(), binarize(), capture_candidates(), dewarp(), DewarpModel (+52 more)

### Community 3 - "text_vision.py"
Cohesion: 0.14
Nodes (25): foundation, work(), find_text_region(), Box, ndarray, _quad_to_box(), Text finding and reading. Two OCR engines with the same interface: apple…, Find and read every line of text in a BGR image. word_boxes=True also records… (+17 more)

### Community 4 - "test_live.py"
Cohesion: 0.06
Nodes (37): PageInfo, What a printed page says about itself., DocAssistant, DuckDuckGo text search: [{title, href, body}, ...]. Only the query leaves the…, Keeps the captured text plus the conversation about it., New capture: replace the text and forget the previous conversation. facts: page…, Same item, slightly different view: replace the text but keep the conversation., Stop any answer currently being generated. (+29 more)

### Community 5 - "server.py"
Cohesion: 0.05
Nodes (61): BaseModel, calendar, date, datetime, dotenv, fastapi, fastapi_concurrency, fastapi_responses (+53 more)

### Community 7 - "app.js"
Cohesion: 0.10
Nodes (42): answerEl, ask(), autoCapture(), bookTick(), cameraUnavailable(), COLORS, connectLive(), drawBook() (+34 more)

### Community 8 - "classify"
Cohesion: 0.18
Nodes (16): dataclasses, classify(), Command, Spoken and typed commands handled by the application itself, not the model.…, name is one of: stop, repeat, book_off, book_on, new_capture, search_last,…, Decide what one request means. offer_pending is True right after the app asked…, True for plain "read it all" requests, so they skip the model entirely., wants_read_all() (+8 more)

### Community 9 - "test_server.py"
Cohesion: 0.21
Nodes (13): cv2, fastapi_testclient, ask(), label_jpeg(), Tests for server.py through its real HTTP and WebSocket interface. RapidOCR…, spoken(), test_asking_before_capturing(), test_book_page_endpoint_reads_a_page_and_resumes() (+5 more)

### Community 11 - "book_mode.py"
Cohesion: 0.05
Nodes (50): collections, difflib, find_book(), LineSpan, _median(), PageLayout, PageTurnDetector, parse_page_number() (+42 more)

## Knowledge Gaps
- **23 isolated node(s):** `start.sh script`, `video`, `overlay`, `grab`, `statusEl` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 185 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DocAssistant` connect `test_live.py` to `live.py`, `text_vision.py`, `server.py`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `PageLayout` connect `book_mode.py` to `live.py`, `test_live.py`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `PageTurnDetector` connect `book_mode.py` to `live.py`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `DocAssistant` (e.g. with `Session` and `_warm_up()`) actually correct?**
  _`DocAssistant` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `PageLayout` (e.g. with `BookReader` and `layout_json()`) actually correct?**
  _`PageLayout` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `guide_with()` (e.g. with `assess()` and `detect()`) actually correct?**
  _`guide_with()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `LiveGuide` (e.g. with `QualityReport` and `Session`) actually correct?**
  _`LiveGuide` has 2 INFERRED edges - model-reasoned connections that need verification._