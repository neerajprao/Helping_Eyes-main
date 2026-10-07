# Graph Report - Helping_Eyes-main  (2026-10-07)

## Corpus Check
- 15 files · ~45,745 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 8 file(s) not represented in the graph (top: (none) 5, .example 1, .css 1)

## Summary
- 447 nodes · 842 edges · 43 communities (20 shown, 23 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `133dc716`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- LiveGuide
- Helping Eyes — project overview
- ndarray
- text_vision.py
- test_live.py
- server.py
- ambiguous_python_import_0d57e9c79602
- app.js
- doc_assistant.py
- test_server.py
- deploy_space.sh
- vision.py
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
- test_page_enhance.py
- PageLayout
- split_page_parts
- PageTurnDetector
- What each file does
- assess_quality
- warp_page
- _read_flat_page
- read_page
- .label
- vision

## God Nodes (most connected - your core abstractions)
1. `DocAssistant` - 25 edges
2. `guide_with()` - 17 edges
3. `PageLayout` - 15 edges
4. `TextLine` - 14 edges
5. `_read_flat_page()` - 14 edges
6. `LiveGuide` - 14 edges
7. `Helping Eyes — project overview` - 14 edges
8. `make_page()` - 13 edges
9. `assess_quality()` - 13 edges
10. `ask()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Book Reading Mode (computer vision pipeline)` --references--> `Book mode overlay screenshot (spine, page numbers, paragraph order)`  [EXTRACTED]
  README.md → docs/book_mode_overlay.jpg
- `Reading position highlight (word-level TTS tracking)` --references--> `Book mode reading highlight screenshot (line/word tracking)`  [EXTRACTED]
  README.md → docs/book_mode_highlight.jpg
- `Session` --uses--> `DocAssistant`  [INFERRED]
  helping_eyes/server.py → helping_eyes/doc_assistant.py
- `_warm_up()` --uses--> `DocAssistant`  [INFERRED]
  helping_eyes/server.py → helping_eyes/doc_assistant.py
- `_rows()` --uses--> `TextLine`  [INFERRED]
  helping_eyes/vision.py → helping_eyes/text_vision.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Book mode reading pipeline stages** — readme_page_turn_detection, readme_spine_detection, readme_reading_order_xycut, readme_reading_position_highlight, readme_continuing_after_move [EXTRACTED 0.95]
- **Helping Eyes core innovations** — readme_question_driven_reading, readme_text_detection_vs_object_detection, readme_grounded_ondevice_answers, readme_handsfree_book_reading [EXTRACTED 0.95]

## Communities (43 total, 23 thin omitted)

### Community 0 - "LiveGuide"
Cohesion: 0.09
Nodes (12): Everything the server remembers about one browser: the captured item and the…, Session, assess(), BookReader, BookWatcher, LiveGuide, QualityReport, Feed it preview frames with update(). It says what to draw and speak, and when… (+4 more)

### Community 1 - "Helping Eyes — project overview"
Cohesion: 0.09
Nodes (24): Book mode reading highlight screenshot (line/word tracking), Book mode overlay screenshot (spine, page numbers, paragraph order), CI health-check workflow (web version), Book Reading Mode (computer vision pipeline), Continuing after the view moves (continue_from), Optional consent-based web lookup (DuckDuckGo), Grounded, on-device answers (innovation), Hands-free book reading (innovation) (+16 more)

### Community 2 - "ndarray"
Cohesion: 0.16
Nodes (21): test_dewarp_straightens_curved_lines_and_maps_boxes_back(), _apply_shift(), _best_lag(), dewarp(), DewarpModel, enhance_tone(), estimate_dewarp(), _ink() (+13 more)

### Community 3 - "text_vision.py"
Cohesion: 0.13
Nodes (25): cv2, foundation, work(), find_text_region(), Box, ndarray, _quad_to_box(), Text finding and reading. Two OCR engines with the same interface: apple… (+17 more)

### Community 4 - "test_live.py"
Cohesion: 0.06
Nodes (39): DocAssistant, DuckDuckGo text search: [{title, href, body}, ...]. Only the query leaves the…, Keeps the captured text plus the conversation about it., New capture: replace the text and forget the previous conversation. facts: page…, Same item, slightly different view: replace the text but keep the conversation., Stop any answer currently being generated., Load the model into memory so the first question isn't slow., Answer from the captured text, streamed sentence by sentence. Yields READ_ALL… (+31 more)

### Community 5 - "server.py"
Cohesion: 0.07
Nodes (46): BaseModel, collections, dotenv, fastapi, fastapi_concurrency, fastapi_responses, fastapi_staticfiles, FileResponse (+38 more)

### Community 7 - "app.js"
Cohesion: 0.10
Nodes (42): answerEl, ask(), autoCapture(), bookTick(), cameraUnavailable(), COLORS, connectLive(), drawBook() (+34 more)

### Community 8 - "doc_assistant.py"
Cohesion: 0.09
Nodes (33): calendar, dataclasses, date, datetime, classify(), Command, Spoken and typed commands handled by the application itself, not the model.…, name is one of: stop, repeat, book_off, book_on, new_capture, search_last,… (+25 more)

### Community 9 - "test_server.py"
Cohesion: 0.23
Nodes (12): fastapi_testclient, ask(), label_jpeg(), Tests for server.py through its real HTTP and WebSocket interface. RapidOCR…, spoken(), test_asking_before_capturing(), test_book_page_endpoint_reads_a_page_and_resumes(), test_capture_then_answers_from_the_text() (+4 more)

### Community 11 - "vision.py"
Cohesion: 0.24
Nodes (10): difflib, find_book(), Everything that works on the camera picture, in three parts (classic OpenCV /…, Start/end (exclusive) of each run of True values., The open book = the bright (paper) regions: Otsu threshold on a blurred…, How "busy" each x column is: the local standard deviation of brightness,…, Find the gutter (spine) of an open book and return its x position, or None for…, _runs() (+2 more)

### Community 32 - "test_page_enhance.py"
Cohesion: 0.19
Nodes (17): curve(), make_page(), on_desk(), ndarray, Tests for vision.py part A (image quality and page geometry) on synthetic pages…, A white page with ~28 lines of black text., Place the page on a dark desk with its corners at `quad` (perspective)., Bend the lines like a book toward its spine: text at column x moves down by a… (+9 more)

### Community 33 - "PageLayout"
Cohesion: 0.15
Nodes (14): layout_json(), LineSpan, _norm(), PageLayout, Box, Map a layout found on the flattened page back onto the camera frame (in place)., The page layout as JSON, with boxes normalised to 0-1 of the frame., Map a box from the warped image back to the original frame. (+6 more)

### Community 34 - "split_page_parts"
Cohesion: 0.20
Nodes (14): _median(), PageInfo, parse_page_number(), What a printed page says about itself., Split a header/footer line into (page number, remaining title text). Handles…, Group lines that sit side by side (overlapping in y) into rows, top to bottom., Separate the page's header and footer from its body. A header (footer) is a row…, Recursive XY-cut, a classic document layout algorithm, run on the boxes of the… (+6 more)

### Community 35 - "PageTurnDetector"
Cohesion: 0.19
Nodes (5): PageTurnDetector, Watches the camera for a page turn. TURNING something is moving (a hand, a page…, Start over: the first steady page will be read., A small binary 'fingerprint' of the page layout: text lines appear as dark…, True (and remembered) when the page layout differs from the last page read.

### Community 36 - "What each file does"
Cohesion: 0.20
Nodes (9): Everything else, How Helping Eyes works, Putting it on the internet (`helping_eyes/cloud/`), The brain (`helping_eyes/`), The checkers (`helping_eyes/tests/`), The face (`helping_eyes/web/`), The life cycle: from input to output, The same life cycle in words (+1 more)

### Community 37 - "assess_quality"
Cohesion: 0.25
Nodes (8): _analysis_gray(), assess_quality(), binarize(), capture_candidates(), Grey image at a fixed width, optionally cropped to the text box., Blur, glare and exposure of a BGR frame. Pass the text box (when known) so…, Adaptive (local) threshold: black print on white, robust to uneven light., Images to try reading for a single item (label, page, ...): the untouched frame…

### Community 38 - "warp_page"
Cohesion: 0.25
Nodes (8): find_page_quad(), _needs_warp(), _order_corners(), Corners as top-left, top-right, bottom-right, bottom-left., The page outline as 4 corners (top-left, top-right, bottom-right, bottom-left)…, A page that is already square-on and fills the frame doesn't need warping., Flatten the page with a homography. Returns (image, matrix) where matrix maps…, warp_page()

### Community 39 - "_read_flat_page"
Cohesion: 0.25
Nodes (8): Map a box found on the dewarped image back onto the original frame., The same line with its boxes moved dx pixels to the right., Boxes found on a dewarped page image, mapped back onto the camera frame., read_page() on one image: split the spread, OCR each page, order the text (see…, _read_flat_page(), _shift(), _unwarp(), unwarp_box()

### Community 40 - "read_page"
Cohesion: 0.29
Nodes (6): continue_from(), Split the spread, read each page with Apple Vision, separate headers and…, (start, end, normalised word) for each word, ignoring punctuation-only tokens., Compare the page being read with a new view of it. old_pos is how far reading…, read_page(), _words()

### Community 41 - ".label"
Cohesion: 0.33
Nodes (3): Printed page numbers, left to right. A missing number on one side of a spread…, Page 47', 'Pages 46 and 47', or '' when no number is printed., Page facts for the language model, so it can answer 'what page is this?'.

## Knowledge Gaps
- **29 isolated node(s):** `start.sh script`, `video`, `overlay`, `grab`, `statusEl` (+24 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 189 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **23 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DocAssistant` connect `test_live.py` to `doc_assistant.py`, `LiveGuide`, `server.py`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `PageLayout` connect `PageLayout` to `test_live.py`, `_read_flat_page`, `read_page`, `.label`, `vision.py`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `LiveGuide` connect `LiveGuide` to `vision.py`, `test_live.py`, `server.py`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `DocAssistant` (e.g. with `Session` and `_warm_up()`) actually correct?**
  _`DocAssistant` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `guide_with()` (e.g. with `assess()` and `detect()`) actually correct?**
  _`guide_with()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `TextLine` (e.g. with `_rows()` and `split_page_parts()`) actually correct?**
  _`TextLine` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `start.sh script`, `video`, `overlay` to the rest of the system?**
  _29 weakly-connected nodes found - possible documentation gaps or missing edges._