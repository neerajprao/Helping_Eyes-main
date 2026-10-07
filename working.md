# How Helping Eyes works

Helping Eyes helps a person who cannot see well to read things like medicine boxes, letters and books.
You show the item to the camera. The app tells you how to hold it, takes a photo by itself, and then you
can ask questions like "When does it expire?" or just say "read everything".

It is **one app**. It has two parts:

- **The browser page** (what you see and hear): it uses the camera, the microphone and the speaker.
- **The server** (the brain): it finds the text in the picture, understands what you asked, and writes the answer.

You can run the same app on your own laptop or on the internet (Hugging Face). Only a few settings change.

---

## What each file does

### The brain (`helping_eyes/`)

| File | What it does, in simple words |
|---|---|
| `server.py` | The front desk of the server. It receives camera pictures, photos and questions from the browser and sends back guidance and answers. It also remembers each person's item and conversation for 30 minutes. |
| `vision.py` | All the computer vision (looking at pictures), in three parts, so it is easy to find:<br>**Part A, the photo fixer:** measures blur, glare and darkness, straightens a tilted page, flattens a curved page and brightens shadows, so the reader gets a cleaner picture.<br>**Part B, the book expert:** finds the middle of an open book, splits the two pages, finds page numbers and titles, puts paragraphs in the right reading order, and keeps your place if the book moves a little.<br>**Part C, the coach:** looks at the live camera pictures and decides "move closer", "move left", "too blurry", "hold still" and "now take the photo". In book mode it notices when you turn a page and decides what to read next. |
| `commands.py` | The listener. It decides if what you said is a command ("stop", "next", "book mode", "yes", "no", "look it up") or a question for the AI. |
| `doc_assistant.py` | The answerer. It keeps the text that was captured and talks to an AI chat model over the internet (Gemini Flash-Lite by default; any provider with an OpenAI-style API works). It only answers from the captured text. It works out expiry dates itself, because small AI models get dates wrong. It can search the web, but only if you say yes. |
| `text_vision.py` | The reader. It turns a picture into text (OCR). It uses RapidOCR, which runs on any computer without special hardware. |
| `requirements.txt` | The shopping list of Python packages the app needs. |
| `.env.example` | A sample of the settings. Copy it to `.env` and put your AI key in it (the only required setting); you can also change the model or provider there. |

### The face (`helping_eyes/web/`)

| File | What it does |
|---|---|
| `index.html` | The page layout: the camera picture, the buttons and the answer area. |
| `app.js` | The ears, mouth and eyes in the browser. It opens the camera, sends pictures to the server, speaks the hints and answers out loud, listens to your voice, and draws the boxes and the highlighted word on top of the camera picture. It makes no decisions itself; it does what the server says. |
| `style.css` | How the page looks (colours, big buttons, dark mode). |
| `favicon.svg` | The small icon in the browser tab. |

### Putting it on the internet (`helping_eyes/cloud/`)

| File | What it does |
|---|---|
| `Dockerfile` | The recipe that builds the app into a package that runs on any host that runs Docker (Render, Hugging Face and others). There is no AI model inside; the app calls one over the internet. |
| `start.sh` | The start button inside that package: starts the server. |
| `deploy_space.sh` | One command that gathers the right files and uploads them to Hugging Face. |
| `README.md` | The description card shown on the Hugging Face page. |

### The checkers (`helping_eyes/tests/`)

| File | What it checks |
|---|---|
| `test_page_enhance.py` | Part A of `vision.py` (the photo fixer): tilt, curve, blur, glare and darkness on made-up pages. |
| `test_commands.py` | That "stop", "next", "yes", "book mode" and similar are understood correctly. |
| `test_live.py` | Part C of `vision.py` (the coach): hints, hold-still timing, capture, and the book-page decisions. |
| `test_server.py` | The whole server for real: reading a picture of a label, answering, the live stream, and book pages. |

### Everything else

| File or folder | What it does |
|---|---|
| `README.md` | The full project report: problem, design, how to run it, results, limits. |
| `working.md` | This file. |
| `requirements.txt` (top level) | Just points to the one inside `helping_eyes/`, so you can install from the top folder. |
| `docs/` | Pictures used in the README: the architecture diagram and book-mode screenshots. `make_architecture.py` redraws the diagram (`python docs/make_architecture.py`) whenever the architecture changes. |
| `presentation/` | The slides, the narrated video, the script and the course requirements document. |
| `.github/workflows/check.yml` | Makes GitHub run all the tests automatically whenever code is pushed. |
| `graphify-out/` | A map of the code made by a helper tool (graphify). Not part of the app, and ignored by git, so it is never committed. |
| `.gitignore` | Tells git which files to ignore: your private `.env`, the `.venv` folder, `__pycache__` and `graphify-out/`. |

---

## The life cycle: from input to output

```mermaid
flowchart TD
    CAM[/"INPUT 1: camera pictures"/]
    VOICE[/"INPUT 2: your voice or typing"/]

    CAM --> COACH["vision.py, part C (the coach)<br/>looks for text, checks blur and glare"]
    COACH -->|not ready yet| HINT["Hint: move closer, move left,<br/>hold still"]
    COACH -->|normal mode: text is steady| PHOTO["Full photo"]
    COACH -->|book mode: page was turned| BOOKPHOTO["Full photo of the page"]

    PHOTO --> FIX["vision.py, part A (photo fixer)<br/>fix tilt, glare, shadows"]
    FIX --> READ["text_vision.py<br/>picture to text"]
    READ --> SAVED[("Captured text<br/>saved for questions")]

    BOOKPHOTO --> BOOK["vision.py, part B (book expert)<br/>split the two pages, find page number,<br/>put paragraphs in reading order"]
    BOOK -.uses.-> FIX
    BOOK -.uses.-> READ
    BOOK --> PAGE["Page text + where each word is"]
    PAGE --> SAVED

    VOICE --> STT["Browser: speech to text"]
    STT --> CMD["commands.py<br/>command or question?"]
    CMD -->|command: stop, repeat,<br/>next, book mode| ACT["Do it"]
    CMD -->|question| AI["doc_assistant.py + AI chat model<br/>answers only from the saved text"]
    SAVED --> AI
    AI -->|answer is in the text| ANSWER["Answer, sentence by sentence"]
    AI -->|not in the text and you say yes| WEB["DuckDuckGo search,<br/>then the AI answers from the results"]
    WEB --> ANSWER

    HINT --> OUT
    ACT --> OUT
    ANSWER --> OUT
    PAGE -->|read the page| OUT
    OUT[/"OUTPUT: spoken aloud + shown on screen<br/>(book mode also highlights the word being read)"/]
```

### The same life cycle in words

1. **Input:** the camera sends small pictures all the time, and you speak or type.
2. **Coaching:** The coach (`vision.py`, part C) looks at each picture. If the text is not ready, you hear a hint ("Move closer"). When it is steady, a full photo is taken. In book mode, a full photo is taken each time a page was turned.
3. **Getting the text:**
   - A normal photo is cleaned by the photo fixer (`vision.py`, part A), then turned into text by `text_vision.py`, and the text is saved.
   - A book photo goes through the book expert (`vision.py`, part B), which splits the pages, finds the page number and puts the paragraphs in order (using the photo fixer and the reader). The page is read out straight away and also saved.
4. **Understanding you:** your voice becomes text in the browser. `commands.py` decides if it is a command ("stop", "next") or a question.
5. **Answering:** commands are just done. Questions go to `doc_assistant.py`, which asks the AI chat model to answer using only the saved text. If the text does not have the answer, it offers a web search and searches only if you say yes.
6. **Output:** everything comes back to the browser, which speaks it aloud and shows it on screen. While a book page is being read, the word being spoken is highlighted.
