// Helping Eyes: the browser side.
// The browser owns the camera, the microphone and the speaker and draws the overlay.
// Every decision (guidance hints, when to capture, page turns, what a request means,
// the answers) is made by the server (server.py, live.py, commands.py).

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), grab = $("canvas");
const statusEl = $("status"), answerEl = $("answer");

const sid = newId();          // identifies this browser to the server (its item, conversation, book)
let mode = "normal";          // "normal" (one item) or "book" (page by page)
let thinking = false;         // an answer is being generated
let controller = null;        // aborts the current answer stream
let lastSentences = [];

function newId() {
  const bytes = new Uint8Array(16);
  (window.crypto || window.msCrypto).getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}
function setStatus(text) { statusEl.textContent = text; }
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ================================================================ speech out
// Every word is spoken by Google's Gemini voice (server /api/tts). There is no browser voice.
const speechQueue = [];       // {text, el, start, audio}: waiting to be spoken
let speakingNow = null;
let speechPos = null;         // book mode: character of the page being spoken right now
let lastSpeechActivity = 0;   // performance.now() when speech last started or ended (echo guard)
let player = null;            // the <audio> element that is playing now
const PREFETCH = 3;           // sentences fetched ahead, so the voice does not pause between them
const TRIES = 4;              // attempts per sentence (the free speech quota can say "wait a moment")

const isSpeaking = () => !!speakingNow || speechQueue.length > 0;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Ask the server for the audio of one sentence; resolves to an object URL, or null when it kept failing
function fetchAudio(item) {
  if (item.audio) return item.audio;
  item.audio = (async () => {
    for (let attempt = 1; attempt <= TRIES; attempt++) {
      const slow = new AbortController();
      const giveUp = setTimeout(() => slow.abort(), 45000);
      try {
        const res = await fetch("/api/tts", { method: "POST", headers: { "Content-Type": "application/json" },
                                              body: JSON.stringify({ text: item.text }), signal: slow.signal });
        if (!res.ok) throw new Error("HTTP " + res.status);
        return URL.createObjectURL(await res.blob());
      } catch (err) {
        if (attempt === TRIES) return null;
        await sleep(1500 * attempt);
      } finally {
        clearTimeout(giveUp);
      }
    }
    return null;
  })();
  return item.audio;
}

function prefetch() {
  speechQueue.slice(0, PREFETCH).forEach(fetchAudio);
}

// start: for a piece of a book page, its offset in the page text (enables the reading position)
function say(text, { el = null, start = null } = {}) {
  if (!text || !text.trim()) return;
  if (start !== null) start += text.length - text.trimStart().length;
  speechQueue.push({ text: text.trim(), el, start, audio: null });
  lastSpeechActivity = performance.now();
  if (recognizer) recognizer.abort();      // the microphone must not hear the app
  prefetch();
  if (!speakingNow) speakNext();
}

// Mark the word at charIndex in the sentence being spoken, and move the book reading position
function markWord(item, charIndex) {
  if (item.start !== null) speechPos = item.start + charIndex;
  if (!item.el) return;
  const t = item.text;
  const end = t.slice(charIndex).search(/\s|$/) + charIndex;
  item.el.innerHTML = escapeHtml(t.slice(0, charIndex)) + "<mark>" + escapeHtml(t.slice(charIndex, end)) +
    "</mark>" + escapeHtml(t.slice(end));
}

function speakNext() {
  const item = speechQueue.shift();
  if (!item) {
    speakingNow = null;
    speechPos = null;
    lastSpeechActivity = performance.now();
    return;
  }
  speakingNow = item;
  prefetch();
  const finished = () => {
    if (speakingNow !== item) return;      // a stale event from a cancelled sentence
    if (item.el) item.el.textContent = item.text;
    if (item.start !== null) speechPos = item.start + item.text.length;   // keep the reading position moving
    if (item.url) URL.revokeObjectURL(item.url);
    player = null;
    speakNext();
  };
  fetchAudio(item).then((url) => {
    if (speakingNow !== item) { if (url) URL.revokeObjectURL(url); return; }      // stopped while waiting
    if (url) return playAudio(item, url, finished);
    setStatus("Google's voice is not available right now (is LLM_API_KEY set, or is the free speech limit used up?). The answer is on screen.");
    finished();
  });
}

function playAudio(item, url, finished) {
  item.url = url;
  const audio = new Audio(url);
  player = audio;
  let lastIndex = -1;
  audio.onplay = () => { if (item.start !== null) speechPos = item.start; };
  audio.ontimeupdate = () => {
    if (!audio.duration) return;
    // The server gives no word times: estimate the word from how far through the audio we are
    let index = Math.min(item.text.length - 1, Math.floor((audio.currentTime / audio.duration) * item.text.length));
    while (index > 0 && !/\s/.test(item.text[index - 1])) index--;
    if (index !== lastIndex) { lastIndex = index; markWord(item, index); }
  };
  // Never freeze the app if the audio neither plays nor ends
  const watchdog = setTimeout(() => done(), 8000 + item.text.length * 150);
  const done = () => { clearTimeout(watchdog); audio.onended = audio.onerror = null; audio.pause(); finished(); };
  audio.onended = audio.onerror = done;
  audio.play().catch(done);
}

// Stop talking (and keep any answer that is still arriving)
function silence() {
  speechQueue.length = 0;
  speakingNow = null;
  speechPos = null;
  lastSpeechActivity = performance.now();
  if (player) { player.onended = player.onerror = null; player.pause(); player = null; }
  answerEl.querySelectorAll("p").forEach((p) => { if (p.querySelector("mark")) p.textContent = p.textContent; });
}

// Stop talking and stop the answer that is being generated
function stopSpeaking() {
  silence();
  if (controller) { controller.abort(); controller = null; }
  thinking = false;
}

// ================================================================ camera
let cameraReady = false;

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return cameraUnavailable("This browser has no camera access. Use Upload photo instead.");
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false,
    });
    video.srcObject = stream;
    video.addEventListener("loadedmetadata", () => { cameraReady = true; sizeOverlay(); connectLive(); });
    setStatus("Camera ready. Show something to read.");
    say("System online. Show me something to read.");
  } catch (err) {
    cameraUnavailable("Camera not available (" + err.name + "). Use Upload photo instead.");
  }
}

function cameraUnavailable(message) {
  video.hidden = true;
  overlay.hidden = true;
  $("camera-message").hidden = false;
  $("camera-message").textContent = message;
  $("capture").disabled = true;
  setStatus(message);
}

// A JPEG of the current camera frame, scaled so its longer side is at most maxSide
function frameBlob(maxSide, quality) {
  return new Promise((resolve) => {
    const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
    grab.width = Math.round(video.videoWidth * scale);
    grab.height = Math.round(video.videoHeight * scale);
    grab.getContext("2d").drawImage(video, 0, 0, grab.width, grab.height);
    grab.toBlob(resolve, "image/jpeg", quality);
  });
}

// ================================================================ live stream (guidance, page turns)
let ws = null;
let paused = false;           // frames are not sent (a capture or a page is being read)
let awaiting = false;         // a frame was sent and its reply has not arrived
let sentAt = 0;
let guide = null;             // latest reply, normal mode
let bookState = null;         // latest reply, book mode

function connectLive() {
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live?sid=${sid}`;
  ws = new WebSocket(url);
  ws.onopen = () => { awaiting = false; ws.send(JSON.stringify({ type: "mode", mode })); };
  ws.onmessage = (e) => { awaiting = false; onLiveReply(JSON.parse(e.data)); };
  ws.onclose = () => { ws = null; awaiting = false; setTimeout(connectLive, 1500); };
  ws.onerror = () => ws && ws.close();
}

function wsSend(message) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
}

async function liveLoop() {
  const stuck = awaiting && performance.now() - sentAt > 8000;   // a lost reply must not stall the loop
  // Normal mode looks for text only while the app is quiet; book mode keeps watching for page turns
  const wanted = mode === "book" || !(isSpeaking() || thinking);
  if (cameraReady && ws && ws.readyState === WebSocket.OPEN && (!awaiting || stuck) && !paused && wanted && !document.hidden) {
    awaiting = true;
    sentAt = performance.now();
    const blob = await frameBlob(960, 0.7);
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(blob); else awaiting = false;
  } else if (mode === "book" && bookState) {
    bookTick(bookState, false);       // keep the reading bookkeeping going while no frame is in flight
  }
  setTimeout(liveLoop, 100);
}

function onLiveReply(reply) {
  if (reply.type === "guide" && mode === "normal") {
    guide = reply;
    if (reply.speak) say(reply.speak);
    if (reply.capture) autoCapture();
  } else if (reply.type === "book" && mode === "book") {
    bookState = reply;
    bookTick(reply, true);
  }
}

// What went wrong with a request, in words. The server answers errors with JSON ({"detail": ...});
// when the host itself is down, restarting or out of memory it answers with an HTML page instead.
async function failure(res) {
  try {
    const detail = (await res.json()).detail;
    if (detail) return detail;
  } catch (err) { /* not JSON: fall through */ }
  if (res.status === 502 || res.status === 503 || res.status === 504 || res.status === 0) {
    return "The server is not responding. It may be starting up, or it ran out of memory and is restarting. Wait a minute and try again.";
  }
  return "The server answered with an error (" + res.status + "). Try again.";
}

// ================================================================ capture (normal mode)
function showCaptured(data) {
  $("captured").hidden = false;
  $("captured-text").textContent = data.text;
  $("page-label").textContent = [data.label, data.running_title].filter(Boolean).join(" · ");
}

async function postCapture(blob, book) {
  const form = new FormData();
  form.append("image", blob, "capture.jpg");
  form.append("sid", sid);
  form.append("book", book ? "true" : "false");
  const res = await fetch("/api/capture", { method: "POST", body: form });
  if (!res.ok) throw new Error(await failure(res));
  return res.json();
}

// The server said the item is steady and in view: take the full-size photo
async function autoCapture() {
  paused = true;
  setStatus("Capturing…");
  let ok = false;
  try {
    const data = await postCapture(await frameBlob(1920, 0.92), false);
    ok = data.ok;
    if (ok) {
      stopSpeaking();
      showCaptured(data);
      setStatus(`Captured in ${data.seconds} s. ` + data.message);
      say(data.message);
    } else {
      setStatus(data.message);
    }
  } catch (err) {
    setStatus("Could not read the photo: " + err.message);
    say("Could not read the text. Try again.");
  }
  wsSend({ type: "capture_result", ok });
  paused = false;
}

// The Capture now button: take a photo immediately (normal mode) or read the page in view (book mode)
$("capture").addEventListener("click", async () => {
  if (!video.videoWidth) return setStatus("The camera is not ready yet.");
  if (mode === "book") return handleBookEvent("changed");
  paused = true;
  stopSpeaking();
  setStatus("Reading the text…");
  say("Reading.");
  let ok = false;
  try {
    const data = await postCapture(await frameBlob(1920, 0.92), false);
    ok = data.ok;
    stopSpeaking();
    setStatus(ok ? `Captured in ${data.seconds} s. ` + data.message : data.message);
    if (ok) showCaptured(data);
    say(data.message);
  } catch (err) {
    setStatus("Could not read the photo: " + err.message);
    say("Sorry, I could not read the photo.");
  }
  wsSend({ type: "capture_result", ok });
  paused = false;
});

// An uploaded photo (for browsers without camera access)
$("upload").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  stopSpeaking();
  setStatus("Reading the text…");
  say("Reading.");
  try {
    const data = await postCapture(file, $("book").checked);
    stopSpeaking();
    if (!data.ok) { setStatus(data.message); return say(data.message); }
    showCaptured(data);
    const intro = [data.label, data.running_title].filter(Boolean).join(". ");
    setStatus(`Captured in ${data.seconds} s. ` + data.message);
    say((intro ? intro + ". " : "") + data.message);
    $("question").focus();
  } catch (err) {
    setStatus("Could not read the photo: " + err.message);
    say("Sorry, I could not read the photo.");
  }
});

// ================================================================ book mode
let pageText = "";            // text of the page being read
let pageLayout = null;        // where its lines and words are on the camera frame
let pageLabel = "";
let readPos = 0;              // how far reading has got in pageText (characters)
let lastChunkStart = 0;       // start of the page's last spoken piece
let reading = false;          // the page is being read aloud
let turnPrompted = true;      // "Turn the page." already said for this page
let bookBusy = false;

function resetBook() {
  pageText = ""; pageLayout = null; pageLabel = ""; readPos = 0; lastChunkStart = 0;
  reading = false; turnPrompted = true; bookState = null;
}

function setMode(next) {
  if (next === mode) return;
  mode = next;
  guide = null;
  resetBook();
  $("bookmode").setAttribute("aria-pressed", String(mode === "book"));
  $("capture").firstChild.textContent = mode === "book" ? "Read this page " : "Capture now ";
  wsSend({ type: "mode", mode });
}

// Called for every book-mode reply from the server (about ten times a second)
function bookTick(reply, fresh) {
  if (speechPos !== null) readPos = Math.max(readPos, speechPos);
  if (reading && !isSpeaking()) {
    reading = false;
    if (readPos >= lastChunkStart) readPos = pageText.length;       // read to the end
  }
  if (fresh && reply.event && !bookBusy) return handleBookEvent(reply.event);
  if (pageText && !turnPrompted && !bookBusy && !(isSpeaking() || thinking)) {
    say("Turn the page.");
    turnPrompted = true;
  }
}

function speakChunks(chunks) {
  for (const c of chunks) { say(c.text, { start: c.start }); lastChunkStart = c.start; }
  reading = chunks.length > 0;
}

// The page view became steady (a page turn or the view moved): read it on the server, then speak
async function handleBookEvent(event) {
  if (bookBusy) return;
  bookBusy = true;
  paused = true;
  setStatus("Reading the page…");
  try {
    const form = new FormData();
    form.append("image", await frameBlob(1600, 0.9), "page.jpg");
    form.append("sid", sid);
    form.append("event", event);
    form.append("read_pos", String(Math.max(readPos, speechPos ?? 0)));
    form.append("reading", String(reading));
    form.append("thinking", String(thinking));
    const res = await fetch("/api/book/page", { method: "POST", body: form });
    if (!res.ok) throw new Error(await failure(res));
    const r = await res.json();
    if (mode !== "book") return;

    if (r.action === "new" || r.action === "resume") {
      pageText = r.text; pageLayout = r.layout; pageLabel = r.label || "";
      answerEl.innerHTML = "";
      $("offer").hidden = true;
      showCaptured({ text: r.text, label: r.label });
      stopSpeaking();
      if (r.action === "new") {
        if (r.intro) say(r.intro);              // the page number and running title, then the page
        readPos = 0;
        setStatus(`${r.label || "Page"} read in ${r.seconds} s.`);
      } else {
        readPos = r.from_pos;
        setStatus("Same words in view: continuing.");
      }
      speakChunks(r.chunks);
      turnPrompted = false;
    } else if (r.action === "nothing_new") {
      pageText = r.text; pageLayout = r.layout; pageLabel = r.label || "";
      readPos = pageText.length;
      setStatus("Same words in view: nothing new to read.");
    } else if (r.action === "no_text") {
      say("I can't see any text on this page.");
      setStatus("No text on this page.");
    }
    // "keep": the same page barely moved while reading: nothing changes
  } catch (err) {
    setStatus("Could not read the page: " + err.message);
  } finally {
    bookBusy = false;
    paused = false;
  }
}

// ================================================================ ask (every request goes to the server)
// Show the request (spoken or typed) above the answer, so the user can see what was understood
function showYouSaid(text) {
  const el = $("you-said");
  el.textContent = "You said: " + text;
  el.hidden = false;
}

async function ask(text) {
  text = (text || "").trim();
  if (!text) return;
  stopSpeaking();
  $("offer").hidden = true;
  showYouSaid(text);
  answerEl.innerHTML = "";
  lastSentences = [];
  setStatus("One moment…");
  const mine = new AbortController();
  controller = mine;
  thinking = true;
  const started = performance.now();
  let firstShown = false;
  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sid, question: text }), signal: mine.signal,
    });
    if (!res.ok) throw new Error(await failure(res));
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (line) onAskEvent(JSON.parse(line), started, () => firstShown, () => { firstShown = true; });
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      setStatus("Something went wrong: " + err.message);
      say("Sorry, something went wrong.");
    }
  } finally {
    if (controller === mine) { controller = null; thinking = false; }
  }
}

function onAskEvent(ev, started, wasShown, markShown) {
  if (ev.type === "sentence") {
    if (!wasShown()) {
      markShown();
      setStatus(`Answering (first sentence after ${((performance.now() - started) / 1000).toFixed(1)} s)…`);
    }
    const p = document.createElement("p");
    p.textContent = ev.text;
    if (ev.source === "web") p.className = "web";
    answerEl.appendChild(p);
    lastSentences.push(ev.text);
    say(ev.text, { el: p });
  } else if (ev.type === "command") {
    if (ev.name === "stop") silence();
    else if (ev.name === "book_on") setMode("book");
    else if (ev.name === "book_off") setMode("normal");
    else if (ev.name === "new_capture") { guide = null; resetBook(); }
  } else if (ev.type === "offer_search") {
    $("offer").hidden = false;
    say(ev.text);
  } else if (ev.type === "done") {
    setStatus(ev.seconds !== undefined ? `Answered in ${ev.seconds} s.` : "Done.");
  }
}

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  ask($("question").value);
  $("question").value = "";
});
$("read-all").addEventListener("click", () => ask("read everything"));
$("repeat").addEventListener("click", () => ask("repeat"));
$("stop").addEventListener("click", stopSpeaking);
$("newitem").addEventListener("click", () => ask("next"));
$("bookmode").addEventListener("click", () => ask(mode === "book" ? "normal mode" : "book mode"));
$("offer-yes").addEventListener("click", () => ask("yes"));
$("offer-no").addEventListener("click", () => { $("offer").hidden = true; ask("no"); });

// ================================================================ speech in (hands-free)
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognizer = null;
let listening = false;        // the user turned listening on
let micFailures = 0;          // speech errors in a row that are not just "nobody spoke"
const ECHO_GUARD = 700;       // ms after the app stops talking before the microphone listens

// What each speech error means, in words the user can act on
const MIC_HELP = {
  "network": "The browser could not reach its speech service. Listening needs an internet connection and does not work in some browsers, for example Brave. Type your question instead.",
  "audio-capture": "No microphone was found. Plug one in or choose it in the browser settings, or type your question instead.",
  "not-allowed": "Microphone access was denied. Allow it with the icon in the address bar, then press Listen again, or type your question instead.",
  "language-not-supported": "This browser cannot recognise English speech. Type your question instead.",
};
MIC_HELP["service-not-allowed"] = MIC_HELP["not-allowed"];

function setListening(on) {
  listening = on;
  micFailures = 0;
  $("mic").setAttribute("aria-pressed", String(on));
  $("mic").firstChild.textContent = on ? "Stop listening " : "Listen ";
  setStatus(on ? "Listening. Ask a question, or say a command." : "Listening is off.");
  if (!on && recognizer) recognizer.abort();
}

// Listening cannot work: switch it off and say why (spoken too, for someone who cannot read the screen)
function micProblem(message) {
  setListening(false);
  setStatus(message);
  say(message);
}

// Ask for the microphone now, from the button press, so the browser's permission question appears at once
async function enableListening() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    setStatus("Asking the browser for the microphone…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());       // only checking: the speech engine opens it itself
    } catch (err) {
      return micProblem(MIC_HELP[err.name === "NotFoundError" ? "audio-capture" : "not-allowed"]);
    }
  }
  setListening(true);
}

if (!Recognition) {
  $("mic").disabled = true;
  $("mic").title = "Speech recognition is not supported in this browser (use Chrome or Edge). Type instead.";
} else {
  $("mic").addEventListener("click", () => (listening ? setListening(false) : enableListening()));
}

// Listen only while the app is quiet, so it does not hear its own voice
function listenTick() {
  if (!listening || recognizer || thinking || isSpeaking()) return;
  if (performance.now() - lastSpeechActivity < ECHO_GUARD) return;
  const startedAt = performance.now();
  const r = new Recognition();
  recognizer = r;
  r.lang = "en-US";
  r.interimResults = true;                                // show words as they are heard: proof the microphone works
  r.maxAlternatives = 1;
  r.onresult = (e) => {
    if (lastSpeechActivity > startedAt) return;           // the app started talking: that was its own voice
    const result = e.results[e.results.length - 1];
    const text = result[0].transcript;
    if (!result.isFinal) { showYouSaid(text + "…"); return setStatus("Hearing: " + text + "…"); }
    micFailures = 0;
    setStatus("Heard: " + text);
    ask(text);
  };
  r.onerror = (e) => {
    if (e.error === "aborted" || e.error === "no-speech") return;      // we stopped it, or nobody spoke: just listen again
    if (MIC_HELP[e.error]) return micProblem(MIC_HELP[e.error]);
    if (++micFailures >= 3) micProblem("Listening keeps failing (" + e.error + "). Type your question instead.");
  };
  r.onend = () => { if (recognizer === r) recognizer = null; };
  try { r.start(); } catch (err) { recognizer = null; }
}
setInterval(listenTick, 300);

// ================================================================ keyboard
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return;
  const actions = { c: "capture", r: "newitem", b: "bookmode", a: "read-all", p: "repeat", s: "stop", m: "mic" };
  const id = actions[e.key.toLowerCase()];
  if (id && !$(id).disabled) { e.preventDefault(); $(id).click(); }
});

// ================================================================ overlay
const COLORS = { idle: "#b4b4b4", detect: "#ffc800", hold: "#ffa500", ready: "#00ff00", line: "#0064ff" };

function sizeOverlay() {
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(video.clientWidth * dpr);
  overlay.height = Math.round(video.clientHeight * dpr);
}
window.addEventListener("resize", sizeOverlay);
video.addEventListener("resize", sizeOverlay);

function rect(ctx, box, W, H, pad = 0) {
  ctx.strokeRect(box[0] * W - pad, box[1] * H - pad, (box[2] - box[0]) * W + 2 * pad, (box[3] - box[1]) * H + 2 * pad);
}

function drawGuideBox(ctx, W, H, color, label) {
  const mx = (30 / 1280) * W, my = (20 / 720) * H, c = 30;
  ctx.strokeStyle = color; ctx.lineWidth = 3;
  for (const [x, y, dx, dy] of [[mx, my, 1, 1], [W - mx, my, -1, 1], [mx, H - my, 1, -1], [W - mx, H - my, -1, -1]]) {
    ctx.beginPath(); ctx.moveTo(x + dx * c, y); ctx.lineTo(x, y); ctx.lineTo(x, y + dy * c); ctx.stroke();
  }
  ctx.lineWidth = 1; ctx.strokeRect(mx, my, W - 2 * mx, H - 2 * my);
  if (label) { ctx.fillStyle = color; ctx.font = "bold 22px sans-serif"; ctx.fillText(label, mx + 10, my + 32); }
}

function drawNormal(ctx, W, H) {
  const busy = isSpeaking() || thinking;
  if (busy) return drawGuideBox(ctx, W, H, COLORS.idle, thinking && !isSpeaking() ? "THINKING..." : "SPEAKING...  S = stop");
  if (!guide) return drawGuideBox(ctx, W, H, COLORS.idle, "SHOW TEXT HERE");
  ctx.strokeStyle = COLORS.line; ctx.lineWidth = 1;
  for (const b of guide.lines) rect(ctx, b, W, H);
  if (guide.box) { ctx.lineWidth = 2; rect(ctx, guide.box, W, H); }
  const color = { CAPTURED: COLORS.ready, CAPTURING: COLORS.ready, HOLD: COLORS.hold, COACH: COLORS.hold,
                  ADJUST: COLORS.detect }[guide.state] || COLORS.idle;
  const label = guide.state === "CAPTURED" ? "CAPTURED - ASK ME   R = new item   A = read all" : guide.label;
  drawGuideBox(ctx, W, H, color, label);
  ctx.font = "13px sans-serif";
  ctx.fillStyle = guide.problems.length ? "#ff3030" : COLORS.idle;
  ctx.fillText(guide.quality, (30 / 1280) * W + 10, H - (20 / 720) * H - 12);
}

function lineAt(pos) {
  return pageLayout && pageLayout.lines.find((l) => l.start <= pos && pos < l.end);
}

function drawBook(ctx, W, H) {
  const turning = !bookState || bookState.state === "TURNING";
  if (pageLayout && !turning) {
    const L = pageLayout;
    if (L.gutter) { ctx.strokeStyle = "#ffff00"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(L.gutter * W, 0); ctx.lineTo(L.gutter * W, H); ctx.stroke(); }
    ctx.strokeStyle = "#c800c8"; ctx.lineWidth = 1;
    for (const b of L.margins) rect(ctx, b, W, H, 3);
    ctx.strokeStyle = "#0078ff";
    for (const b of L.columns) rect(ctx, [b[0], 20 / 720, b[2], 1 - 20 / 720], W, H);
    L.blocks.forEach((b, i) => {
      ctx.strokeStyle = "#00c800"; ctx.lineWidth = 2; rect(ctx, b, W, H);
      const bx = Math.max(14, b[0] * W - 20), by = b[1] * H + 12;
      ctx.fillStyle = "#dc0000"; ctx.beginPath(); ctx.arc(bx, by, 14, 0, 7); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = "bold 15px sans-serif"; ctx.textAlign = "center";
      ctx.fillText(String(i + 1), bx, by + 5); ctx.textAlign = "left";
    });
    // The line and word being spoken
    const pos = speechPos;
    const span = pos === null ? null : lineAt(pos);
    if (span) {
      ctx.fillStyle = "rgba(255,255,0,0.3)";
      ctx.fillRect(span.box[0] * W - 4, span.box[1] * H - 3, (span.box[2] - span.box[0]) * W + 8, (span.box[3] - span.box[1]) * H + 6);
      const word = span.words.find((w) => pos < w[1]);
      if (word) { ctx.strokeStyle = "#ff7800"; ctx.lineWidth = 2; rect(ctx, word[2], W, H, 3); }
    }
  }
  const bar = 40;
  ctx.fillStyle = "#282828"; ctx.fillRect(0, H - bar, W, bar);
  const s = bookState;
  ctx.fillStyle = "#00ffff"; ctx.font = "bold 16px sans-serif";
  ctx.fillText(`BOOK MODE   ${pageLabel || "Page -"}   ${bookBusy ? "READING PAGE..." : s ? s.state : "WAITING"}   ` +
    (s ? `motion ${s.motion} (still < ${s.still_below})   ` : "") + "B = exit", 15, H - 14);
}

function render() {
  const ctx = overlay.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = overlay.width / dpr, H = overlay.height / dpr;
  ctx.clearRect(0, 0, W, H);
  if (cameraReady && W > 0) { if (mode === "book") drawBook(ctx, W, H); else drawNormal(ctx, W, H); }
  requestAnimationFrame(render);
}

startCamera();
liveLoop();
render();
