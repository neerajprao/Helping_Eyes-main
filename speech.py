"""
Cross-platform text-to-speech used by both readers.

Backends (all run as a subprocess so speech can be interrupted instantly):
    macOS   -> built-in `say` command
    Windows -> PowerShell + System.Speech (SAPI)
    Linux   -> `espeak-ng` / `espeak` / `spd-say`

Optional .env settings:
    SPEECH_VOICE = voice name (macOS: see `say -v '?'`, e.g. "Samantha")
    SPEECH_RATE  = words per minute (e.g. 180)
"""

import logging
import os
import queue
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


def _build_command() -> List[str]:
    """Return the TTS command for this OS. Text is always fed via stdin."""
    voice = os.getenv("SPEECH_VOICE")
    rate = os.getenv("SPEECH_RATE")

    if sys.platform == "darwin":
        cmd = ["say"]
        if voice:
            cmd += ["-v", voice]
        if rate:
            cmd += ["-r", rate]
        return cmd  # `say` with no message reads from stdin

    if sys.platform == "win32":
        script = "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        if voice:
            script += f"$s.SelectVoice('{voice}'); "
        if rate:
            # SAPI rate is -10..10; map roughly from words per minute (~180 = 0)
            script += f"$s.Rate = {max(-10, min(10, (int(rate) - 180) // 20))}; "
        script += "$s.Speak([Console]::In.ReadToEnd())"
        return ["powershell", "-NoProfile", "-Command", script]

    for engine in ("espeak-ng", "espeak"):
        if shutil.which(engine):
            cmd = [engine, "--stdin"]
            if voice:
                cmd += ["-v", voice]
            if rate:
                cmd += ["-s", rate]
            return cmd
    if shutil.which("spd-say"):
        return ["spd-say", "--wait", "-e"]

    raise RuntimeError("No TTS engine found. Install espeak-ng (e.g. `sudo apt install espeak-ng`).")


# `say --interactive` redraws the text on a terminal with the word being spoken
# in bold. Reading that from a pseudo-terminal tells us exactly which word is
# being spoken right now (macOS only).
_REDRAW = b"\r\x1b[M"
_BOLD = b"\x1b[1m"
_ESCAPES = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]|\x1b\([A-Z]")


class Speaker:
    """Queued, interruptible speech. Utterances are spoken one at a time, in order."""

    def __init__(self, max_queue: int = 50):
        self._cmd = _build_command()
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._busy = threading.Event()  # set while anything is queued or playing
        self._generation = 0            # bumped by stop(); stale queued items are dropped
        self._proc: Optional[subprocess.Popen] = None
        self._closed = False
        self.last_active = 0.0          # time.time() when speech was last queued or playing
        self.position: Optional[int] = None  # character being spoken, for tracked text (see say())
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info(f"Speech ready ({self._cmd[0]})")

    @property
    def is_speaking(self) -> bool:
        return self._busy.is_set()

    def spoke_since(self, t: float) -> bool:
        """True if anything was queued or playing at any point after time t"""
        return self.is_speaking or self.last_active >= t

    def say(self, text: str, track_from: Optional[int] = None) -> None:
        """
        Queue text to be spoken (non-blocking).

        track_from: for text taken from a longer document, the character offset
        where this text starts in it. While it's spoken, `position` then holds
        the document offset of the word being spoken (macOS only).
        """
        if not text or not text.strip() or self._closed:
            return
        if track_from is not None:
            track_from += len(text) - len(text.lstrip())   # keep offsets right after strip()
        with self._lock:
            try:
                self._queue.put_nowait((self._generation, text.strip(), track_from))
            except queue.Full:
                logger.warning("Speech queue full, skipping")
                return
            self._busy.set()
            self.last_active = time.time()

    def stop(self) -> None:
        """Silence current speech and discard everything queued."""
        with self._lock:
            self._generation += 1
            self._drain()
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                self.last_active = time.time()
            self._busy.clear()
            self.position = None

    def wait_until_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until nothing is queued or playing. Returns False on timeout."""
        step = 0.05
        waited = 0.0
        while self._busy.is_set():
            if timeout is not None and waited >= timeout:
                return False
            threading.Event().wait(step)
            waited += step
        return True

    def shutdown(self) -> None:
        self.stop()
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    # ---------------- internals ----------------
    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            generation, text, track_from = item
            tracked = track_from is not None and sys.platform == "darwin"

            with self._lock:
                if generation != self._generation:
                    continue  # stop() was called after this was queued
                if tracked:
                    self.position = track_from
                    self._proc, master = self._start_tracked(text)
                else:
                    master = None
                    try:
                        self._proc = subprocess.Popen(
                            self._cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            text=True,
                        )
                    except Exception as e:
                        logger.error(f"Speech error: {e}")
                        self._proc = None
                proc = self._proc

            if proc is not None:
                try:
                    if master is not None:
                        self._follow(proc, master, track_from, generation)
                    else:
                        proc.communicate(input=text)
                except Exception as e:
                    logger.error(f"Speech error: {e}")

            with self._lock:
                self._proc = None
                self.last_active = time.time()
                if self._queue.empty():
                    self._busy.clear()
                    self.position = None

    def _start_tracked(self, text: str):
        """Start `say --interactive` on a pseudo-terminal; returns (process, pty fd)."""
        import fcntl, pty, struct, termios
        master, slave = pty.openpty()
        # Very wide terminal so the text is never wrapped or scrolled
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 4000, 0, 0))
        try:
            proc = subprocess.Popen(
                self._cmd + ["--interactive=bold", "-f", "-"],
                stdin=subprocess.PIPE, stdout=slave, stderr=subprocess.DEVNULL,
                env={**os.environ, "TERM": "xterm-256color"},
            )
            proc.stdin.write(text.encode())
            proc.stdin.close()
            return proc, master
        except Exception as e:
            logger.error(f"Speech error: {e}")
            os.close(master)
            return None, None
        finally:
            os.close(slave)

    def _follow(self, proc: subprocess.Popen, master: int, track_from: int, generation: int) -> None:
        """Read say's redraws until it finishes, updating `position` word by word."""
        buf = b""
        try:
            while True:
                ready, _, _ = select.select([master], [], [], 0.05)
                if ready:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    buf += data
                    redraws = buf.split(_REDRAW)
                    buf = redraws[-1]
                    for redraw in redraws[:-1]:
                        if _BOLD in redraw and generation == self._generation:
                            before = _ESCAPES.sub(b"", redraw.split(_BOLD)[0])
                            self.position = track_from + len(before.decode("utf-8", errors="ignore"))
                elif proc.poll() is not None:
                    break
            proc.wait()
        finally:
            os.close(master)
