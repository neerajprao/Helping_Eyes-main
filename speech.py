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
import shutil
import subprocess
import sys
import threading
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
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info(f"✅ Speech ready ({self._cmd[0]})")

    @property
    def is_speaking(self) -> bool:
        return self._busy.is_set()

    def say(self, text: str) -> None:
        """Queue text to be spoken (non-blocking)."""
        if not text or not text.strip() or self._closed:
            return
        with self._lock:
            try:
                self._queue.put_nowait((self._generation, text.strip()))
            except queue.Full:
                logger.warning("⚠️ Speech queue full, skipping")
                return
            self._busy.set()

    def stop(self) -> None:
        """Silence current speech and discard everything queued."""
        with self._lock:
            self._generation += 1
            self._drain()
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
            self._busy.clear()

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
            generation, text = item

            with self._lock:
                if generation != self._generation:
                    continue  # stop() was called after this was queued
                try:
                    self._proc = subprocess.Popen(
                        self._cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        text=True,
                    )
                except Exception as e:
                    logger.error(f"❌ Speech error: {e}")
                    self._proc = None
                proc = self._proc

            if proc is not None:
                try:
                    proc.communicate(input=text)
                except Exception as e:
                    logger.error(f"⚠️ Speech error: {e}")

            with self._lock:
                self._proc = None
                if self._queue.empty():
                    self._busy.clear()
