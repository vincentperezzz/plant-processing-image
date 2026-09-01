"""One-slot frame hand-off between the Tk thread and the HTTP threads.

Pure plumbing on purpose: no PIL, no cv2, no sockets. The Tk thread calls
`publish()` with already-encoded JPEG bytes; streaming handlers call `wait()`.

The single rule this module exists to enforce is that **the Tk thread never
waits for a client**. Only the newest frame is kept, `publish()` takes the lock
for the few microseconds it needs to swap a bytes object, and a stalled client
sitting in `wait()` is not holding the lock (Condition.wait releases it). A
viewer on a slow phone therefore drops frames instead of dropping the kiosk's
framerate.

`wants_frame()` is the other half of that bargain: when nobody is watching it
returns False, so the display path never pays for a JPEG encode.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class ClientLimit(RuntimeError):
    """Raised by `FrameBus.client()` when the viewer cap is already taken."""


class FrameBus:
    def __init__(self, *, max_fps: float = 20.0) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes = b""
        self._seq = 0
        self._clients = 0
        self._last_publish = 0.0
        try:
            fps = float(max_fps)
        except (TypeError, ValueError):
            fps = 20.0
        self._interval = 1.0 / fps if fps > 0 else 0.0
        self._enc_lock = threading.Lock()
        self._pending_frame = None
        self._enc_running = True
        self._enc_thread = threading.Thread(target=self._encoder_loop, daemon=True)
        self._enc_thread.start()

    def _encoder_loop(self) -> None:
        while self._enc_running:
            item = None
            with self._enc_lock:
                if self._pending_frame is not None:
                    item = self._pending_frame
                    self._pending_frame = None
            if item is None:
                time.sleep(0.005)
                continue
            arr, stream_w, quality = item
            try:
                import cv2

                h, w = arr.shape[:2]
                if w > stream_w:
                    sh = max(1, round(h * stream_w / w))
                    arr = cv2.resize(arr, (stream_w, sh), interpolation=cv2.INTER_LINEAR)
                ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if ok:
                    self.publish(buf.tobytes())
            except Exception:
                pass

    def publish_array_async(self, bgr_array, stream_w: int = 1024, quality: int = 80) -> None:
        """Enqueue a BGR numpy frame for background JPEG encoding."""
        if not self.wants_frame():
            return
        with self._enc_lock:
            self._pending_frame = (bgr_array, stream_w, quality)

    # — producer side (Tk thread) ————————————————————————————————

    def publish(self, jpeg: bytes, now: float | None = None) -> int:
        """Store the newest frame and wake every waiter. Never blocks."""
        if now is None:
            now = _now()
        with self._cond:
            self._jpeg = bytes(jpeg)
            self._seq += 1
            self._last_publish = now
            seq = self._seq
            self._cond.notify_all()
        return seq

    def wants_frame(self, now: float | None = None) -> bool:
        """True only if somebody is watching and the frame interval elapsed."""
        if now is None:
            now = _now()
        with self._cond:
            if self._clients <= 0:
                return False
            return (now - self._last_publish) >= self._interval

    # — consumer side (HTTP threads) ——————————————————————————————

    def wait(self, last_seq: int, timeout: float = 2.0) -> tuple[int, bytes] | None:
        """Newest (seq, jpeg) once seq > last_seq, or None on timeout."""
        with self._cond:
            if self._seq > last_seq and self._jpeg:
                return self._seq, self._jpeg
            self._cond.wait(timeout)
            if self._seq > last_seq and self._jpeg:
                return self._seq, self._jpeg
        return None

    def latest(self) -> tuple[int, bytes]:
        with self._cond:
            return self._seq, self._jpeg

    @property
    def client_count(self) -> int:
        with self._cond:
            return self._clients

    @property
    def seq(self) -> int:
        with self._cond:
            return self._seq

    def age(self, now: float | None = None) -> float:
        """Seconds since the last publish. Large number when nothing yet."""
        if now is None:
            now = _now()
        with self._cond:
            if self._seq == 0:
                return 1e9
            return max(0.0, now - self._last_publish)

    @contextmanager
    def client(self, limit: int | None = None):
        """Count one live viewer for the duration of the block.

        A handler that dies mid-stream still runs the `finally`, so the count
        cannot leak and pin the encoder on forever.
        """
        with self._cond:
            if limit is not None and self._clients >= limit:
                raise ClientLimit(f"stream client limit reached ({limit})")
            self._clients += 1
        try:
            yield self
        finally:
            with self._cond:
                self._clients = max(0, self._clients - 1)
                # Wake waiters so a shutdown does not sit out the full timeout.
                self._cond.notify_all()


def _now() -> float:
    return time.monotonic()
