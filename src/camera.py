from __future__ import annotations

import os
import sys

import numpy as np


def _fourcc(code: str) -> int:
    import cv2

    return cv2.VideoWriter_fourcc(*code)


def _ok_frame(frame) -> bool:
    return frame is not None and getattr(frame, "size", 0) > 0


def _cover_np(arr, tw: int, th: int):
    h, w = arr.shape[:2]
    if w == tw and h == th:
        return arr
    import cv2

    scale = max(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    small = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
    x = max(0, (nw - tw) // 2)
    y = max(0, (nh - th) // 2)
    return np.ascontiguousarray(small[y : y + th, x : x + tw])


class _PiCam:
    def __init__(self, picam2, rgb=True, stream="main", view=(1024, 576), yuv=False):
        self._cam = picam2
        self.rgb = rgb
        self._stream = stream
        self._view = view
        self._yuv = yuv
        self._timeout_kw: dict | None = None
        self._base_gains: tuple[float, float] | None = None

    def _grab(self):
        """capture_request, with a timeout when this picamera2 supports one.

        Older builds have no `timeout` kwarg, and probing for it by catching
        TypeError on every frame costs an exception per frame. Decide once.
        """
        if self._timeout_kw is None:
            import inspect

            try:
                params = inspect.signature(self._cam.capture_request).parameters
            except (TypeError, ValueError):
                params = {}
            self._timeout_kw = {"timeout": _CAPTURE_TIMEOUT} if "timeout" in params else {}
        return self._cam.capture_request(**self._timeout_kw)

    def read(self):
        arr = None
        try:
            req = self._grab()
        except Exception:
            return False, None
        try:
            try:
                arr = req.make_array(self._stream)
            except Exception:
                arr = req.make_array("main")
        finally:
            req.release()
        if not _ok_frame(arr):
            return False, None
        tw, th = self._view
        if arr.ndim == 2:
            import cv2

            # I420/YUV420 arrives as a single plane of height 1.5x the image.
            if self._yuv or arr.shape[0] == th * 3 // 2 or arr.shape[0] == arr.shape[1] * 3 // 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_YUV2RGB_I420)
                self.rgb = True
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                self.rgb = False
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.shape[1] != tw or arr.shape[0] != th:
            arr = _cover_np(arr, tw, th)
        elif not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return True, arr

    def set_profile(self, profile) -> bool:
        """Push a ColorProfile at the ISP. True when it landed, never raises.

        Only the CSI path has this; `_open_usb` hands back a bare
        cv2.VideoCapture, so callers gate on `hasattr(cap, "set_profile")`.
        """
        try:
            if self._base_gains is None:
                meta = self.metadata()
                gains = meta.get("ColourGains")
                if gains and len(gains) >= 2 and gains[0] > 0.1 and gains[1] > 0.1:
                    self._base_gains = (float(gains[0]), float(gains[1]))
                else:
                    self._base_gains = (1.4, 2.2)

            if profile.is_neutral() or (
                abs(profile.red - 1.0) < 0.01
                and abs(profile.green - 1.0) < 0.01
                and abs(profile.blue - 1.0) < 0.01
            ):
                from libcamera import controls

                self._cam.set_controls(
                    {
                        "AwbEnable": True,
                        "AwbMode": controls.AwbModeEnum.Auto,
                        "Saturation": profile.saturation,
                        "Brightness": profile.brightness,
                        "Contrast": profile.contrast,
                    }
                )
            else:
                self._cam.set_controls(profile.to_controls(base_gains=self._base_gains))
            return True
        except Exception:
            return False

    def control_ranges(self) -> dict:
        """`Picamera2.camera_controls`: name -> (min, max, default). {} on failure."""
        try:
            ranges = self._cam.camera_controls
        except Exception:
            return {}
        return ranges if isinstance(ranges, dict) else {}

    def metadata(self) -> dict:
        """Last frame's control metadata (includes ColourGains). {} on failure."""
        try:
            meta = self._cam.capture_metadata()
        except Exception:
            return {}
        return meta if isinstance(meta, dict) else {}

    def release(self) -> None:
        for fn in (self._cam.stop, self._cam.close):
            try:
                fn()
            except Exception:
                pass

    def isOpened(self) -> bool:
        return True


CSI_VIEW = (1024, 576)
_CAPTURE_TIMEOUT = 2.0
_OPEN_TIMEOUT = float(os.environ.get("PLANT_CAMERA_TIMEOUT", "8"))


def _csi_plans(view):
    """Configs to try, cheapest and least exotic first.

    A single small `main` stream is the whole point: the ISP scales on-chip, so
    the CPU never touches a 2MP buffer. `lores` is only a fallback for stacks
    that refuse an RGB main at this size.
    """
    w, h = view
    return (
        ({"main": {"size": (w, h), "format": "RGB888"}}, "main", True, False),
        ({"main": {"size": (w, h), "format": "BGR888"}}, "main", False, False),
        (
            {
                "main": {"size": (1640, 1232), "format": "RGB888"},
                "lores": {"size": (w, h), "format": "YUV420"},
            },
            "lores",
            True,
            True,
        ),
        ({"main": {"size": (1280, 720), "format": "RGB888"}}, "main", True, False),
    )


def _probe_frame(cam, stream: str, seconds: float):
    """capture_array with a hard wall-clock cap.

    picamera2 blocks forever when the pipeline never delivers, which froze the
    kiosk before it ever painted. A dead camera has to look like `None`, not a
    hang.
    """
    import threading

    out = {}

    def work():
        try:
            out["arr"] = cam.capture_array(stream)
        except Exception as exc:
            out["err"] = exc

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        return None, TimeoutError(f"no frame within {seconds:.0f}s")
    return out.get("arr"), out.get("err")


def _open_csi():
    try:
        from picamera2 import Picamera2
    except Exception as exc:
        print(f"csi unavailable: {exc}", flush=True)
        return None
    view = CSI_VIEW
    cam = None
    try:
        try:
            tuning = Picamera2.load_tuning_file("imx219_noir.json")
            cam = Picamera2(tuning=tuning)
        except Exception:
            cam = Picamera2()

        picked = None
        for main_kwargs, stream, rgb, yuv in _csi_plans(view):
            kwargs = dict(main_kwargs)
            kwargs["buffer_count"] = 4
            try:
                cam.configure(cam.create_preview_configuration(**kwargs))
            except Exception as exc:
                print(f"csi config rejected ({stream}): {exc}", flush=True)
                continue
            picked = (kwargs, stream, rgb, yuv)
            break
        if picked is None:
            cam.close()
            print("csi: no usable configuration", flush=True)
            return None

        cam.start()
        try:
            from libcamera import controls

            cam.set_controls(
                {
                    "AwbEnable": True,
                    "AwbMode": controls.AwbModeEnum.Auto,
                    "FrameDurationLimits": (16666, 33333),
                }
            )
        except Exception:
            pass

        kwargs, stream, rgb, yuv = picked
        probe, err = _probe_frame(cam, stream, _OPEN_TIMEOUT)
        if not _ok_frame(probe):
            print(f"csi probe failed: {err}", flush=True)
            for fn in (cam.stop, cam.close):
                try:
                    fn()
                except Exception:
                    pass
            return None
        size = kwargs["main"]["size"]
        fmt = kwargs["main"]["format"]
        print(
            f"csi {size[0]}x{size[1]}-{fmt} stream={stream} view={view[0]}x{view[1]} "
            f"shape={getattr(probe, 'shape', None)}",
            flush=True,
        )
        return _PiCam(cam, rgb=rgb, stream=stream, view=view, yuv=yuv)
    except Exception as exc:
        print(f"csi open failed: {exc}", flush=True)
        if cam is not None:
            for fn in (cam.stop, cam.close):
                try:
                    fn()
                except Exception:
                    pass
        return None


def _usb_attempts(index: int | None):
    import cv2

    idxs = (index,) if index is not None else (1, 0, 2, 4)
    if sys.platform == "win32":
        backends = (
            (cv2.CAP_DSHOW, "MJPG"),
            (cv2.CAP_DSHOW, None),
            (cv2.CAP_MSMF, "MJPG"),
            (cv2.CAP_MSMF, None),
            (cv2.CAP_ANY, None),
        )
    else:
        backends = ((cv2.CAP_V4L2, "MJPG"), (cv2.CAP_V4L2, None), (cv2.CAP_ANY, None))
    out = []
    for idx in idxs:
        for backend, fourcc in backends:
            out.append((idx, backend, fourcc))
    return out


class _ThreadedUsbCam:
    """Zero-latency threaded reader for USB webcams on Linux/Windows.

    Decouples OpenCV blocking hardware USB buffer reads from the UI/Streamer.
    Constantly drains the V4L2 USB hardware ring buffer in a background thread,
    guaranteeing that cap.read() always returns the latest frame in 0.0ms.
    """

    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._running = True
        self._ok = False
        self._frame = None
        self.rgb = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running and self._cap.isOpened():
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._ok = ok
                    self._frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame

    def isOpened(self) -> bool:
        return self._cap.isOpened() and self._running

    def release(self) -> None:
        self._running = False
        self._cap.release()


def _open_usb(index: int | None = None):
    import cv2

    for idx, backend, fourcc in _usb_attempts(index):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, _fourcc(fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = False
        frame = None
        for _ in range(8):
            ok, frame = cap.read()
            if ok and _ok_frame(frame):
                break
        if not ok or not _ok_frame(frame):
            cap.release()
            continue
        if frame.shape[0] < 32 or frame.shape[1] < 32:
            cap.release()
            continue
        return _ThreadedUsbCam(cap)
    return None


def open_capture(prefer: str = "auto"):
    prefer = (prefer or "auto").strip().lower()
    env = os.environ.get("PLANT_CAMERA", "").strip().lower()
    if prefer == "auto" and env:
        prefer = env
    if prefer in ("csi", "picam", "pi"):
        return _open_csi()
    if prefer.isdigit():
        return _open_usb(int(prefer))
    if prefer in ("usb", "webcam"):
        return _open_usb(None)
    cap = _open_csi()
    if cap is not None:
        return cap
    return _open_usb(None)
