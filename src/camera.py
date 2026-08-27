from __future__ import annotations

import os
import sys

import numpy as np


def _fourcc(code: str) -> int:
    import cv2

    return cv2.VideoWriter_fourcc(*code)


def _ok_frame(frame) -> bool:
    return frame is not None and getattr(frame, "size", 0) > 0


class _PiCam:
    def __init__(self, picam2):
        self._cam = picam2

    def read(self):
        try:
            arr = self._cam.capture_array()
        except Exception:
            return False, None
        if not _ok_frame(arr):
            return False, None
        if arr.ndim == 2:
            import cv2

            return True, cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return True, np.ascontiguousarray(arr[:, :, ::-1])

    def release(self) -> None:
        for fn in (self._cam.stop, self._cam.close):
            try:
                fn()
            except Exception:
                pass

    def isOpened(self) -> bool:
        return True


def _open_csi():
    try:
        from picamera2 import Picamera2
    except Exception:
        return None
    try:
        try:
            tuning = Picamera2.load_tuning_file("imx219_noir.json")
            cam = Picamera2(tuning=tuning)
        except Exception:
            cam = Picamera2()
        cfg = cam.create_preview_configuration(main={"size": (1640, 1232), "format": "BGR888"})
        cam.configure(cfg)
        cam.start()
        try:
            from libcamera import controls

            cam.set_controls({"AwbEnable": True, "AwbMode": controls.AwbModeEnum.Indoor})
        except Exception:
            pass
        probe = cam.capture_array()
        if not _ok_frame(probe):
            cam.stop()
            cam.close()
            return None
        return _PiCam(cam)
    except Exception:
        return None


def _usb_attempts(index: int | None):
    import cv2

    idxs = (index,) if index is not None else (0, 1)
    if sys.platform == "win32":
        backends = (
            (cv2.CAP_DSHOW, "MJPG"),
            (cv2.CAP_DSHOW, None),
            (cv2.CAP_MSMF, "MJPG"),
            (cv2.CAP_MSMF, None),
            (cv2.CAP_ANY, None),
        )
    else:
        backends = ((cv2.CAP_V4L2, None), (cv2.CAP_ANY, None))
    out = []
    for idx in idxs:
        for backend, fourcc in backends:
            out.append((idx, backend, fourcc))
    return out


def _open_usb(index: int | None = None):
    import cv2

    for idx, backend, fourcc in _usb_attempts(index):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, _fourcc(fourcc))
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
        return cap
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
