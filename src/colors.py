"""Live colour tuning for the kiosk camera.

The Camera Module V2 NoIR has no IR-cut filter, so daylight and lamp light both
land with a blue/purple cast. These six controls — red, green, blue, saturation,
brightness, contrast — are pushed straight at the ISP when the camera is a CSI
sensor, and re-created in PIL/numpy for USB webcams and the PC sim.

Green gain and the ISP
----------------------
libcamera's `ColourGains` is a 2-tuple `(red_gain, blue_gain)`. There is no
green gain: green is the fixed reference channel at 1.0, because that is what
white balance is *measured against* — only the ratios between channels carry
any meaning. A user-facing green gain `g` is therefore exact when expressed as

    ColourGains = (red / green, blue / green)

which makes the green slider a green<->magenta tint axis. Auto-exposure stays
on and absorbs the overall brightness change that scaling all three implies, so
nothing here touches DigitalGain.

`AwbEnable` must go to False alongside the gains. With AWB left on, the ISP
re-converges within about a second and silently discards the manual gains.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from src.paths import scans_dir

PROFILE_FILENAME = "camera_profile.json"
PROFILE_NAMES = ("night", "morning")
DEFAULT_ACTIVE = "night"

# Practical UI limits, NOT hardware limits. Whenever picamera2 reports real
# ranges via `Picamera2.camera_controls` those win (see `merge_ranges`); these
# are only the fallback for a USB webcam, the PC sim, or a stack that does not
# advertise a control.
FALLBACK_RANGES: dict[str, tuple[float, float]] = {
    "red": (0.5, 4.0),
    "green": (0.5, 4.0),
    "blue": (0.5, 4.0),
    "saturation": (0.0, 2.0),
    "brightness": (-1.0, 1.0),
    "contrast": (0.0, 2.0),
}

# Which libcamera control each slider reads its range from. The three gains all
# come out of ColourGains, which is why they share an entry.
_RANGE_CONTROL = {
    "red": "ColourGains",
    "green": "ColourGains",
    "blue": "ColourGains",
    "saturation": "Saturation",
    "brightness": "Brightness",
    "contrast": "Contrast",
}

_EPSILON = 1e-6


def _as_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def merge_ranges(camera_ranges: dict | None) -> dict[str, tuple[float, float]]:
    """Slider ranges: the camera's limits intersected with the useful ones.

    `camera_ranges` is `Picamera2.camera_controls`: control name -> (min, max,
    default). Anything missing, malformed, or non-scalar falls back to
    FALLBACK_RANGES.

    Intersect rather than adopt. The IMX219 reports ColourGains, Saturation and
    Contrast as 0.0-32.0, but everything usable sits below ~4.0, so handing the
    hardware range straight to a tk.Scale would squeeze the entire working range
    into the leftmost few percent of the slider — unusable on a touchscreen,
    which is the one place these sliders have to work. A hardware limit that is
    *narrower* than the practical one still wins; a hardware range that misses
    the practical window entirely is taken as-is, since the device knows best.
    """
    out = dict(FALLBACK_RANGES)
    if not isinstance(camera_ranges, dict):
        return out
    for name, control in _RANGE_CONTROL.items():
        entry = camera_ranges.get(control)
        if not isinstance(entry, (tuple, list)) or len(entry) < 2:
            continue
        low = _as_float(entry[0])
        high = _as_float(entry[1])
        if low is None or high is None or high <= low:
            continue
        want_low, want_high = FALLBACK_RANGES[name]
        lo, hi = max(low, want_low), min(high, want_high)
        out[name] = (lo, hi) if hi > lo else (low, high)
    return out


def _clamp(value: float, span: tuple[float, float]) -> float:
    low, high = span
    if value < low:
        return low
    if value > high:
        return high
    return value


@dataclass
class ColorProfile:
    """Six user-facing colour controls. Defaults are the neutral no-op."""

    red: float = 1.0
    green: float = 1.0
    blue: float = 1.0
    saturation: float = 1.0
    brightness: float = 0.0
    contrast: float = 1.0

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_dict(cls, data) -> "ColorProfile":
        """Tolerant load: missing keys keep their default, junk is ignored."""
        out = cls()
        if not isinstance(data, dict):
            return out
        for name in cls.field_names():
            value = _as_float(data.get(name))
            if value is not None:
                setattr(out, name, value)
        return out

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    def copy(self) -> "ColorProfile":
        return ColorProfile(**self.to_dict())

    def clamped(self, ranges: dict[str, tuple[float, float]] | None = None) -> "ColorProfile":
        spans = ranges or FALLBACK_RANGES
        return ColorProfile(
            **{
                name: _clamp(float(getattr(self, name)), spans.get(name, FALLBACK_RANGES[name]))
                for name in self.field_names()
            }
        )

    def is_neutral(self) -> bool:
        return (
            abs(self.red - 1.0) < _EPSILON
            and abs(self.green - 1.0) < _EPSILON
            and abs(self.blue - 1.0) < _EPSILON
            and abs(self.saturation - 1.0) < _EPSILON
            and abs(self.brightness) < _EPSILON
            and abs(self.contrast - 1.0) < _EPSILON
        )

    def to_controls(
        self,
        ranges: dict[str, tuple[float, float]] | None = None,
        base_gains: tuple[float, float] | None = None,
    ) -> dict:
        """The dict to hand to `Picamera2.set_controls`.

        When neutral, caller enables AWB. When tuned, gains scale relative to base_gains.
        """
        prof = self.clamped(ranges)
        green = prof.green if prof.green > _EPSILON else 1.0
        bg_r, bg_b = base_gains if base_gains and len(base_gains) >= 2 else (1.0, 1.0)
        return {
            "AwbEnable": False,
            "ColourGains": (bg_r * (prof.red / green), bg_b * (prof.blue / green)),
            "Saturation": prof.saturation,
            "Brightness": prof.brightness,
            "Contrast": prof.contrast,
        }

    def apply_pil(self, img):
        """PIL/numpy stand-in for the ISP, for USB webcams and the PC sim.

        Returns `img` itself — no copy, no allocation — when neutral, because
        this sits in the per-frame path.

        One numpy pass over a float32 copy, in the same order the ISP applies
        them: channel gains (green normalised away exactly as in `to_controls`),
        additive brightness, contrast about mid-grey, then saturation as a blend
        against Rec.601 luma. PIL's ImageEnhance would need three separate
        full-image passes plus a fourth for the gains.
        """
        if self.is_neutral():
            return img
        prof = self.clamped()
        green = prof.green if prof.green > _EPSILON else 1.0
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim != 3 or arr.shape[2] < 3:
            return img
        gains = np.array([prof.red / green, 1.0, prof.blue / green], dtype=np.float32)
        rgb = arr[:, :, :3] * gains
        if prof.brightness:
            rgb += np.float32(prof.brightness)
        if prof.contrast != 1.0:
            rgb = (rgb - np.float32(0.5)) * np.float32(prof.contrast) + np.float32(0.5)
        if prof.saturation != 1.0:
            luma = (
                rgb[:, :, 0] * np.float32(0.299)
                + rgb[:, :, 1] * np.float32(0.587)
                + rgb[:, :, 2] * np.float32(0.114)
            )[:, :, None]
            rgb = luma + (rgb - luma) * np.float32(prof.saturation)
        np.clip(rgb, 0.0, 1.0, out=rgb)

        from PIL import Image

        return Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), "RGB")


def profile_from_gains(gains) -> ColorProfile | None:
    """Turn a libcamera `ColourGains` 2-tuple into a profile.

    Metadata reports (red_gain, blue_gain) relative to green == 1.0, which is
    exactly the convention `to_controls` writes back out.
    """
    if not isinstance(gains, (tuple, list)) or len(gains) < 2:
        return None
    red = _as_float(gains[0])
    blue = _as_float(gains[1])
    if red is None or blue is None or red <= 0 or blue <= 0:
        return None
    return ColorProfile(red=red, green=1.0, blue=blue)


class ProfileStore:
    """Named profiles on disk, next to the scans.

    Deliberately *not* inside the app directory: the Pi is updated by unzipping
    a fresh copy of the app over the old one, which would wipe tuned values.

    Schema:
        {"active": "night", "profiles": {"night": {...}, "morning": {...}}}
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else (scans_dir() / PROFILE_FILENAME)
        self.active = DEFAULT_ACTIVE
        self.profiles: dict[str, ColorProfile] = {name: ColorProfile() for name in PROFILE_NAMES}
        self.first_run = True
        self.load()

    def load(self) -> None:
        """Never raises. A missing or corrupt file just means defaults."""
        self.active = DEFAULT_ACTIVE
        self.profiles = {name: ColorProfile() for name in PROFILE_NAMES}
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            self.first_run = True
            return
        self.first_run = False
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return
        if not isinstance(data, dict):
            return
        stored = data.get("profiles")
        if isinstance(stored, dict):
            for name in PROFILE_NAMES:
                self.profiles[name] = ColorProfile.from_dict(stored.get(name))
        active = data.get("active")
        if isinstance(active, str) and active in PROFILE_NAMES:
            self.active = active

    def save(self) -> bool:
        """Atomic write: temp file in the same directory, then os.replace.

        A power cut mid-write can then only lose the new values, never shred
        the old ones.
        """
        payload = {
            "active": self.active,
            "profiles": {name: self.profiles[name].to_dict() for name in PROFILE_NAMES},
        }
        tmp = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".camera_profile-", suffix=".json", dir=str(self.path.parent))
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
            tmp = None
            self.first_run = False
            return True
        except (OSError, TypeError, ValueError):
            return False
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def get(self, name: str) -> ColorProfile:
        return self.profiles.get(name) or ColorProfile()

    def current(self) -> ColorProfile:
        return self.get(self.active)

    def set_profile(self, name: str, profile: ColorProfile, *, activate: bool = False) -> bool:
        if name not in PROFILE_NAMES:
            return False
        self.profiles[name] = profile.copy()
        if activate:
            self.active = name
        return self.save()

    def set_active(self, name: str) -> bool:
        if name not in PROFILE_NAMES:
            return False
        self.active = name
        return self.save()
