import math
import os
import platform
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tkinter as tk
import tkinter.font as tkfont

from PIL import Image, ImageDraw, ImageFont, ImageTk

from src.camera import open_capture
from src.colors import (
    FALLBACK_RANGES,
    ColorProfile,
    ProfileStore,
    merge_ranges,
    profile_from_gains,
)
from src.detect import PlantFinder, _iou, crop_xyxy, match_tracks
from src.infer import FARM_CROPS, Scanner
from src.paths import CKPT
from src.scan_drop import _box_area, _frame_to_pil, _track_label
from src.signal_light import DEFAULT_PIN, CaptureLight

FONTS_DIR = ROOT / "vendor" / "fonts"

W = 1024
H = 600

# — design tokens (exact hex, from docs/UI-modernization/README.md) —
CREAM = "#f5ead8"
SURFACE = "#ebddc5"
TEXT = "#201e1d"
ACCENT = "#c67139"
ACCENT_2_400 = "#aebf92"
ALERT = "#a52929"
NEUTRAL_400 = "#c0b6a5"
NEUTRAL_900 = "#2e2b25"
DIVIDER = "#ded2ba"
LEAF_GREEN = "#036819"
MUTED = "#6d6559"

BG = CREAM
VIEW = NEUTRAL_900
DASH = "—"
LOOKING = "Looking for plant"
NO_CAMERA = "No camera. Check the ribbon or plug in a USB webcam."

# Frame pacing. The floor is what buys Tk the idle time it needs to
# actually blit the canvas; without it the screen stops updating.
TICK_BUDGET_MS = 33
TICK_FLOOR_MS = 8

# health-grade tokens: (background, text/value, border/accent) — all RGB(A)
HEALTH_TOKENS = {
    "healthy": ((205, 240, 205), (0, 73, 6), (63, 174, 74)),
    "mild": ((250, 226, 176), (110, 56, 0), (197, 141, 4)),
    "critical": ((255, 228, 225), (131, 0, 13), (199, 75, 71)),
    "dead": ((220, 211, 196), (46, 43, 37), (71, 66, 56)),
}
HEALTH_DISPLAY = {
    "healthy": "Healthy",
    "mild": "Mild stress",
    "critical": "Critical",
    "dead": "Dead",
}


def _hex_rgb(value: str) -> tuple[int, int, int]:
    h = value.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _display_name(value: str) -> str:
    text = (value or "").strip()
    if not text or text == DASH:
        return text or DASH
    lower = text.lower()
    if lower in HEALTH_DISPLAY:
        return HEALTH_DISPLAY[lower]
    return text[:1].upper() + text[1:]

SHUTTER_R = 44
SHUTTER_RING = 5
SHUTTER_EDGE_INSET = 24
HUD_W = 236
HUD_R = 16
VIEW_RGB = (46, 43, 37)
TEXT_RGB = (32, 30, 29)
LABEL_RGB = (198, 113, 57)
WARN_RGB = (165, 41, 41)
CARD_FILL = (235, 221, 197, 255)
CARD_LINE = (32, 30, 29, 40)
CARD_FILL_RGB = CARD_FILL[:3]
# CARD_LINE pre-blended over CARD_FILL_RGB (its usual background), for cheap opaque drawing.
CARD_LINE_RGB = (203, 191, 171)


def _pick_font(root: tk.Tk, *, heading: bool = False) -> str:
    want = (
        ("Caprasimo", "Segoe UI", "DejaVu Sans", "Piboto", "Liberation Sans", "FreeSans")
        if heading
        else ("Figtree", "Segoe UI", "DejaVu Sans", "Piboto", "Liberation Sans", "FreeSans")
    )
    have = set(tkfont.families(root))
    for name in want:
        if name in have:
            return name
    return "TkDefaultFont"


def _load_font(size: int, *, heading: bool = False, weight: str = "regular") -> ImageFont.ImageFont:
    names: list[Path] = []
    if heading:
        names.append(FONTS_DIR / "Caprasimo-Regular.ttf")
    else:
        vendored = {
            "regular": "Figtree-Regular.ttf",
            "semibold": "Figtree-SemiBold.ttf",
            "bold": "Figtree-Bold.ttf",
        }[weight]
        names.append(FONTS_DIR / vendored)
    bold = weight in ("semibold", "bold")
    if sys.platform == "win32":
        wind = Path(r"C:\Windows\Fonts")
        names += [wind / ("segoeuib.ttf" if bold else "segoeui.ttf")]
    names += [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/piboto/Piboto-Bold.ttf" if bold else "/usr/share/fonts/truetype/piboto/Piboto-Regular.ttf"),
    ]
    for path in names:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _leaf_icon(size: int, color: str) -> Image.Image:
    big = size * 4
    canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    w, h = int(big * 0.95), int(big * 0.55)
    x0, y0 = (big - w) // 2, (big - h) // 2
    stroke = max(2, big // 18)
    draw.ellipse((x0, y0, x0 + w, y0 + h), outline=color, width=stroke)
    draw.line((x0 + stroke, y0 + h // 2, x0 + w - stroke, y0 + h // 2), fill=color, width=stroke)
    rotated = canvas.rotate(-45, resample=Image.BICUBIC, expand=False)
    draw2 = ImageDraw.Draw(rotated)
    draw2.line((big * 0.08, big * 0.92, big * 0.24, big * 0.76), fill=color, width=stroke)
    return rotated.resize((size, size), Image.LANCZOS)


def _trash_icon(size: int, color: str) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    stroke = max(2, size // 12)
    top, bottom = size * 0.3, size * 0.9
    left, right = size * 0.2, size * 0.8
    draw.line((size * 0.1, top, size * 0.9, top), fill=color, width=stroke)
    draw.rounded_rectangle((left, top, right, bottom), radius=size * 0.08, outline=color, width=stroke)
    draw.line((size * 0.38, top, size * 0.38, size * 0.14), fill=color, width=stroke)
    draw.line((size * 0.62, top, size * 0.62, size * 0.14), fill=color, width=stroke)
    draw.line((size * 0.38, size * 0.14, size * 0.62, size * 0.14), fill=color, width=stroke)
    for fx in (0.38, 0.5, 0.62):
        draw.line(
            (size * fx, top + stroke * 1.6, size * fx, bottom - stroke * 1.6),
            fill=color,
            width=max(1, stroke // 2),
        )
    return img


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font, max_w: int, limit: int = 4) -> list[str]:
    words = (text or "").split()
    if not words:
        return [DASH]
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        tw, _ = _text_size(draw, trial, font)
        if tw <= max_w:
            cur = trial
            continue
        if cur:
            lines.append(cur)
            if len(lines) >= limit:
                cur = ""
                break
        cur = word
    if cur and len(lines) < limit:
        lines.append(cur)
    if len(lines) == limit:
        last = lines[-1].rstrip("…")
        while last and _text_size(draw, last + "…", font)[0] > max_w:
            last = last[:-1]
        lines[-1] = (last + "…") if last else "…"
    return lines or [DASH]


class PiSim:
    def __init__(
        self,
        root: tk.Tk,
        *,
        fullscreen: bool = False,
        lite: bool = False,
        camera: str = "auto",
        gpio_pin: int = DEFAULT_PIN,
        gpio_active_low: bool = False,
        gpio: bool = True,
        serve: bool = False,
        port: int = 8000,
        token: str | None = None,
        open_mode: bool = False,
        stream_width: int = 640,
        stream_quality: int = 70,
        stream_fps: float = 12.0,
    ):
        self.root = root
        self._lite = lite
        self._camera_pref = camera
        # — capture indicator LED (silent no-op when there is no GPIO) —
        env_pin = os.environ.get("PLANT_GPIO_PIN")
        if env_pin:
            try:
                gpio_pin = int(env_pin)
            except ValueError:
                pass
        self.light = CaptureLight(gpio_pin, active_high=not gpio_active_low, enabled=gpio)
        self._face = _pick_font(root)
        self._face_head = _pick_font(root, heading=True)
        self.root.title("Plant Health")
        self.root.configure(bg=BG)
        self._want_full = fullscreen
        self.root.geometry(f"{W}x{H}")
        if not fullscreen:
            self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind("<Escape>", lambda _e: self._close())
        self.scanner = None
        self.finder = None
        self.photo = None
        self._photo_size: tuple[int, int] | None = None
        self.cap = None
        self._cam_opening = False
        self._frame_pil = None
        self._tick_id = None
        self._tick_n = 0
        self._tick_t0 = 0.0
        self._tick_warned = False
        self._scan_gen = 0
        self._tracks: list[dict] = []
        self._detect_busy = False
        self._grade_busy = False
        self._view_w = 1024
        self._view_h = 556
        self._shutter_enabled = True
        self._picked_tid: int | None = None
        self._disp = {"ox": 0.0, "oy": 0.0, "scale": 1.0}
        self._page = "scan"
        self._gallery_photo = None
        self._gal_src = None
        self._gal_path = None
        self._thumbs: list[ImageTk.PhotoImage] = []
        self._snap_busy = False
        self._flash_until = 0.0
        self._shutter_punch = 0.0
        self._shutter_xy = (0.0, 0.0)
        self._hud_crop = DASH
        self._hud_health = DASH
        self._hud_notes = LOOKING
        self._hud_tone = "muted"
        self._hud_confidence: int | None = None
        self._hud_extra = ""
        self._font_label = _load_font(11, weight="semibold")
        self._font_value = _load_font(26, heading=True)
        self._font_notes = _load_font(14)
        self._font_mini = _load_font(15)
        self._font_pill = _load_font(11, weight="semibold")
        self._font_chip = _load_font(15, weight="bold")
        self._font_nav = _load_font(15, weight="semibold")
        self._font_gallery_title = _load_font(20, heading=True)
        self._font_caption = _load_font(11)
        self._icon_cache: dict[tuple, ImageTk.PhotoImage] = {}
        self._pill_cache: dict[tuple, ImageTk.PhotoImage] = {}
        self._delete_armed_until = 0.0
        self._gal_record: dict | None = None
        self._gal_records: list[dict] = []
        self._gal_card_photo = None
        # — colour tuning (Settings tab) —
        self._color_store = ProfileStore()
        self._color = self._color_store.current().copy()
        self._color_ranges = dict(FALLBACK_RANGES)
        # True once the ISP took the profile; the PIL fallback then costs nothing.
        self._color_native = False
        # First run seeds `night` from whatever AWB converged to, so the shipped
        # look (AwbEnable + Indoor) is the baseline and not a regression.
        self._color_seeded = not self._color_store.first_run
        self._color_seed_frames = 0
        self._color_syncing = False
        self._color_scales: dict[str, tk.Scale] = {}
        self._color_value_labels: dict[str, tk.Label] = {}
        self._profile_btns: dict[str, tk.Label] = {}
        self._settings_status: tk.Label | None = None
        self._settings_photo = None
        self._set_preview_key: tuple | None = None
        # — end colour tuning —
        # — phone remote (Track C). Everything here stays None unless --serve,
        # and src.server is not imported at all in that case. —
        self._serve = bool(serve)
        self._serve_port = int(port)
        self._serve_token = token
        self._serve_open = bool(open_mode)
        self._stream_w = max(160, int(stream_width))
        self._stream_q = max(20, min(95, int(stream_quality)))
        self._stream_fps = float(stream_fps)
        self._bus = None
        self._server = None
        self._remote_label: tk.Label | None = None
        self._scan = tk.Frame(self.root, bg=BG)
        self._gallery = tk.Frame(self.root, bg=BG)
        self._settings = tk.Frame(self.root, bg=BG)
        self._build_scan()
        self._build_gallery()
        self._build_settings()
        self._scan.pack(fill="both", expand=True)
        self._boot()
        if self._want_full:
            self.root.after(80, self._go_fullscreen)

    def _go_fullscreen(self) -> None:
        self.root.update_idletasks()
        sw = max(1, int(self.root.winfo_screenwidth() or W))
        sh = max(1, int(self.root.winfo_screenheight() or H))
        try:
            self.root.attributes("-fullscreen", True)
        except tk.TclError:
            pass
        self.root.geometry(f"{sw}x{sh}+0+0")
        if platform.machine().lower() in ("aarch64", "armv7l", "armv8l"):
            self.root.config(cursor="none")

    def _icon(self, kind: str, size: int, color: str) -> ImageTk.PhotoImage:
        key = (kind, size, color)
        cached = self._icon_cache.get(key)
        if cached is None:
            img = _leaf_icon(size, color) if kind == "leaf" else _trash_icon(size, color)
            cached = ImageTk.PhotoImage(img)
            self._icon_cache[key] = cached
        return cached

    def _pill_photo(
        self,
        text: str,
        *,
        font,
        fg: str,
        bg: str | None = None,
        outline: str | None = None,
        min_w: int = 0,
        min_h: int = 44,
        pad_x: int = 22,
    ) -> ImageTk.PhotoImage:
        key = (text, id(font), fg, bg, outline, min_w, min_h, pad_x)
        cached = self._pill_cache.get(key)
        if cached is not None:
            return cached
        scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        tw, th = _text_size(scratch, text, font)
        w = max(min_w, tw + pad_x * 2)
        h = max(min_h, th + 16)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        radius = h // 2
        if bg is not None:
            draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=bg)
        elif outline is not None:
            draw.rounded_rectangle((1, 1, w - 2, h - 2), radius=radius, outline=outline, width=2)
        draw.text(((w - tw) // 2, (h - th) // 2 - 1), text, font=font, fill=fg)
        photo = ImageTk.PhotoImage(img)
        self._pill_cache[key] = photo
        return photo

    def _build_nav(self, parent: tk.Frame, active: str) -> None:
        top = tk.Frame(
            parent,
            bg=SURFACE,
            height=64,
            highlightthickness=1,
            highlightbackground=DIVIDER,
            highlightcolor=DIVIDER,
        )
        top.pack(fill="x")
        top.pack_propagate(False)
        leaf = self._icon("leaf", 24, LEAF_GREEN)
        tk.Label(top, image=leaf, bg=SURFACE).pack(side="left", padx=(20, 10))
        tk.Label(
            top,
            text="Plant Health",
            bg=SURFACE,
            fg=TEXT,
            font=(self._face_head, 18),
        ).pack(side="left")

        right = tk.Frame(top, bg=SURFACE)
        right.pack(side="right", padx=20)

        exit_img = self._pill_photo("Exit", font=self._font_nav, fg=ALERT, outline=ALERT)
        exit_btn = tk.Label(right, image=exit_img, bg=SURFACE, cursor="hand2")
        exit_btn.pack(side="right", padx=(16, 0))
        exit_btn.bind("<Button-1>", lambda _e: self._close())

        def tab(name: str, label: str, cmd) -> None:
            is_active = active == name
            img = self._pill_photo(
                label,
                font=self._font_nav,
                fg=CREAM if is_active else TEXT,
                bg=ACCENT if is_active else None,
                outline=None if is_active else DIVIDER,
            )
            btn = tk.Label(right, image=img, bg=SURFACE, cursor="hand2")
            btn.pack(side="right", padx=4)
            btn.bind("<Button-1>", lambda _e: cmd())

        # Packed side="right", so the last tab() call lands leftmost:
        # this order paints Scan | Gallery | Settings.
        tab("settings", "Settings", self._show_settings)
        tab("gallery", "Gallery", self._show_gallery)
        tab("scan", "Scan", self._show_scan)

    def _boot(self) -> None:
        self._set_result(DASH, DASH, "Loading inspector…")
        self._start_camera()
        self._tick()
        self._start_server()
        # Torch starves the camera thread on a Pi, and a live view the operator
        # can see beats a grader they can't. Let the sensor come up first.
        self.root.after(1500, lambda: threading.Thread(target=self._load_brains, daemon=True).start())

    def _load_brains(self) -> None:
        scanner = None
        finder = None
        try:
            scanner = Scanner(CKPT, use_dictionary=not self._lite) if CKPT.exists() else None
        except Exception:
            scanner = None
        try:
            finder = PlantFinder(backend="color" if self._lite else "auto")
        except Exception:
            finder = None
        self.root.after(0, lambda: self._after_boot(scanner, finder))

    def _after_boot(self, scanner, finder) -> None:
        self.scanner = scanner
        self.finder = finder
        if scanner is None:
            self._set_result(DASH, DASH, "No grader loaded. Train best.pt first.", tone="warn")
        elif self.cap is not None:
            self._set_result(DASH, DASH, LOOKING)
        # A missing camera is the louder problem; don't paper over it with LOOKING.

    def _build_scan(self) -> None:
        self._build_nav(self._scan, "scan")

        self.stage = tk.Frame(self._scan, bg=VIEW)
        self.stage.pack(fill="both", expand=True)
        self.view = tk.Canvas(self.stage, bg=VIEW, highlightthickness=0, cursor="arrow")
        self.view.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.view.bind("<Button-1>", self._on_view_click)
        self.stage.bind("<Configure>", self._on_stage)

        self._live_img = ImageTk.PhotoImage(self._make_live_badge())
        self.badge = tk.Label(self.stage, image=self._live_img, bg=VIEW, bd=0)
        self.badge.place(x=16, y=16)

    def _set_result(
        self,
        crop: str,
        health: str,
        notes: str,
        *,
        tone: str = "muted",
        confidence: int | None = None,
        extra: str = "",
    ) -> None:
        self._hud_crop = crop or DASH
        self._hud_health = health or DASH
        self._hud_notes = notes or DASH
        self._hud_tone = tone
        self._hud_confidence = confidence
        self._hud_extra = extra
        if self._page == "scan" and self._frame_pil is not None:
            self._show_image(self._frame_pil)

    def _build_gallery(self) -> None:
        self._build_nav(self._gallery, "gallery")
        body = tk.Frame(self._gallery, bg=BG)
        body.pack(fill="both", expand=True)
        left = tk.Frame(
            body,
            bg=SURFACE,
            width=260,
            highlightthickness=1,
            highlightbackground=DIVIDER,
            highlightcolor=DIVIDER,
        )
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._gal_list = tk.Canvas(left, bg=SURFACE, highlightthickness=0)
        scroll = tk.Scrollbar(left, orient="vertical", command=self._gal_list.yview)
        self._gal_list.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._gal_list.pack(side="left", fill="both", expand=True)
        self._gal_items = tk.Frame(self._gal_list, bg=SURFACE)
        self._gal_list.create_window((0, 0), window=self._gal_items, anchor="nw")
        self._gal_items.bind(
            "<Configure>",
            lambda _e: self._gal_list.configure(scrollregion=self._gal_list.bbox("all")),
        )
        right = tk.Frame(body, bg=VIEW)
        right.pack(side="left", fill="both", expand=True)
        self._gal_canvas = tk.Canvas(right, bg=VIEW, highlightthickness=0)
        self._gal_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._gal_canvas.bind("<Configure>", self._paint_gallery)
        self._del_btn = tk.Label(
            right,
            image=self._icon("trash", 20, ALERT),
            bg=SURFACE,
            cursor="hand2",
            width=48,
            height=48,
        )
        self._del_btn.place(relx=1.0, rely=1.0, x=-44, y=-44, anchor="se")
        self._del_btn.bind("<Button-1>", self._on_delete_click)

    # — colour tuning (Settings tab) —————————————————————————————

    _COLOR_SLIDERS = (
        ("red", "Red"),
        ("green", "Green"),
        ("blue", "Blue"),
        ("saturation", "Saturation"),
        ("brightness", "Brightness"),
        ("contrast", "Contrast"),
    )

    def _build_settings(self) -> None:
        self._build_nav(self._settings, "settings")
        body = tk.Frame(self._settings, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(
            body,
            bg=SURFACE,
            width=470,
            highlightthickness=1,
            highlightbackground=DIVIDER,
            highlightcolor=DIVIDER,
        )
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        self._settings_status = tk.Label(
            left,
            text="",
            bg=SURFACE,
            fg=MUTED,
            font=(self._face, 11),
            anchor="w",
        )
        self._settings_status.pack(fill="x", padx=18, pady=(12, 6))

        # Where the operator reads the phone address and token. Empty (and
        # invisible) unless --serve actually brought a server up.
        self._remote_label = tk.Label(
            left,
            text="",
            bg=SURFACE,
            fg=TEXT,
            font=(self._face, 11),
            anchor="w",
            justify="left",
        )
        self._remote_label.pack(fill="x", padx=18, pady=(0, 6))

        for name, label in self._COLOR_SLIDERS:
            row = tk.Frame(left, bg=SURFACE)
            row.pack(fill="x", padx=18, pady=2)
            tk.Label(
                row, text=label, bg=SURFACE, fg=TEXT, font=(self._face, 11), width=10, anchor="w"
            ).pack(side="left")
            value = tk.Label(row, text="", bg=SURFACE, fg=MUTED, font=(self._face, 11), width=6, anchor="e")
            value.pack(side="right")
            low, high = self._color_ranges[name]
            scale = tk.Scale(
                row,
                from_=low,
                to=high,
                resolution=0.01,
                orient="horizontal",
                showvalue=False,
                length=300,
                width=16,
                sliderlength=28,
                bg=SURFACE,
                fg=TEXT,
                troughcolor=NEUTRAL_400,
                activebackground=ACCENT,
                highlightthickness=0,
                bd=0,
                sliderrelief="flat",
                command=lambda v, n=name: self._on_color_slider(n, v),
            )
            scale.pack(side="left", fill="x", expand=True, padx=(8, 8))
            self._color_scales[name] = scale
            self._color_value_labels[name] = value

        def button(parent, text, cmd, *, key=None):
            img = self._pill_photo(text, font=self._font_nav, fg=TEXT, outline=DIVIDER, min_h=40)
            btn = tk.Label(parent, image=img, bg=SURFACE, cursor="hand2")
            btn.pack(side="left", padx=5)
            btn.bind("<Button-1>", lambda _e: cmd())
            if key is not None:
                self._profile_btns[key] = btn
            return btn

        picks = tk.Frame(left, bg=SURFACE)
        picks.pack(fill="x", padx=13, pady=(12, 4))
        button(picks, "Night", lambda: self._activate_profile("night"), key="night")
        button(picks, "Morning", lambda: self._activate_profile("morning"), key="morning")
        button(picks, "Reset", self._reset_color)

        saves = tk.Frame(left, bg=SURFACE)
        saves.pack(fill="x", padx=13, pady=(0, 12))
        button(saves, "Save as Night", lambda: self._save_color("night"))
        button(saves, "Save as Morning", lambda: self._save_color("morning"))

        right = tk.Frame(body, bg=VIEW)
        right.pack(side="left", fill="both", expand=True)
        self._set_canvas = tk.Canvas(right, bg=VIEW, highlightthickness=0)
        self._set_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._set_canvas.bind("<Configure>", lambda _e: self._paint_settings_preview())

        self._sync_color_sliders()
        self._paint_profile_buttons()

    def _on_color_slider(self, name: str, value) -> None:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return
        setattr(self._color, name, val)
        label = self._color_value_labels.get(name)
        if label is not None:
            label.config(text=f"{val:.2f}")
        if not self._color_syncing:
            # Push while dragging so the change is visible; no disk write here.
            self._push_color()

    def _sync_color_sliders(self) -> None:
        """Paint the widgets from `self._color` without re-pushing six times."""
        self._color_syncing = True
        try:
            for name, scale in self._color_scales.items():
                low, high = self._color_ranges[name]
                scale.config(from_=low, to=high)
                scale.set(float(getattr(self._color, name)))
                label = self._color_value_labels.get(name)
                if label is not None:
                    label.config(text=f"{float(getattr(self._color, name)):.2f}")
        finally:
            self._color_syncing = False

    def _paint_profile_buttons(self) -> None:
        active = self._color_store.active
        for key, btn in self._profile_btns.items():
            on = key == active
            img = self._pill_photo(
                key.capitalize(),
                font=self._font_nav,
                fg=CREAM if on else TEXT,
                bg=ACCENT if on else None,
                outline=None if on else DIVIDER,
                min_h=40,
            )
            btn.configure(image=img)
        if self._settings_status is not None:
            mode = "camera" if self._color_native else "software"
            self._settings_status.config(text=f"Active profile: {active.capitalize()}  ·  applied in {mode}")

    def _push_color(self) -> None:
        """Hand the current profile to the ISP. Sets the native flag either way."""
        cap = self.cap
        if cap is None or not hasattr(cap, "set_profile") or not self._color_seeded:
            self._color_native = False
            return
        self._color_native = bool(cap.set_profile(self._color))

    def _activate_profile(self, name: str) -> None:
        self._color = self._color_store.get(name).copy().clamped(self._color_ranges)
        self._color_store.set_active(name)
        self._sync_color_sliders()
        self._push_color()
        self._paint_profile_buttons()

    def _save_color(self, name: str) -> None:
        self._color_store.set_profile(name, self._color.clamped(self._color_ranges), activate=True)
        self._paint_profile_buttons()

    def _reset_color(self) -> None:
        self._color = ColorProfile()
        self._sync_color_sliders()
        self._push_color()
        self._paint_profile_buttons()

    def _maybe_seed_night(self) -> None:
        """Once, on first run, record what AWB converged to as `night`."""
        if self._color_seeded or self.cap is None:
            return
        if not hasattr(self.cap, "metadata"):
            # USB webcam / PC sim: no ISP metadata to read, neutral it is.
            self._finish_seed(None)
            return
        self._color_seed_frames += 1
        if self._color_seed_frames < 30:
            return
        seeded = None
        try:
            seeded = profile_from_gains(self.cap.metadata().get("ColourGains"))
        except Exception:
            seeded = None
        self._finish_seed(seeded)

    def _finish_seed(self, profile: ColorProfile | None) -> None:
        self._color_seeded = True
        base = (profile or ColorProfile()).clamped(self._color_ranges)
        self._color_store.profiles["night"] = base.copy()
        self._color_store.active = "night"
        self._color_store.save()
        self._color = base.copy()
        self._sync_color_sliders()
        self._push_color()
        self._paint_profile_buttons()

    def _paint_settings_preview(self) -> None:
        img = self._frame_pil
        if img is None or self._page != "settings":
            return
        cw = max(1, self._set_canvas.winfo_width())
        ch = max(1, self._set_canvas.winfo_height())
        if cw < 64 or ch < 64:
            return
        show = img.copy()
        show.thumbnail((cw - 24, ch - 24))
        key = (cw, ch, show.size)
        if key != self._set_preview_key or self._settings_photo is None:
            self._set_preview_key = key
            self._settings_photo = ImageTk.PhotoImage(show)
            self._set_canvas.delete("all")
            self._set_canvas.create_image(cw // 2, ch // 2, image=self._settings_photo)
        else:
            self._settings_photo.paste(show)

    # — end colour tuning ——————————————————————————————————————

    def _show_page(self, name: str) -> None:
        pages = {"scan": self._scan, "gallery": self._gallery, "settings": self._settings}
        self._page = name
        for key, frame in pages.items():
            if key != name:
                frame.pack_forget()
        pages[name].pack(fill="both", expand=True)

    def _show_gallery(self) -> None:
        self._show_page("gallery")
        self.root.update_idletasks()
        self._fill_gallery()

    def _show_scan(self) -> None:
        self._show_page("scan")

    def _show_settings(self) -> None:
        self._show_page("settings")
        self.root.update_idletasks()
        self._paint_settings_preview()

    def _relative_time(self, created_at: str) -> str:
        try:
            dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return ""
        secs = (datetime.now(timezone.utc) - dt).total_seconds()
        if secs < 60:
            return "Just now"
        mins = int(secs // 60)
        if mins < 60:
            return f"{mins} min ago"
        hours = int(mins // 60)
        if hours < 24:
            return f"{hours} hr ago"
        days = int(hours // 24)
        return "Yesterday" if days == 1 else f"{days} days ago"

    def _row_photo(self, rec: dict, *, selected: bool) -> Image.Image:
        w, h = 228, 64
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        if selected:
            draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=HUD_R, fill=CARD_FILL, outline=(*_hex_rgb(ACCENT), 255), width=2)
        crop = _display_name(rec.get("crop") or DASH)
        health_key = (rec.get("health") or "").strip().lower()
        tokens = HEALTH_TOKENS.get(health_key)
        title_font = self._font_gallery_title
        tw, th = _text_size(draw, crop, title_font)
        max_title_w = w - 24 - 70
        while crop and _text_size(draw, crop, title_font)[0] > max_title_w:
            crop = crop[:-1]
        tw, th = _text_size(draw, crop, title_font)
        draw.text((12, 8), crop, font=title_font, fill=TEXT_RGB)
        if tokens:
            tag_bg, tag_fg, _ = tokens
            tag_text = _display_name(rec.get("health") or "")
            ttw, tth = _text_size(draw, tag_text, self._font_pill)
            pad = 8
            tag_w, tag_h = ttw + pad * 2, tth + 8
            tx = w - 12 - tag_w
            ty = 8 + (th - tag_h) // 2
            draw.rounded_rectangle((tx, ty, tx + tag_w, ty + tag_h), radius=tag_h // 2, fill=(*tag_bg, 255))
            draw.text((tx + pad, ty + 4), tag_text, font=self._font_pill, fill=tag_fg)
        confidence = rec.get("confidence")
        caption = self._relative_time(rec.get("created_at") or "")
        if isinstance(confidence, (int, float)):
            pct = f"{int(round(confidence * 100))}% confidence"
            caption = f"{caption} · {pct}" if caption else pct
        caption = caption or DASH
        max_caption_w = w - 24
        while caption and _text_size(draw, caption, self._font_caption)[0] > max_caption_w:
            caption = caption[:-1]
        draw.text((12, 36), caption, font=self._font_caption, fill=(*_hex_rgb(MUTED), 255))
        return img

    def _fill_gallery(self) -> None:
        from src.records import list_scans

        for child in self._gal_items.winfo_children():
            child.destroy()
        self._thumbs = []
        self._gal_src = None
        self._gal_path = None
        self._gal_record = None
        self._gal_canvas.delete("all")
        records = [r for r in list_scans() if r.get("image_path") and Path(r["image_path"]).exists()]
        if not records:
            tk.Label(
                self._gal_items,
                text="No photos yet.\nTap the shutter first.",
                bg=SURFACE,
                fg=MUTED,
                font=(self._face, 12),
                padx=16,
                pady=20,
                justify="left",
            ).pack(anchor="w")
            return
        self._gal_records = records
        for rec in records:
            photo = ImageTk.PhotoImage(self._row_photo(rec, selected=False))
            self._thumbs.append(photo)
            item = tk.Label(self._gal_items, image=photo, bg=SURFACE, bd=0)
            item.pack(fill="x", padx=6, pady=3)
            item.bind("<Button-1>", lambda _e, r=rec: self._open_photo(r))
        self._open_photo(records[0])
        self.root.after_idle(self._paint_gallery)

    def _open_photo(self, rec: dict) -> None:
        path = Path(rec["image_path"])
        try:
            self._gal_src = Image.open(path).convert("RGB")
            self._gal_path = path
            self._gal_record = rec
        except Exception:
            return
        self._refresh_gallery_rows()
        self._paint_gallery()

    def _refresh_gallery_rows(self) -> None:
        selected_id = self._gal_record.get("id") if self._gal_record else None
        self._thumbs = []
        for widget, rec in zip(self._gal_items.winfo_children(), self._gal_records):
            photo = ImageTk.PhotoImage(self._row_photo(rec, selected=(rec.get("id") == selected_id)))
            self._thumbs.append(photo)
            widget.configure(image=photo)

    def _make_info_card(self, rec: dict) -> Image.Image:
        crop = _display_name(rec.get("crop") or DASH)
        health_key = (rec.get("health") or "").strip().lower()
        tokens = HEALTH_TOKENS.get(health_key)
        tip = rec.get("tip") or ""
        time_txt = self._relative_time(rec.get("created_at") or "")
        w = 420
        scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        tip_lines = _wrap_lines(scratch, tip, self._font_notes, w - 36, limit=3) if tip else []
        h = 20 + 34 + (len(tip_lines) * 20 if tip_lines else 0) + 16
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=HUD_R, fill=CARD_FILL, outline=CARD_LINE, width=2)
        x, y = 18, 14
        title_font = self._font_gallery_title
        tw, th = _text_size(draw, crop, title_font)
        draw.text((x, y), crop, font=title_font, fill=TEXT_RGB)
        cx = x + tw + 10
        if tokens:
            tag_bg, tag_fg, _ = tokens
        else:
            tag_bg, tag_fg = (235, 221, 197), TEXT_RGB
        tag_text = _display_name(rec.get("health") or "") or DASH
        ttw, tth = _text_size(draw, tag_text, self._font_pill)
        pad = 8
        tag_w, tag_h = ttw + pad * 2, tth + 8
        ty = y + (th - tag_h) // 2
        draw.rounded_rectangle((cx, ty, cx + tag_w, ty + tag_h), radius=tag_h // 2, fill=(*tag_bg, 255))
        draw.text((cx + pad, ty + 4), tag_text, font=self._font_pill, fill=tag_fg)
        if time_txt:
            cx2 = cx + tag_w + 10
            draw.text((cx2, y + (th - 12) // 2), time_txt, font=self._font_mini, fill=(*_hex_rgb(MUTED), 255))
        ny = y + th + 10
        for line in tip_lines:
            draw.text((x, ny), line, font=self._font_notes, fill=(*TEXT_RGB, 216))
            ny += 20
        return img

    def _on_delete_click(self, _event=None) -> None:
        if self._gal_record is None:
            return
        now = time.monotonic()
        if now < self._delete_armed_until:
            from src.records import delete_scan

            rec = self._gal_record
            self._delete_armed_until = 0.0
            delete_scan(rec["id"], rec["image_path"])
            self._del_btn.configure(bg=SURFACE, image=self._icon("trash", 20, ALERT))
            self._fill_gallery()
            return
        self._delete_armed_until = now + 2.5
        self._del_btn.configure(bg=ALERT, image=self._icon("trash", 20, "#ffffff"))
        self.root.after(2600, self._reset_delete_arm)

    def _reset_delete_arm(self) -> None:
        if time.monotonic() >= self._delete_armed_until:
            self._del_btn.configure(bg=SURFACE, image=self._icon("trash", 20, ALERT))

    def _paint_gallery(self, event=None) -> None:
        if self._gal_src is None or self._page != "gallery":
            return
        cw = event.width if event is not None else self._gal_canvas.winfo_width()
        ch = event.height if event is not None else self._gal_canvas.winfo_height()
        cw = max(1, cw)
        ch = max(1, ch)
        if cw < 64 or ch < 64:
            return
        inset = 24
        show = self._gal_src.copy()
        show.thumbnail((max(1, cw - inset * 2), max(1, ch - inset * 2)))
        self._gallery_photo = ImageTk.PhotoImage(show)
        self._gal_canvas.delete("all")
        self._gal_canvas.create_image(cw // 2, ch // 2, image=self._gallery_photo)
        if self._gal_record is not None:
            card = self._make_info_card(self._gal_record)
            self._gal_card_photo = ImageTk.PhotoImage(card)
            self._gal_canvas.create_image(44, ch - 44, image=self._gal_card_photo, anchor="sw")

    def _dash_store(self, value: str) -> str:
        text = (value or "").strip()
        if not text or text == DASH:
            return ""
        return text

    def _persist_capture(self, composed: Image.Image) -> None:
        from src.records import add_scan, write_png

        track = self._focus_track()
        named = ""
        confidence = None
        if track:
            named = track.get("named_plant") or ""
            confidence = track.get("health_confidence")
        try:
            path = write_png(composed)
            add_scan(
                crop=self._dash_store(self._hud_crop),
                health=self._dash_store(self._hud_health),
                named_plant=named,
                tip=self._dash_store(self._hud_notes),
                image_path=path,
                confidence=confidence,
            )
        except Exception:
            pass

    def _on_stage(self, event: tk.Event) -> None:
        if event.widget is not self.stage:
            return
        self._view_w = max(1, event.width)
        self._view_h = max(1, event.height)
        if self._frame_pil is not None:
            self._show_image(self._frame_pil)
        else:
            self._sync_shutter()
            self._place_live()

    def _value_ready(self, value: str) -> bool:
        text = (value or "").strip().lower()
        return text not in {"", "—", "-", "unknown"} and text != DASH.lower()

    def _notes_kind(self) -> str:
        notes = (self._hud_notes or "").strip()
        if notes in {"", DASH}:
            return "hidden"
        idle = {
            "Point the camera at a plant.",
            "Loading inspector…",
            "Loading inspector...",
        }
        if notes in idle:
            return "hidden"
        if notes.rstrip(".") == LOOKING:
            return "mini"
        return "full"

    def _hud_boxes(self) -> dict[str, tuple[int, int, int, int]]:
        boxes: dict[str, tuple[int, int, int, int]] = {}
        x1, y = 16, 72
        card_h = 78
        gap = 10
        if self._value_ready(self._hud_crop):
            boxes["type"] = (x1, y, x1 + HUD_W, y + card_h)
            y += card_h + gap
        if self._value_ready(self._hud_health):
            health_h = 112 if self._hud_confidence is not None else card_h
            boxes["health"] = (x1, y, x1 + HUD_W, y + health_h)
        kind = self._notes_kind()
        if kind == "full":
            notes_w = min(560, self._view_w - 32)
            boxes["notes"] = (16, self._view_h - 126, 16 + notes_w, self._view_h - 16)
        elif kind == "mini":
            scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))
            tw, _ = _text_size(scratch, self._hud_notes, self._font_mini)
            w = min(self._view_w - 24, max(160, tw + 28))
            h = 40
            boxes["notes"] = (12, self._view_h - 12 - h, 12 + w, self._view_h - 12)
        return boxes

    def _chip_boxes(self) -> dict[int, tuple[int, int, int, int]]:
        boxes: dict[int, tuple[int, int, int, int]] = {}
        if len(self._tracks) < 2:
            return boxes
        ranked = sorted(self._tracks, key=_box_area, reverse=True)[:2]
        size, gap = 40, 8
        total_w = size * 2 + gap
        x0 = (self._view_w - total_w) // 2
        y0 = 16
        for i, track in enumerate(ranked):
            x = x0 + i * (size + gap)
            boxes[track.get("tid")] = (x, y0, x + size, y0 + size)
        return boxes

    def _make_live_badge(self) -> Image.Image:
        font = self._font_pill
        text = "LIVE"
        scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        tw, th = _text_size(scratch, text, font)
        dot, gap, pad_x, pad_y = 7, 6, 14, 7
        w = pad_x * 2 + dot + gap + tw
        h = th + pad_y * 2
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=ALERT)
        cy = h // 2
        draw.ellipse((pad_x, cy - dot // 2, pad_x + dot, cy + dot // 2), fill="#ffffff")
        draw.text((pad_x + dot + gap, (h - th) // 2 - 1), text, font=font, fill="#ffffff")
        return img

    def _place_live(self) -> None:
        self.badge.place(x=16, y=16)

    def _draw_card(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        title: str,
        value: str,
        value_rgb: tuple[int, int, int],
        *,
        fill: tuple[int, int, int] = CARD_FILL_RGB,
        border: tuple[int, int, int] = CARD_LINE_RGB,
        confidence: int | None = None,
    ) -> None:
        x1, y1, x2, y2 = box
        draw.rounded_rectangle(box, radius=HUD_R, fill=fill, outline=border, width=2)
        inset = 16
        draw.text((x1 + inset, y1 + 10), title.upper(), font=self._font_label, fill=LABEL_RGB)
        max_w = x2 - x1 - inset * 2
        text = value or DASH
        while text and _text_size(draw, text, self._font_value)[0] > max_w:
            text = text[:-1]
        if text != (value or DASH):
            text = text[:-1] + "…" if len(text) > 1 else "…"
        draw.text((x1 + inset, y1 + 32), text, font=self._font_value, fill=value_rgb)
        if confidence is not None:
            pill_text = f"{confidence}% confident"
            ptw, pth = _text_size(draw, pill_text, self._font_pill)
            pad_x, pad_y = 10, 5
            pw, ph = ptw + pad_x * 2, pth + pad_y * 2
            px, py = x1 + inset, y1 + 70
            draw.rounded_rectangle((px, py, px + pw, py + ph), radius=ph // 2, outline=value_rgb, width=2)
            draw.text((px + pad_x, py + pad_y - 1), pill_text, font=self._font_pill, fill=value_rgb)

    def _cover_frame(self, img: Image.Image, vw: int, vh: int) -> Image.Image:
        iw, ih = img.size
        if iw < 1 or ih < 1:
            self._disp = {"ox": 0.0, "oy": 0.0, "scale": 1.0}
            return Image.new("RGB", (vw, vh), VIEW_RGB)
        if (iw, ih) == (vw, vh):
            self._disp = {"ox": 0.0, "oy": 0.0, "scale": 1.0}
            return img.copy()
        if iw == vw and ih > vh:
            top = (ih - vh) // 2
            self._disp = {"ox": 0.0, "oy": float(-top), "scale": 1.0}
            return img.crop((0, top, vw, top + vh)).copy()
        if ih == vh and iw > vw:
            left = (iw - vw) // 2
            self._disp = {"ox": float(-left), "oy": 0.0, "scale": 1.0}
            return img.crop((left, 0, left + vw, vh)).copy()
        scale = max(vw / iw, vh / ih)
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        show = img.resize((nw, nh), Image.BILINEAR)
        ox, oy = (vw - nw) // 2, (vh - nh) // 2
        if ox < 0 or oy < 0:
            show = show.crop((-ox, -oy, -ox + vw, -oy + vh))
        elif show.size != (vw, vh):
            canvas = Image.new("RGB", (vw, vh), VIEW_RGB)
            canvas.paste(show, (ox, oy))
            show = canvas
        self._disp = {"ox": float(ox), "oy": float(oy), "scale": scale}
        return show

    def _blend_patch(self, show: Image.Image, box: tuple[int, int, int, int], draw_fn) -> None:
        """Alpha-blend a small region in place, instead of compositing the whole frame."""
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(show.width, x2), min(show.height, y2)
        if x2 <= x1 or y2 <= y1:
            return
        patch = show.crop((x1, y1, x2, y2)).convert("RGBA")
        overlay = Image.new("RGBA", patch.size, (0, 0, 0, 0))
        draw_fn(ImageDraw.Draw(overlay))
        show.paste(Image.alpha_composite(patch, overlay).convert("RGB"), (x1, y1))

    def _dash_rect(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        color: str,
        width: int,
        dash: int = 10,
        gap: int = 6,
    ) -> None:
        x1, y1, x2, y2 = box

        def dash_line(p1: tuple[float, float], p2: tuple[float, float]) -> None:
            x1_, y1_ = p1
            x2_, y2_ = p2
            length = math.hypot(x2_ - x1_, y2_ - y1_)
            if length <= 0:
                return
            step = dash + gap
            t = 0.0
            while t < 1.0:
                t_end = min(1.0, t + dash / length)
                xa, ya = x1_ + (x2_ - x1_) * t, y1_ + (y2_ - y1_) * t
                xb, yb = x1_ + (x2_ - x1_) * t_end, y1_ + (y2_ - y1_) * t_end
                draw.line((xa, ya, xb, yb), fill=color, width=width)
                t += step / length

        dash_line((x1, y1), (x2, y1))
        dash_line((x2, y1), (x2, y2))
        dash_line((x2, y2), (x1, y2))
        dash_line((x1, y2), (x1, y1))

    def _draw_tracks(self, img: Image.Image) -> Image.Image:
        if not self._tracks:
            return img
        out = img
        draw = ImageDraw.Draw(out)
        focus = self._focus_track()
        focus_tid = focus.get("tid") if focus else None
        for track in self._tracks:
            x1, y1, x2, y2 = [int(v) for v in track["xyxy"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(out.size[0] - 1, x2), min(out.size[1] - 1, y2)
            if track.get("tid") == focus_tid:
                draw.rounded_rectangle((x1, y1, x2, y2), radius=8, outline=ACCENT_2_400, width=3)
            else:
                self._dash_rect(draw, (x1, y1, x2, y2), NEUTRAL_400, 3)
        return out

    def _draw_tracks_view(self, img: Image.Image) -> Image.Image:
        return self._draw_tracks_mapped(
            img,
            self._disp.get("ox", 0.0),
            self._disp.get("oy", 0.0),
            self._disp.get("scale", 1.0),
        )

    def _draw_tracks_mapped(
        self, img: Image.Image, ox: float, oy: float, scale: float
    ) -> Image.Image:
        """Draw the detection boxes onto `img` using an explicit image->img map.

        The kiosk canvas and the phone stream cover-fit the same source frame
        into different rectangles, so each passes its own offset/scale rather
        than sharing `self._disp`.
        """
        mapped = []
        for track in self._tracks:
            x1, y1, x2, y2 = track["xyxy"]
            item = dict(track)
            item["xyxy"] = (
                x1 * scale + ox,
                y1 * scale + oy,
                x2 * scale + ox,
                y2 * scale + oy,
            )
            mapped.append(item)
        saved = self._tracks
        self._tracks = mapped
        try:
            return self._draw_tracks(img)
        finally:
            self._tracks = saved

    def _draw_note_tail(
        self,
        show: Image.Image,
        draw: ImageDraw.ImageDraw,
        pos: tuple[int, int],
        line: str,
        extra: str,
        max_w: int,
        note_rgb: tuple[int, int, int],
    ) -> None:
        x, y = pos
        tw, th = _text_size(draw, line, self._font_notes)
        draw.text((x, y), line, font=self._font_notes, fill=note_rgb)
        remaining = max_w - tw
        if remaining <= 10:
            return
        text = extra
        while text and _text_size(draw, text, self._font_notes)[0] > remaining:
            text = text[:-1]
        if not text:
            return
        if text != extra and len(text) > 1:
            text = text[:-1] + "…"
        etw, eth = _text_size(draw, text, self._font_notes)

        def paint(patch_draw: ImageDraw.ImageDraw) -> None:
            patch_draw.text((0, 0), text, font=self._font_notes, fill=(*note_rgb, 165))

        self._blend_patch(show, (x + tw, y, x + tw + etw + 2, y + eth + 4), paint)

    def _draw_chips(self, show: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        chips = self._chip_boxes()
        if not chips:
            return
        ranked = sorted(self._tracks, key=_box_area, reverse=True)[:2]
        focus = self._focus_track()
        focus_tid = focus.get("tid") if focus else None
        for i, track in enumerate(ranked):
            tid = track.get("tid")
            box = chips.get(tid)
            if box is None:
                continue
            active = tid == focus_tid
            label = str(i + 1)
            x1, y1, x2, y2 = box
            if active:
                draw.ellipse(box, fill=_hex_rgb(ACCENT))
                tw, th = _text_size(draw, label, self._font_chip)
                draw.text(
                    ((x1 + x2) // 2 - tw // 2, (y1 + y2) // 2 - th // 2 - 1),
                    label,
                    font=self._font_chip,
                    fill=_hex_rgb(CREAM),
                )
                continue

            def paint(patch_draw: ImageDraw.ImageDraw, label=label, size=(x2 - x1, y2 - y1)) -> None:
                patch_draw.ellipse((0, 0, size[0], size[1]), fill=(20, 20, 20, 140), outline=(255, 255, 255, 230), width=2)
                tw, th = _text_size(patch_draw, label, self._font_chip)
                patch_draw.text(
                    (size[0] // 2 - tw // 2, size[1] // 2 - th // 2 - 1),
                    label,
                    font=self._font_chip,
                    fill=(255, 255, 255, 255),
                )

            self._blend_patch(show, box, paint)

    def _compose_frame(self, img: Image.Image) -> Image.Image:
        vw = max(1, self._view_w)
        vh = max(1, self._view_h)
        show = self._cover_frame(img, vw, vh)
        if self._tracks:
            show = self._draw_tracks_view(show)
        draw = ImageDraw.Draw(show)
        boxes = self._hud_boxes()
        health_key = (self._hud_health or "").strip().lower()
        tokens = HEALTH_TOKENS.get(health_key)
        if "type" in boxes:
            self._draw_card(draw, boxes["type"], "Plant type", _display_name(self._hud_crop), TEXT_RGB)
        if "health" in boxes:
            if tokens:
                bg_rgb, text_rgb, border_rgb = tokens
                fill, border, value_rgb = bg_rgb, border_rgb, text_rgb
            else:
                fill, border, value_rgb = CARD_FILL_RGB, CARD_LINE_RGB, TEXT_RGB
            self._draw_card(
                draw,
                boxes["health"],
                "Plant health",
                _display_name(self._hud_health),
                value_rgb,
                fill=fill,
                border=border,
                confidence=self._hud_confidence,
            )
        self._draw_chips(show, draw)
        if "notes" in boxes:
            notes_box = boxes["notes"]
            nx1, ny1, nx2, ny2 = notes_box
            draw.rounded_rectangle(notes_box, radius=HUD_R, fill=CARD_FILL_RGB, outline=CARD_LINE_RGB, width=2)
            note_rgb = WARN_RGB if self._hud_tone == "warn" else TEXT_RGB
            if self._notes_kind() == "mini":
                lines = _wrap_lines(draw, self._hud_notes, self._font_mini, nx2 - nx1 - 28, limit=1)
                draw.text((nx1 + 14, ny1 + 12), lines[0], font=self._font_mini, fill=note_rgb)
            else:
                draw.text((nx1 + 18, ny1 + 10), "NOTES", font=self._font_label, fill=LABEL_RGB)
                max_w = nx2 - nx1 - 36
                lines = _wrap_lines(draw, self._hud_notes, self._font_notes, max_w, limit=3)
                ty = ny1 + 34
                for i, line in enumerate(lines):
                    if i == len(lines) - 1 and self._hud_extra:
                        self._draw_note_tail(show, draw, (nx1 + 18, ty), line, self._hud_extra, max_w, note_rgb)
                    else:
                        draw.text((nx1 + 18, ty), line, font=self._font_notes, fill=note_rgb)
                    ty += 22
        return show

    def _sync_shutter(self) -> None:
        r = SHUTTER_R
        x = max(r + 16, self._view_w - r - SHUTTER_EDGE_INSET)
        y = max(r + 16, self._view_h // 2)
        self._shutter_xy = (x, y)
        on = self._shutter_enabled
        ring = "#ffffff" if on else "#8a8378"
        disk = "#ffffff" if on else "#8a8378"
        well = ""
        punch = time.monotonic() < self._shutter_punch
        key = (x, y, on, punch)
        if key == getattr(self, "_shutter_key", None) and self.view.find_withtag("shutter"):
            return
        self._shutter_key = key
        self.view.delete("shutter")
        self.view.create_oval(
            x - r,
            y - r,
            x + r,
            y + r,
            outline=ring,
            width=SHUTTER_RING,
            fill=well,
            tags="shutter",
        )
        ir = r - SHUTTER_RING - (10 if punch else 4)
        self.view.create_oval(
            x - ir,
            y - ir,
            x + ir,
            y + ir,
            outline="",
            fill=disk,
            tags="shutter",
        )
        self.view.tag_bind("shutter", "<Button-1>", self._on_shutter)
        self.view.tag_bind("shutter", "<Enter>", lambda _e: self.view.config(cursor="hand2"))
        self.view.tag_bind("shutter", "<Leave>", lambda _e: self.view.config(cursor="arrow"))
        self.view.tag_raise("shutter")

    def _on_shutter(self, _event=None) -> None:
        if self._shutter_enabled:
            self.snap()

    def _in_hud(self, x: float, y: float) -> bool:
        for box in self._hud_boxes().values():
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def _on_view_click(self, event: tk.Event) -> None:
        sx, sy = self._shutter_xy
        if (event.x - sx) ** 2 + (event.y - sy) ** 2 <= (SHUTTER_R + 8) ** 2:
            return
        for tid, (x1, y1, x2, y2) in self._chip_boxes().items():
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self._picked_tid = tid
                self._paint_primary()
                if self._frame_pil is not None:
                    self._show_image(self._frame_pil)
                return
        if self._in_hud(event.x, event.y):
            return
        if not self._tracks or self._disp["scale"] <= 0:
            return
        ix = (event.x - self._disp["ox"]) / self._disp["scale"]
        iy = (event.y - self._disp["oy"]) / self._disp["scale"]
        hits = []
        for track in self._tracks:
            x1, y1, x2, y2 = track["xyxy"]
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                hits.append(track)
        if not hits:
            self._picked_tid = None
            self._paint_primary()
            if self._frame_pil is not None:
                self._show_image(self._frame_pil)
            return
        pick = min(hits, key=_box_area)
        self._picked_tid = pick.get("tid")
        self._paint_primary()
        if self._frame_pil is not None:
            self._show_image(self._frame_pil)

    def _view_message(self, text: str) -> None:
        self.view.delete("frame")
        self.photo = None
        self._photo_size = None
        self.view.delete("empty")
        self.view.create_text(
            self._view_w // 2,
            self._view_h // 2,
            text=text,
            fill=MUTED,
            font=(self._face, 16),
            tags="empty",
        )

    def _start_camera(self) -> None:
        """Open the camera off the UI thread.

        Opening it inline blocked __init__ before the first paint, so a slow or
        wedged libcamera showed up as a blank cream screen instead of the app.
        """
        if self.cap is not None or self._cam_opening:
            return
        self._cam_opening = True
        self._shutter_enabled = False
        self._set_result(DASH, DASH, "Starting camera…")
        self._sync_shutter()
        self._place_live()
        self._view_message("Starting camera…")
        threading.Thread(target=self._open_camera_worker, daemon=True).start()

    def _open_camera_worker(self) -> None:
        cap = None
        try:
            cap = open_capture(self._camera_pref)
        except Exception as exc:
            print(f"camera open failed: {exc}", flush=True)
        self.root.after(0, lambda c=cap: self._after_camera(c))

    def _after_camera(self, cap) -> None:
        self._cam_opening = False
        if cap is None:
            self._shutter_enabled = False
            self._set_result(DASH, DASH, NO_CAMERA, tone="warn")
            self._sync_shutter()
            self._place_live()
            self._view_message(NO_CAMERA)
            return
        self.cap = cap
        if hasattr(cap, "control_ranges"):
            self._color_ranges = merge_ranges(cap.control_ranges())
            self._color = self._color.clamped(self._color_ranges)
            self._sync_color_sliders()
        self._push_color()
        self._paint_profile_buttons()
        self._shutter_enabled = True
        if self.scanner is None:
            self._set_result(DASH, DASH, "Loading inspector…")
        else:
            self._set_result(DASH, DASH, LOOKING)
        self._sync_shutter()
        self._place_live()

    def _tick(self) -> None:
        # Settings needs live frames too, or the sliders tune a still picture.
        # Gallery keeps its cheap re-arm — unless a phone is watching, in which
        # case the stream would otherwise freeze the moment the kiosk is left
        # on Gallery. Frames are still only serviced, not painted.
        bus = self._bus
        watched = bus is not None and bus.client_count > 0
        if self._page not in ("scan", "settings") and not watched:
            self._tick_id = self.root.after(250, self._tick)
            return
        live = self._page == "scan"
        t0 = time.monotonic()
        try:
            if self.cap is not None:
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    self._frame_pil = _frame_to_pil(frame, rgb=bool(getattr(self.cap, "rgb", False)))
                    if not self._color_native:
                        # Only when the ISP could not do it. Native means zero
                        # per-frame cost; apply_pil is a no-op when neutral.
                        self._frame_pil = self._color.apply_pil(self._frame_pil)
                    self._maybe_seed_night()
                    if self._page == "settings":
                        # No detect/grade here: YOLO would make the sliders lag.
                        self._paint_settings_preview()
                        self._stream_only(self._frame_pil)
                    elif not live:
                        # Gallery with a remote viewer: stream only, no canvas.
                        self._stream_only(self._frame_pil)
                    else:
                        self._show_image(self._frame_pil)
                        self._kick_detect()
                        self._kick_grade()
                        now = time.monotonic()
                        if self._tick_n == 0:
                            self._tick_t0 = now
                        self._tick_n += 1
                        if self._tick_n in (45, 150, 300):
                            dt = now - self._tick_t0
                            print(f"live {self._tick_n / dt:.1f} fps over {dt:.1f}s", flush=True)
                            self._tick_t0 = now
                            self._tick_n = 0
            if live and time.monotonic() < max(self._flash_until, self._shutter_punch):
                self._sync_shutter()
        except Exception as exc:
            # Swallowing this silently is how a broken frame path looks
            # identical to a dead camera. Say it once, then stay quiet.
            if not self._tick_warned:
                self._tick_warned = True
                traceback.print_exc()
                print(f"live frame failed: {exc}", flush=True)
        finally:
            # Re-arm only after the work, and always leave Tk idle time.
            # Re-arming first kept the timer queue permanently ready, which
            # starved the canvas redraw: the loop ran flat out while the
            # picture on the panel sat frozen. The finally also means a raising
            # read() can no longer kill the loop outright.
            spent = (time.monotonic() - t0) * 1000.0
            delay = max(TICK_FLOOR_MS, int(TICK_BUDGET_MS - spent))
            self._tick_id = self.root.after(delay, self._tick)

    def _kick_detect(self) -> None:
        if self._detect_busy or self._frame_pil is None or self.finder is None:
            return
        src = self._frame_pil
        gen = self._scan_gen
        self._detect_busy = True

        def work() -> None:
            boxes = []
            try:
                iw, ih = src.size
                det_scale = 1.0
                img = src
                if iw > 640:
                    det_scale = 640.0 / iw
                    img = src.resize((640, max(1, int(ih * det_scale))), Image.BILINEAR)
                boxes = self.finder.find(img)
                if det_scale != 1.0:
                    inv = 1.0 / det_scale
                    for box in boxes:
                        x1, y1, x2, y2 = box["xyxy"]
                        box["xyxy"] = (x1 * inv, y1 * inv, x2 * inv, y2 * inv)
            except Exception:
                boxes = []
            self.root.after(0, lambda b=boxes, g=gen: self._after_detect(g, b))

        threading.Thread(target=work, daemon=True).start()

    def _after_detect(self, gen: int, boxes: list[dict]) -> None:
        self._detect_busy = False
        if gen != self._scan_gen:
            return
        self._tracks = match_tracks(self._tracks, boxes)
        if self._picked_tid is not None and not any(t.get("tid") == self._picked_tid for t in self._tracks):
            self._picked_tid = None
        n = len(self._tracks)
        if n == 0:
            self._clear_findings(LOOKING)
        elif not any(t.get("stage") == "graded" for t in self._tracks):
            self._set_result(DASH, DASH, "")
        else:
            self._paint_primary()

    def _kick_grade(self) -> None:
        if self._grade_busy or self.scanner is None or self._frame_pil is None:
            return
        found = [t for t in self._tracks if t.get("stage") == "found"]
        named = [t for t in self._tracks if t.get("stage") == "named"]
        picked = self._picked_tid
        if found:
            track = next((t for t in found if t.get("tid") == picked), None) or max(found, key=_box_area)
            detail = self._lite
        elif named:
            if self._lite:
                for track in named:
                    track["stage"] = "graded"
                    if not track.get("crop"):
                        track["crop"] = None
                    if not track.get("tip"):
                        track["tip"] = "Cannot name this crop yet."
                    track["label"] = _track_label(track)
                self._paint_primary()
                if self._frame_pil is not None:
                    self._show_image(self._frame_pil)
                return
            track = next((t for t in named if t.get("tid") == picked), None) or max(named, key=_box_area)
            detail = True
        else:
            return
        img = self._frame_pil.copy()
        xyxy = tuple(track["xyxy"])
        gen = self._scan_gen
        self._grade_busy = True

        def work() -> None:
            try:
                crop = crop_xyxy(img, xyxy)
                result = self.scanner.scan(crop, detail=detail, assume_plant=True)
                err = None
            except Exception as exc:
                result = None
                err = exc
            self.root.after(0, lambda r=result, e=err, g=gen, b=xyxy, d=detail: self._after_grade(g, b, d, r, e))

        threading.Thread(target=work, daemon=True).start()

    def _after_grade(self, gen: int, xyxy: tuple, detail: bool, result: dict | None, err: Exception | None) -> None:
        try:
            if gen != self._scan_gen:
                return
            if err is not None or result is None:
                return
            if not self._tracks:
                self._clear_findings()
                return
            target = None
            best = 0.0
            for track in self._tracks:
                score = _iou(track["xyxy"], xyxy)
                if score > best:
                    best, target = score, track
            if target is None or best < 0.2:
                return
            locked = target.get("stage") == "graded" and target.get("crop") in FARM_CROPS
            if not result["unknown"]:
                target["crop"] = result["crop"]
                target["health"] = result["health"]
                target["named_plant"] = None
                target["tip"] = result.get("tip") or ""
                target["health_confidence"] = result.get("health_confidence")
                target["stage"] = "graded"
                target["label"] = _track_label(target)
            elif detail and not locked:
                target["named_plant"] = result.get("named_plant")
                target["crop"] = None if result["crop"] == "unknown" else result["crop"]
                target["health"] = None if result["health"] == "unknown" else result["health"]
                target["tip"] = result.get("tip") or ""
                target["health_confidence"] = result.get("health_confidence")
                target["stage"] = "graded"
                target["label"] = _track_label(target)
            elif not detail:
                target["stage"] = "named"
                target["label"] = "plant"
            self._paint_primary()
            if self._frame_pil is not None:
                self._show_image(self._frame_pil)
        finally:
            self._grade_busy = False

    def _show_image(self, img: Image.Image) -> None:
        composed = self._compose_frame(img)
        if time.monotonic() < self._flash_until:
            white = Image.new("RGB", composed.size, (255, 255, 255))
            composed = Image.blend(composed, white, 0.62)
        # The phone gets a card-free frame built at stream size (boxes only);
        # the HUD text reaches it as HTML via /api/status instead.
        self._publish_stream(img)
        # Building a fresh PhotoImage and re-creating the canvas item every
        # frame was the most expensive thing in the loop. Paste into the
        # existing image instead; only rebuild when the view size changes.
        if self.photo is None or self._photo_size != composed.size:
            self.photo = ImageTk.PhotoImage(composed)
            self._photo_size = composed.size
            self.view.delete("frame")
            self.view.delete("empty")
            self.view.create_image(
                self._view_w // 2, self._view_h // 2, image=self.photo, tags="frame"
            )
            self.view.tag_lower("frame")
        else:
            self.photo.paste(composed)
        self._sync_shutter()
        self._place_live()

    def snap(self) -> None:
        if self._snap_busy or not self._shutter_enabled:
            return
        if self._frame_pil is None:
            self._set_result(DASH, DASH, "No camera frame yet.", tone="warn")
            return
        self.light.pulse()
        self._snap_busy = True
        self._flash_until = time.monotonic() + 0.16
        self._shutter_punch = time.monotonic() + 0.18
        img = self._frame_pil.copy()
        composed = self._compose_frame(img)
        self._show_image(img)

        def work() -> None:
            try:
                self._persist_capture(composed)
            finally:
                self.root.after(180, self._unlock_snap)

        threading.Thread(target=work, daemon=True).start()

    def _unlock_snap(self) -> None:
        self._snap_busy = False

    def _clear_findings(self, notes: str = "", *, tone: str = "muted") -> None:
        self._set_result(DASH, DASH, notes, tone=tone)

    def _focus_track(self) -> dict | None:
        if not self._tracks:
            return None
        if self._picked_tid is not None:
            for track in self._tracks:
                if track.get("tid") == self._picked_tid:
                    return track
        return max(self._tracks, key=_box_area)

    def _box_number(self, track: dict | None) -> int:
        if track is None:
            return 1
        ranked = sorted(self._tracks, key=_box_area, reverse=True)
        for i, item in enumerate(ranked, start=1):
            if item.get("tid") == track.get("tid"):
                return i
        return 1

    def _paint_primary(self) -> None:
        track = self._focus_track()
        if track is None:
            return
        n = len(self._tracks)
        extra = (
            f" Box {self._box_number(track)} of {n} — tap the other box to inspect it."
            if n > 1
            else ""
        )
        crop = track.get("crop")
        health = track.get("health")
        named = track.get("named_plant")
        tip = track.get("tip") or ""
        health_txt = health if health and health != "unknown" else DASH
        confidence = track.get("health_confidence")
        conf_pct = int(round(confidence * 100)) if isinstance(confidence, (int, float)) else None
        if track.get("stage") != "graded":
            self._set_result(DASH, DASH, "")
            return
        if crop and crop != "unknown":
            self._set_result(str(crop), health_txt, tip or DASH, confidence=conf_pct, extra=extra)
            return
        if named:
            plant = str(named).split(" (")[0]
            notes = tip or "Cannot name this crop yet."
            self._set_result(plant, health_txt, notes, tone="warn", confidence=conf_pct, extra=extra)
            return
        notes = tip or "Cannot name this crop yet."
        self._set_result(DASH, DASH, notes, tone="warn", extra=extra)

    # — phone remote (Track C) ————————————————————————————————————

    def _start_server(self) -> None:
        """Bring the LAN server up. No-op, and no import, without --serve."""
        if not self._serve:
            return
        try:
            from src.frame_bus import FrameBus
            from src.server import KioskServer, lan_ip

            bus = FrameBus(max_fps=self._stream_fps)
            server = KioskServer(
                self,
                bus=bus,
                port=self._serve_port,
                token=self._serve_token,
                stream_width=self._stream_w,
                open_mode=self._serve_open,
            )
            server.start()
        except Exception as exc:
            print(f"remote: server did not start ({exc})", flush=True)
            return
        self._bus = bus
        self._server = server
        host = lan_ip()
        if self._serve_open:
            print(f"remote: http://{host}:{server.port}/  (open mode, no token)", flush=True)
            print(
                "remote: OPEN MODE - the stream and controls are reachable by ANYONE "
                "on the network with no authentication.",
                flush=True,
            )
        else:
            print(f"remote: http://{host}:{server.port}/  token {server.token}", flush=True)
        print(
            "remote: plain HTTP on the local network - the stream is NOT encrypted.",
            flush=True,
        )
        if self._remote_label is not None:
            if self._serve_open:
                self._remote_label.config(text=f"Remote: http://{host}:{server.port}\n(open, no token)")
            else:
                self._remote_label.config(
                    text=f"Remote: http://{host}:{server.port}\nToken: {server.token}"
                )

    def _stream_frame(self, src: Image.Image) -> Image.Image:
        """Build the phone's frame directly at stream resolution.

        The phone renders Plant type / health / Notes as real HTML, so the
        streamed pixels carry only the detection boxes — those are spatial and
        mean nothing away from the plant they outline. Building straight from
        the raw camera frame at ~stream size (instead of composing a second
        1024-wide HUD frame and shrinking it) keeps the extra per-tick cost to
        one small resize plus a handful of rectangles.
        """
        vw = max(1, self._view_w)
        vh = max(1, self._view_h)
        sw = max(1, min(self._stream_w, vw))
        sh = max(1, round(vh * sw / vw))
        iw, ih = src.size
        if iw < 1 or ih < 1:
            return Image.new("RGB", (sw, sh), VIEW_RGB)
        # Same cover-fit as the kiosk canvas, so the phone sees the same framing.
        scale = max(sw / iw, sh / ih)
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        out = src if (nw, nh) == (iw, ih) else src.resize((nw, nh), Image.BILINEAR)
        ox = (sw - nw) // 2
        oy = (sh - nh) // 2
        # crop() always copies, so the shared self._frame_pil is never drawn on.
        out = out.crop((-ox, -oy, -ox + sw, -oy + sh))
        if self._tracks:
            out = self._draw_tracks_mapped(out, ox, oy, scale)
        if time.monotonic() < self._flash_until:
            white = Image.new("RGB", out.size, (255, 255, 255))
            out = Image.blend(out, white, 0.62)
        return out

    def _publish_stream(self, src: Image.Image) -> None:
        """Encode one JPEG for the phone. Cheap no-op when nobody is watching.

        Skipped outright while YOLO holds a worker: on four A72 cores an extra
        full-frame encode per tick is visible in the live framerate, and a
        dropped remote frame is the cheaper loss.
        """
        bus = self._bus
        if bus is None or self._detect_busy:
            return
        if not bus.wants_frame(time.monotonic()):
            return
        try:
            frame = self._stream_frame(src)
        except Exception:
            return
        self._publish_jpeg(frame)

    def _publish_jpeg(self, img: Image.Image) -> None:
        bus = self._bus
        if bus is None:
            return
        try:
            import cv2
            import numpy as np

            if img.width > self._stream_w:
                height = max(1, round(img.height * self._stream_w / img.width))
                img = img.resize((self._stream_w, height), Image.BILINEAR)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.asarray(img)[:, :, ::-1]  # PIL RGB -> cv2 BGR
            ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), self._stream_q])
            if ok:
                bus.publish(buf.tobytes())
        except Exception:
            # A broken encode must never take the display path down with it.
            pass

    def _stream_only(self, img: Image.Image) -> None:
        """Publish for a remote viewer without painting the canvas.

        This is the Gallery/Settings page with somebody watching from a phone:
        the stream has to keep moving, but the scan canvas is not on screen.
        """
        self._publish_stream(img)

    def _call_on_tk(self, fn, timeout: float = 1.5):
        """Run `fn` on the Tk thread from an HTTP thread and return its result.

        Bounded on purpose: a wedged UI turns into a 503 for the phone, not a
        hung request holding one of the three handler threads forever.
        """
        done = threading.Event()
        box: dict = {}

        def run() -> None:
            try:
                box["value"] = fn()
            except Exception as exc:  # noqa: BLE001 - re-raised on the caller
                box["error"] = exc
            finally:
                done.set()

        self.root.after(0, run)
        if not done.wait(timeout):
            raise TimeoutError("kiosk UI thread did not respond")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    # — the KioskBridge interface. All of these run on an HTTP thread. —

    def remote_shutter(self) -> dict:
        return self._call_on_tk(self._remote_snap)

    def _remote_snap(self) -> dict:
        # On the Tk thread, so these are the same preconditions snap() itself
        # checks and the answer we hand the phone cannot go stale in between.
        if self._snap_busy:
            return {"ok": False, "reason": "a capture is already running"}
        if not self._shutter_enabled:
            return {"ok": False, "reason": "the shutter is disabled right now"}
        if self._frame_pil is None:
            return {"ok": False, "reason": "no camera frame yet"}
        self.snap()
        return {"ok": True, "reason": "capture started"}

    def remote_color(self) -> dict:
        return {
            "profile": self._color.to_dict(),
            "ranges": {k: [float(v[0]), float(v[1])] for k, v in self._color_ranges.items()},
            "active": self._color_store.active,
            "native": bool(self._color_native),
            "sliders": [{"name": n, "label": lbl} for n, lbl in self._COLOR_SLIDERS],
        }

    def remote_hud(self) -> dict:
        """The HUD text the phone renders as HTML cards.

        Deliberately *not* marshalled through `_call_on_tk`: this is read on
        every status poll, the six fields are plain attribute reads (each one
        atomic in CPython), and bouncing it off the Tk thread would both add
        work to the render loop and let a busy UI turn the phone's liveness
        poll into a 503. The worst case is a torn read that mixes one grade's
        crop with the next grade's health for a single poll.
        """
        health = (self._hud_health or "").strip()
        key = health.lower()
        return {
            "crop": _display_name(self._hud_crop),
            "health": key if key in HEALTH_DISPLAY else "",
            "health_label": _display_name(health),
            "notes": self._hud_notes or "",
            "notes_extra": self._hud_extra or "",
            "tone": self._hud_tone or "muted",
            "confidence": self._hud_confidence,
        }

    def remote_set_slider(self, name: str, value: float) -> dict:
        if name not in dict(self._COLOR_SLIDERS):
            raise KeyError(f"unknown slider {name!r}")
        low, high = self._color_ranges[name]
        val = min(float(high), max(float(low), float(value)))
        self._call_on_tk(lambda: self._remote_apply_slider(name, val))
        return self.remote_color()

    def _remote_apply_slider(self, name: str, value: float) -> None:
        # Drive the model first, then the widget. Going the other way round --
        # scale.set() and letting the widget's -command fire back -- looks
        # tidier but silently loses the edit: set() moves the slider without
        # invoking the callback, so self._color never changed and every phone
        # adjustment was dropped while reset/activate (which assign _color
        # directly) appeared to work. The sync guard keeps the widget move from
        # re-entering if a later Tk build does fire the command.
        self._on_color_slider(name, value)
        scale = self._color_scales.get(name)
        if scale is not None:
            self._color_syncing = True
            try:
                scale.set(value)
            finally:
                self._color_syncing = False

    def remote_profile(self, action: str, name: str | None = None) -> dict:
        if action in ("activate", "save") and name not in ("night", "morning"):
            raise ValueError(f"unknown profile {name!r}")
        self._call_on_tk(lambda: self._remote_profile(action, name))
        return self.remote_color()

    def _remote_profile(self, action: str, name: str | None) -> None:
        if action == "activate":
            self._activate_profile(str(name))
        elif action == "save":
            self._save_color(str(name))
        elif action == "reset":
            self._reset_color()
        else:
            raise ValueError(f"unknown action {action!r}")
        self._sync_color_sliders()
        self._paint_profile_buttons()

    def remote_gallery_changed(self) -> None:
        self.root.after(0, self._remote_refill_gallery)

    def _remote_refill_gallery(self) -> None:
        if self._page == "gallery":
            self._fill_gallery()

    # — end phone remote ——————————————————————————————————————————

    def _close(self) -> None:
        self._scan_gen += 1
        server = self._server
        self._server = None
        self._bus = None
        if server is not None:
            # From the Tk thread while serve_forever runs on its own: calling
            # shutdown() from a serving thread deadlocks, and the join inside
            # stop() is bounded so a wedged viewer cannot hold up exit.
            server.stop()
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.light.close()
        self.root.destroy()


def main() -> None:
    from src.kiosk import main as kiosk_main

    kiosk_main()


if __name__ == "__main__":
    main()
