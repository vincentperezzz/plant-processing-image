import platform
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tkinter as tk
import tkinter.font as tkfont

from PIL import Image, ImageDraw, ImageFont, ImageTk

from src.camera import open_capture
from src.detect import PlantFinder, _iou, crop_xyxy, draw_boxes, match_tracks
from src.infer import FARM_CROPS, Scanner
from src.paths import CKPT, SCANS_DIR
from src.scan_drop import MUTED, TEXT, _box_area, _frame_to_pil, _track_label

W = 1024
H = 600
BG = "#0c0e0d"
PANEL = "#141816"
VIEW = "#050505"
CARD = "#121614"
CHIP = "#1c221e"
CRIT = "#ff453a"
LIVE_BG = "#2c2c2e"
LABEL = "#8a9a8e"
DASH = "—"
LOOKING = "Looking for plant"
HEALTH_RGB = {
    "healthy": (52, 199, 89),
    "mild": (255, 214, 10),
    "critical": (255, 69, 58),
    "dead": (142, 142, 147),
}
SHUTTER_R = 36
SHUTTER_RING = 5
HUD_W = 236
HUD_R = 18
VIEW_RGB = (5, 5, 5)
TEXT_RGB = (241, 248, 233)
LABEL_RGB = (220, 228, 222)
WARN_RGB = (255, 168, 162)
CARD_FILL = (12, 16, 14, 230)
CARD_LINE = (200, 212, 204, 140)
HEALTH_FILL = {
    "healthy": (16, 64, 32, 235),
    "mild": (72, 58, 6, 235),
    "critical": (82, 18, 18, 238),
    "dead": (42, 42, 46, 235),
}


def _pick_font(root: tk.Tk) -> str:
    want = ("Segoe UI", "DejaVu Sans", "Piboto", "Liberation Sans", "FreeSans")
    have = set(tkfont.families(root))
    for name in want:
        if name in have:
            return name
    return "TkDefaultFont"


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = []
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
    def __init__(self, root: tk.Tk, *, fullscreen: bool = False, lite: bool = False, camera: str = "auto"):
        self.root = root
        self._lite = lite
        self._camera_pref = camera
        self._face = _pick_font(root)
        self.root.title("Plant Health")
        self.root.geometry(f"{W}x{H}")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)
        if fullscreen:
            try:
                self.root.attributes("-fullscreen", True)
            except tk.TclError:
                pass
            if platform.machine().lower() in ("aarch64", "armv7l", "armv8l"):
                self.root.config(cursor="none")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind("<Escape>", lambda _e: self._close())
        self.scanner = None
        self.finder = None
        self.photo = None
        self.cap = None
        self._frame_pil = None
        self._tick_id = None
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
        self._font_label = _load_font(14, bold=True)
        self._font_value = _load_font(26, bold=True)
        self._font_notes = _load_font(17)
        self._font_mini = _load_font(15)
        self._scan = tk.Frame(self.root, bg=BG)
        self._gallery = tk.Frame(self.root, bg=BG)
        self._build_scan()
        self._build_gallery()
        self._scan.pack(fill="both", expand=True)
        self._boot()

    def _boot(self) -> None:
        self._set_result(DASH, DASH, "Loading inspector…")
        self._start_camera()
        self._tick()
        threading.Thread(target=self._load_brains, daemon=True).start()

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
        else:
            self._set_result(DASH, DASH, LOOKING)

    def _build_scan(self) -> None:
        top = tk.Frame(self._scan, bg=PANEL, height=44)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(
            top,
            text="Plant Health",
            bg=PANEL,
            fg=TEXT,
            font=(self._face, 16, "bold"),
        ).pack(side="left", padx=16)
        self._exit_btn = tk.Label(
            top,
            text="EXIT",
            bg=CRIT,
            fg="#ffffff",
            font=(self._face, 11, "bold"),
            padx=14,
            pady=6,
        )
        self._exit_btn.pack(side="right", padx=(0, 12))
        self._exit_btn.bind("<Button-1>", lambda _e: self._close())
        self._gal_btn = tk.Label(
            top,
            text="GALLERY",
            bg=CHIP,
            fg=TEXT,
            font=(self._face, 11, "bold"),
            padx=14,
            pady=6,
        )
        self._gal_btn.pack(side="right", padx=12)
        self._gal_btn.bind("<Button-1>", lambda _e: self._show_gallery())

        self.stage = tk.Frame(self._scan, bg=VIEW)
        self.stage.pack(fill="both", expand=True)
        self.view = tk.Canvas(self.stage, bg=VIEW, highlightthickness=0, cursor="arrow")
        self.view.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.view.bind("<Button-1>", self._on_view_click)
        self.stage.bind("<Configure>", self._on_stage)

        self.badge = tk.Label(
            self.stage,
            text="LIVE",
            bg=LIVE_BG,
            fg=TEXT,
            font=(self._face, 11, "bold"),
            padx=12,
            pady=4,
            anchor="center",
        )
        self.badge.place(x=0, y=12)

    def _nav_btn(self, parent, text: str, cmd) -> tk.Label:
        btn = tk.Label(
            parent,
            text=text,
            bg=CHIP,
            fg=TEXT,
            font=(self._face, 11, "bold"),
            padx=14,
            pady=6,
        )
        btn.pack(side="left", padx=(0, 8))
        btn.bind("<Button-1>", lambda _e: cmd())
        return btn

    def _set_result(self, crop: str, health: str, notes: str, *, tone: str = "muted") -> None:
        self._hud_crop = crop or DASH
        self._hud_health = health or DASH
        self._hud_notes = notes or DASH
        self._hud_tone = tone
        if self._page == "scan" and self._frame_pil is not None:
            self._show_image(self._frame_pil)

    def _build_gallery(self) -> None:
        top = tk.Frame(self._gallery, bg=PANEL, height=44)
        top.pack(fill="x")
        top.pack_propagate(False)
        self._nav_btn(top, "BACK", self._show_scan)
        tk.Label(
            top,
            text="Gallery",
            bg=PANEL,
            fg=TEXT,
            font=(self._face, 16, "bold"),
        ).pack(side="left", padx=8)
        bye = tk.Label(
            top,
            text="EXIT",
            bg=CRIT,
            fg="#ffffff",
            font=(self._face, 11, "bold"),
            padx=14,
            pady=6,
        )
        bye.pack(side="right", padx=12)
        bye.bind("<Button-1>", lambda _e: self._close())
        where = tk.Label(
            top,
            text=str(SCANS_DIR),
            bg=PANEL,
            fg=LABEL,
            font=(self._face, 10),
        )
        where.pack(side="right", padx=12)
        body = tk.Frame(self._gallery, bg=BG)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=CARD, width=220)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._gal_list = tk.Canvas(left, bg=CARD, highlightthickness=0)
        scroll = tk.Scrollbar(left, orient="vertical", command=self._gal_list.yview)
        self._gal_list.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._gal_list.pack(side="left", fill="both", expand=True)
        self._gal_items = tk.Frame(self._gal_list, bg=CARD)
        self._gal_list.create_window((0, 0), window=self._gal_items, anchor="nw")
        self._gal_items.bind(
            "<Configure>",
            lambda _e: self._gal_list.configure(scrollregion=self._gal_list.bbox("all")),
        )
        self._gal_canvas = tk.Canvas(body, bg=VIEW, highlightthickness=0)
        self._gal_canvas.pack(side="left", fill="both", expand=True)
        self._gal_canvas.bind("<Configure>", self._paint_gallery)

    def _show_gallery(self) -> None:
        self._page = "gallery"
        self._scan.pack_forget()
        self._gallery.pack(fill="both", expand=True)
        self.root.update_idletasks()
        self._fill_gallery()

    def _show_scan(self) -> None:
        self._page = "scan"
        self._gallery.pack_forget()
        self._scan.pack(fill="both", expand=True)

    def _fill_gallery(self) -> None:
        from src.records import list_photos

        for child in self._gal_items.winfo_children():
            child.destroy()
        self._thumbs = []
        self._gal_src = None
        self._gal_path = None
        self._gal_canvas.delete("all")
        photos = list_photos()
        if not photos:
            tk.Label(
                self._gal_items,
                text="No photos yet.\nTap the shutter first.",
                bg=CARD,
                fg=MUTED,
                font=(self._face, 12),
                padx=16,
                pady=20,
                justify="left",
            ).pack(anchor="w")
            return
        first = None
        for path in photos:
            try:
                thumb = Image.open(path).convert("RGB")
                thumb.thumbnail((188, 110))
                photo = ImageTk.PhotoImage(thumb)
            except Exception:
                continue
            if first is None:
                first = path
            self._thumbs.append(photo)
            item = tk.Label(self._gal_items, image=photo, bg=CHIP, bd=0)
            item.pack(fill="x", padx=8, pady=6)
            item.bind("<Button-1>", lambda _e, p=path: self._open_photo(p))
        if first is not None:
            self._open_photo(first)
            self.root.after_idle(self._paint_gallery)

    def _open_photo(self, path: Path) -> None:
        try:
            self._gal_src = Image.open(path).convert("RGB")
            self._gal_path = path
        except Exception:
            return
        self._paint_gallery()

    def _paint_gallery(self, event=None) -> None:
        if self._gal_src is None or self._page != "gallery":
            return
        cw = event.width if event is not None else self._gal_canvas.winfo_width()
        ch = event.height if event is not None else self._gal_canvas.winfo_height()
        cw = max(1, cw)
        ch = max(1, ch)
        if cw < 64 or ch < 64:
            return
        show = self._gal_src.copy()
        show.thumbnail((cw, ch))
        self._gallery_photo = ImageTk.PhotoImage(show)
        self._gal_canvas.delete("all")
        self._gal_canvas.create_image(cw // 2, ch // 2, image=self._gallery_photo)

    def _dash_store(self, value: str) -> str:
        text = (value or "").strip()
        if not text or text == DASH:
            return ""
        return text

    def _persist_capture(self, composed: Image.Image) -> None:
        from src.records import add_scan, write_png

        track = self._focus_track()
        named = ""
        if track:
            named = track.get("named_plant") or ""
        try:
            path = write_png(composed)
            add_scan(
                crop=self._dash_store(self._hud_crop),
                health=self._dash_store(self._hud_health),
                named_plant=named,
                tip=self._dash_store(self._hud_notes),
                image_path=path,
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
        x1, y = 14, 48
        card_h = 78
        gap = 10
        if self._value_ready(self._hud_crop):
            boxes["type"] = (x1, y, x1 + HUD_W, y + card_h)
            y += card_h + gap
        if self._value_ready(self._hud_health):
            boxes["health"] = (x1, y, x1 + HUD_W, y + card_h)
        kind = self._notes_kind()
        if kind == "full":
            boxes["notes"] = (12, self._view_h - 120, self._view_w - 12, self._view_h - 12)
        elif kind == "mini":
            scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))
            tw, _ = _text_size(scratch, self._hud_notes, self._font_mini)
            w = min(self._view_w - 24, max(160, tw + 28))
            h = 40
            boxes["notes"] = (12, self._view_h - 12 - h, 12 + w, self._view_h - 12)
        return boxes

    def _place_live(self) -> None:
        w = 88
        self.badge.place(x=14, y=12, width=w, height=28)

    def _draw_card(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        title: str,
        value: str,
        value_rgb: tuple[int, int, int],
        *,
        fill: tuple[int, int, int, int] = CARD_FILL,
        accent: tuple[int, int, int] | None = None,
    ) -> None:
        outline = (*accent, 220) if accent else CARD_LINE
        draw.rounded_rectangle(box, radius=HUD_R, fill=fill, outline=outline, width=2)
        x1, y1, x2, y2 = box
        if accent:
            draw.rounded_rectangle((x1 + 6, y1 + 10, x1 + 12, y2 - 10), radius=3, fill=(*accent, 255))
            pip = 11
            cx, cy = x2 - 22, (y1 + y2) // 2
            draw.ellipse((cx - pip, cy - pip, cx + pip, cy + pip), fill=(*accent, 255))
        inset = 22 if accent else 16
        draw.text((x1 + inset, y1 + 10), title, font=self._font_label, fill=LABEL_RGB)
        max_w = x2 - x1 - inset - (40 if accent else 16)
        text = value or DASH
        while text and _text_size(draw, text, self._font_value)[0] > max_w:
            text = text[:-1]
        if text != (value or DASH):
            text = text[:-1] + "…" if len(text) > 1 else "…"
        draw.text((x1 + inset, y1 + 34), text, font=self._font_value, fill=value_rgb)

    def _cover_frame(self, img: Image.Image, vw: int, vh: int) -> Image.Image:
        iw, ih = img.size
        if iw < 1 or ih < 1:
            self._disp = {"ox": 0.0, "oy": 0.0, "scale": 1.0}
            return Image.new("RGB", (vw, vh), VIEW_RGB)
        scale = max(vw / iw, vh / ih)
        nw = max(1, int(round(iw * scale)))
        nh = max(1, int(round(ih * scale)))
        resized = img.resize((nw, nh), Image.BILINEAR)
        left = max(0, (nw - vw) // 2)
        top = max(0, (nh - vh) // 2)
        filled = resized.crop((left, top, left + vw, top + vh))
        if filled.size != (vw, vh):
            canvas = Image.new("RGB", (vw, vh), VIEW_RGB)
            canvas.paste(filled, (0, 0))
            filled = canvas
        self._disp = {"ox": float(-left), "oy": float(-top), "scale": scale}
        return filled.convert("RGB")

    def _compose_frame(self, img: Image.Image) -> Image.Image:
        vw = max(1, self._view_w)
        vh = max(1, self._view_h)
        focus = self._focus_track()
        selected = focus.get("tid") if focus else None
        framed = draw_boxes(img, self._tracks, selected_tid=selected, captions=False) if self._tracks else img
        show = self._cover_frame(framed, vw, vh)
        base = Image.new("RGBA", (vw, vh), (*VIEW_RGB, 255))
        base.paste(show.convert("RGBA"), (0, 0))
        overlay = Image.new("RGBA", (vw, vh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        boxes = self._hud_boxes()
        health_key = (self._hud_health or "").strip().lower()
        health_rgb = HEALTH_RGB.get(health_key)
        if "type" in boxes:
            self._draw_card(draw, boxes["type"], "Plant type", self._hud_crop, TEXT_RGB)
        if "health" in boxes:
            self._draw_card(
                draw,
                boxes["health"],
                "Plant health",
                self._hud_health,
                health_rgb or TEXT_RGB,
                fill=HEALTH_FILL.get(health_key, CARD_FILL),
                accent=health_rgb,
            )
        if "notes" in boxes:
            notes_box = boxes["notes"]
            nx1, ny1, nx2, ny2 = notes_box
            draw.rounded_rectangle(notes_box, radius=HUD_R, fill=CARD_FILL, outline=CARD_LINE, width=2)
            note_rgb = WARN_RGB if self._hud_tone == "warn" else TEXT_RGB
            if self._notes_kind() == "mini":
                lines = _wrap_lines(draw, self._hud_notes, self._font_mini, nx2 - nx1 - 28, limit=1)
                draw.text((nx1 + 14, ny1 + 12), lines[0], font=self._font_mini, fill=note_rgb)
            else:
                draw.text((nx1 + 18, ny1 + 10), "Notes", font=self._font_label, fill=LABEL_RGB)
                lines = _wrap_lines(draw, self._hud_notes, self._font_notes, nx2 - nx1 - 36, limit=3)
                ty = ny1 + 34
                for line in lines:
                    draw.text((nx1 + 18, ty), line, font=self._font_notes, fill=note_rgb)
                    ty += 22
        return Image.alpha_composite(base, overlay).convert("RGB")

    def _sync_shutter(self) -> None:
        r = SHUTTER_R
        x = max(r + 16, self._view_w - r - 20)
        y = max(r + 16, self._view_h // 2)
        self._shutter_xy = (x, y)
        self.view.delete("shutter")
        on = self._shutter_enabled
        ring = "#ffffff" if on else "#636366"
        disk = "#ffffff" if on else "#636366"
        well = "#1c1c1e"
        punch = time.monotonic() < self._shutter_punch
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
        ir = r - SHUTTER_RING - (12 if punch else 6)
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

    def _start_camera(self) -> None:
        if self.cap is not None:
            return
        cap = open_capture(self._camera_pref)
        if cap is None:
            self._shutter_enabled = False
            self._set_result(DASH, DASH, "No camera. Check the ribbon or plug in a USB webcam.", tone="warn")
            self._sync_shutter()
            self._place_live()
            self.view.delete("empty")
            self.view.create_text(
                self._view_w // 2,
                self._view_h // 2,
                text="No camera. Check the ribbon or plug in a USB webcam.",
                fill=MUTED,
                font=(self._face, 16),
                tags="empty",
            )
            return
        self.cap = cap
        self._shutter_enabled = True
        if self.scanner is None:
            self._set_result(DASH, DASH, "Loading inspector…")
        else:
            self._set_result(DASH, DASH, LOOKING)
        self._sync_shutter()
        self._place_live()

    def _tick(self) -> None:
        if self._page != "scan":
            self._tick_id = self.root.after(250, self._tick)
            return
        if self.cap is not None:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                try:
                    self._frame_pil = _frame_to_pil(frame)
                    self._show_image(self._frame_pil)
                    self._kick_detect()
                    self._kick_grade()
                except Exception:
                    pass
        if time.monotonic() < max(self._flash_until, self._shutter_punch):
            self._sync_shutter()
        self._tick_id = self.root.after(33, self._tick)

    def _kick_detect(self) -> None:
        if self._detect_busy or self._frame_pil is None or self.finder is None:
            return
        img = self._frame_pil.copy()
        gen = self._scan_gen
        self._detect_busy = True

        def work() -> None:
            try:
                boxes = self.finder.find(img)
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
                target["stage"] = "graded"
                target["label"] = _track_label(target)
            elif detail and not locked:
                target["named_plant"] = result.get("named_plant")
                target["crop"] = None if result["crop"] == "unknown" else result["crop"]
                target["health"] = None if result["health"] == "unknown" else result["health"]
                target["tip"] = result.get("tip") or ""
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
        self.photo = ImageTk.PhotoImage(composed)
        self.view.delete("frame")
        self.view.delete("empty")
        self.view.create_image(self._view_w // 2, self._view_h // 2, image=self.photo, tags="frame")
        self.view.tag_lower("frame")
        self._sync_shutter()
        self._place_live()

    def snap(self) -> None:
        if self._snap_busy or not self._shutter_enabled:
            return
        if self._frame_pil is None:
            self._set_result(DASH, DASH, "No camera frame yet.", tone="warn")
            return
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
        extra = f"Box {self._box_number(track)} of {n}. Tap another box to inspect it." if n > 1 else ""
        crop = track.get("crop")
        health = track.get("health")
        named = track.get("named_plant")
        tip = track.get("tip") or ""
        health_txt = health if health and health != "unknown" else DASH
        if track.get("stage") != "graded":
            self._set_result(DASH, DASH, "")
            return
        if crop and crop != "unknown":
            notes = tip or extra or DASH
            if tip and extra:
                notes = f"{tip}  {extra}"
            self._set_result(str(crop), health_txt, notes)
            return
        if named:
            plant = str(named).split(" (")[0]
            notes = tip or "Cannot name this crop yet."
            if extra:
                notes = f"{notes}  {extra}"
            self._set_result(plant, health_txt, notes, tone="warn")
            return
        notes = tip or "Cannot name this crop yet."
        if extra:
            notes = f"{notes}  {extra}"
        self._set_result(DASH, DASH, notes, tone="warn")

    def _close(self) -> None:
        self._scan_gen += 1
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.root.destroy()


def main() -> None:
    from src.kiosk import main as kiosk_main

    kiosk_main()


if __name__ == "__main__":
    main()
