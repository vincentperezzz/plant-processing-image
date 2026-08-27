import math
import platform
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tkinter as tk
import tkinter.font as tkfont

from PIL import Image, ImageDraw, ImageTk

from src.camera import open_capture
from src.detect import PlantFinder, _iou, crop_xyxy, draw_boxes, match_tracks
from src.infer import FARM_CROPS, Scanner
from src.paths import CKPT
from src.scan_drop import (
    BG,
    MUTED,
    PANEL,
    TEXT,
    WARN,
    _box_area,
    _frame_to_pil,
    _track_label,
)

W = 1024
H = 600
INK = "#102010"
VIEW = "#070c08"
CARD = "#1a281c"
CHIP = "#243328"
OK = "#43a047"
MILD = "#ffb300"
CRIT = "#e53935"
GONE = "#7f0000"
LIVE_BG = "#455a64"
FREEZE_BG = "#607d8b"
SNAP = "#eceff1"
HEALTH_CHIP = {
    "healthy": (OK, "#ffffff"),
    "mild": (MILD, INK),
    "critical": (CRIT, "#ffffff"),
    "dead": (GONE, "#ffffff"),
}


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _draw_camera(d: ImageDraw.ImageDraw, s: int, fill: tuple[int, int, int], ink: tuple[int, int, int]) -> None:
    cx = s / 2
    cy = s / 2 + s * 0.03
    body = (cx - s * 0.22, cy - s * 0.08, cx + s * 0.22, cy + s * 0.16)
    d.rounded_rectangle(body, radius=max(4, s * 0.045), fill=ink)
    hump = (cx - s * 0.11, cy - s * 0.18, cx + s * 0.03, cy - s * 0.05)
    d.rounded_rectangle(hump, radius=max(3, s * 0.03), fill=ink)
    lx, ly, r = cx + s * 0.02, cy + s * 0.04, s * 0.09
    d.ellipse((lx - r, ly - r, lx + r, ly + r), fill=fill)
    r2 = r * 0.42
    d.ellipse((lx - r2, ly - r2, lx + r2, ly + r2), fill=ink)


def _draw_retake(d: ImageDraw.ImageDraw, s: int, ink: tuple[int, int, int]) -> None:
    m = s * 0.30
    w = max(4, int(s * 0.07))
    d.arc((m, m, s - m, s - m), start=50, end=310, fill=ink, width=w)
    cx = s / 2
    cy = s / 2
    r = (s - 2 * m) / 2
    ang = math.radians(50)
    ax = cx + r * math.cos(ang)
    ay = cy - r * math.sin(ang)
    tx = math.cos(ang + math.pi / 2)
    ty = -math.sin(ang + math.pi / 2)
    ah = s * 0.10
    aw = s * 0.08
    p1 = (ax + tx * aw, ay + ty * aw)
    p2 = (ax - tx * aw, ay - ty * aw)
    p3 = (ax + math.cos(ang) * ah, ay - math.sin(ang) * ah)
    d.polygon([p1, p2, p3], fill=ink)


def _shutter_icon(size: int, kind: str) -> Image.Image:
    scale = 3
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if kind == "off":
        fill = _rgb("#3a4a3c")
        ink = _rgb("#8a9a88")
    else:
        fill = _rgb(SNAP)
        ink = _rgb(INK)
    d.ellipse((6, 6, s - 7, s - 7), fill=(0, 0, 0, 110))
    d.ellipse((18, 18, s - 19, s - 19), fill=fill + (255,))
    d.ellipse((18, 18, s - 19, s - 19), outline=(255, 255, 255, 70), width=max(3, s // 40))
    if kind == "retake":
        _draw_retake(d, s, ink)
    else:
        _draw_camera(d, s, fill, ink)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _pick_font(root: tk.Tk) -> str:
    want = ("Segoe UI", "DejaVu Sans", "Piboto", "Liberation Sans", "FreeSans")
    have = set(tkfont.families(root))
    for name in want:
        if name in have:
            return name
    return "TkDefaultFont"


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
        self._mode = "live"
        self._tick_id = None
        self._scan_gen = 0
        self._tracks: list[dict] = []
        self._detect_busy = False
        self._grade_busy = False
        self._view_w = 1008
        self._view_h = 390
        self._shutter_enabled = True
        self._picked_tid: int | None = None
        self._disp = {"ox": 0.0, "oy": 0.0, "scale": 1.0}
        self._page = "scan"
        self._log_photo = None
        self._log_rows: list[dict] = []
        self._scan = tk.Frame(self.root, bg=BG)
        self._log = tk.Frame(self.root, bg=BG)
        self._build_scan()
        self._build_log()
        self._scan.pack(fill="both", expand=True)
        self._boot()

    def _boot(self) -> None:
        self.hint.config(text="Loading inspector…", fg=MUTED)
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
            self.hint.config(text="No grader loaded. Train best.pt first.", fg=WARN)
        elif self._mode == "live":
            self.hint.config(text="Point the camera at a plant.", fg=MUTED)

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
        self._log_btn = tk.Label(
            top,
            text="LOG",
            bg=CHIP,
            fg=TEXT,
            font=(self._face, 11, "bold"),
            padx=14,
            pady=6,
        )
        self._log_btn.pack(side="right", padx=12)
        self._log_btn.bind("<Button-1>", lambda _e: self._show_log())

        sheet = tk.Frame(self._scan, bg=BG)
        sheet.pack(side="bottom", fill="x")
        card = tk.Frame(sheet, bg=CARD)
        card.pack(fill="x", padx=10, pady=(8, 10))
        tk.Label(
            card,
            text="DETECTION",
            bg=CARD,
            fg=MUTED,
            font=(self._face, 9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=16, pady=(10, 0))
        self.chip_row = tk.Frame(card, bg=CARD)
        self.chip_row.pack(fill="x", padx=16, pady=(8, 4))
        self.hint = tk.Label(
            card,
            text="Point the camera at a plant.",
            bg=CARD,
            fg=MUTED,
            font=(self._face, 12),
            wraplength=980,
            justify="left",
            anchor="w",
        )
        self.hint.pack(fill="x", padx=16, pady=(0, 12))
        card.bind("<Configure>", self._on_card)

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
            font=(self._face, 10, "bold"),
            padx=12,
            pady=4,
        )
        self.badge.place(x=14, y=12)

        self._icon_snap = ImageTk.PhotoImage(_shutter_icon(84, "snap"))
        self._icon_hold = ImageTk.PhotoImage(_shutter_icon(84, "retake"))
        self._icon_off = ImageTk.PhotoImage(_shutter_icon(84, "off"))
        self._set_chips([])

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

    def _build_log(self) -> None:
        top = tk.Frame(self._log, bg=PANEL, height=44)
        top.pack(fill="x")
        top.pack_propagate(False)
        self._nav_btn(top, "BACK", self._show_scan)
        tk.Label(
            top,
            text="Scan log",
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
        exp = tk.Label(
            top,
            text="EXPORT CSV",
            bg=CHIP,
            fg=TEXT,
            font=(self._face, 11, "bold"),
            padx=14,
            pady=6,
        )
        exp.pack(side="right", padx=(0, 8))
        exp.bind("<Button-1>", lambda _e: self._export_csv())
        body = tk.Frame(self._log, bg=BG)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=CARD, width=340)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._log_list = tk.Canvas(left, bg=CARD, highlightthickness=0)
        scroll = tk.Scrollbar(left, orient="vertical", command=self._log_list.yview)
        self._log_list.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._log_list.pack(side="left", fill="both", expand=True)
        self._log_items = tk.Frame(self._log_list, bg=CARD)
        self._log_list.create_window((0, 0), window=self._log_items, anchor="nw")
        self._log_items.bind(
            "<Configure>",
            lambda _e: self._log_list.configure(scrollregion=self._log_list.bbox("all")),
        )
        right = tk.Frame(body, bg=VIEW)
        right.pack(side="left", fill="both", expand=True)
        self._log_canvas = tk.Canvas(right, bg=VIEW, highlightthickness=0)
        self._log_canvas.pack(fill="both", expand=True)
        self._log_meta = tk.Label(
            right,
            text="Tap a scan.",
            bg=CARD,
            fg=MUTED,
            font=(self._face, 12),
            wraplength=640,
            justify="left",
            anchor="w",
            padx=16,
            pady=12,
        )
        self._log_meta.pack(fill="x", side="bottom")

    def _show_log(self) -> None:
        self._page = "log"
        self._scan.pack_forget()
        self._log.pack(fill="both", expand=True)
        self._fill_log()

    def _show_scan(self) -> None:
        self._page = "scan"
        self._log.pack_forget()
        self._scan.pack(fill="both", expand=True)

    def _fill_log(self) -> None:
        from src.records import list_scans

        for child in self._log_items.winfo_children():
            child.destroy()
        self._log_rows = list_scans()
        if not self._log_rows:
            tk.Label(
                self._log_items,
                text="No snaps yet. Freeze a scan first.",
                bg=CARD,
                fg=MUTED,
                font=(self._face, 12),
                padx=16,
                pady=20,
            ).pack(anchor="w")
            self._log_meta.config(text="No records.")
            self._log_canvas.delete("all")
            return
        for row in self._log_rows:
            when = row.get("created_at") or ""
            label = when.replace("T", "  ").replace("Z", "")
            item = tk.Label(
                self._log_items,
                text=label,
                bg=CHIP,
                fg=TEXT,
                font=(self._face, 12, "bold"),
                padx=12,
                pady=14,
                anchor="w",
                width=28,
            )
            item.pack(fill="x", padx=8, pady=4)
            item.bind("<Button-1>", lambda _e, r=row: self._open_record(r))
        self._open_record(self._log_rows[0])

    def _open_record(self, row: dict) -> None:
        path = row.get("image_path") or ""
        when = row.get("created_at") or ""
        self._log_meta.config(text=when.replace("T", "  ").replace("Z", "  UTC"))
        self._log_canvas.delete("all")
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return
        cw = max(200, self._log_canvas.winfo_width())
        ch = max(160, self._log_canvas.winfo_height())
        show = img.copy()
        show.thumbnail((cw, ch))
        self._log_photo = ImageTk.PhotoImage(show)
        self._log_canvas.create_image(cw // 2, ch // 2, image=self._log_photo)

    def _export_csv(self) -> None:
        from src.records import export_csv

        path = export_csv()
        self._log_meta.config(text=f"Exported {path}")

    def _save_snapshot(self, img: Image.Image) -> None:
        from src.records import save_snapshot

        try:
            save_snapshot(img)
        except Exception:
            pass

    def _on_card(self, event: tk.Event) -> None:
        self.hint.config(wraplength=max(200, event.width - 40))

    def _on_stage(self, event: tk.Event) -> None:
        if event.widget is not self.stage:
            return
        self._view_w = max(1, event.width)
        self._view_h = max(1, event.height)
        if self._frame_pil is not None:
            self._show_image(self._frame_pil)
        else:
            self._sync_shutter()

    def _shutter_image(self) -> ImageTk.PhotoImage:
        if not self._shutter_enabled:
            return self._icon_off
        if self._mode == "frozen":
            return self._icon_hold
        return self._icon_snap

    def _sync_shutter(self) -> None:
        x = max(48, self._view_w - 54)
        y = max(48, self._view_h // 2)
        icon = self._shutter_image()
        item = self.view.find_withtag("shutter")
        if item:
            self.view.coords("shutter", x, y)
            self.view.itemconfig("shutter", image=icon)
        else:
            self.view.create_image(x, y, image=icon, tags="shutter")
            self.view.tag_bind("shutter", "<Button-1>", self._on_shutter)
            self.view.tag_bind("shutter", "<Enter>", lambda _e: self.view.config(cursor="hand2"))
            self.view.tag_bind("shutter", "<Leave>", lambda _e: self.view.config(cursor="arrow"))
        self.view.tag_raise("shutter")

    def _on_shutter(self, _event=None) -> None:
        if self._shutter_enabled:
            self.snap()

    def _on_view_click(self, event: tk.Event) -> None:
        sx = max(48, self._view_w - 54)
        sy = max(48, self._view_h // 2)
        if (event.x - sx) ** 2 + (event.y - sy) ** 2 <= 46 ** 2:
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

    def _set_chips(self, items: list[tuple[str, str, str]]) -> None:
        for child in self.chip_row.winfo_children():
            child.destroy()
        if not items:
            items = [("No detection", CHIP, MUTED)]
        for text, bg, fg in items:
            tk.Label(
                self.chip_row,
                text=text,
                bg=bg,
                fg=fg,
                font=(self._face, 13, "bold"),
                padx=14,
                pady=7,
            ).pack(side="left", padx=(0, 8))

    def _paint_chrome(self) -> None:
        if self._mode == "frozen":
            self.badge.config(text="HOLD", bg=FREEZE_BG, fg=TEXT)
        else:
            self.badge.config(text="LIVE", bg=LIVE_BG, fg=TEXT)
        self._sync_shutter()

    def _start_camera(self) -> None:
        if self.cap is not None:
            return
        cap = open_capture(self._camera_pref)
        if cap is None:
            self._shutter_enabled = False
            self.hint.config(text="No camera.", fg=WARN)
            self._set_chips([("No camera", CHIP, WARN)])
            self._sync_shutter()
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
        self._mode = "live"
        self._shutter_enabled = True
        if self.scanner is None:
            self.hint.config(text="Loading inspector…", fg=MUTED)
        else:
            self.hint.config(text="Point the camera at a plant.", fg=MUTED)
        self._paint_chrome()

    def _resume_live(self) -> None:
        if self.cap is None:
            self._start_camera()
            if self.cap is None:
                return
        self._scan_gen += 1
        self._tracks = []
        self._picked_tid = None
        self._mode = "live"
        self._clear_findings()
        self.hint.config(text="Point the camera at a plant.", fg=MUTED)
        self._paint_chrome()

    def _tick(self) -> None:
        if self._page != "scan":
            self._tick_id = self.root.after(250, self._tick)
            return
        if self._mode == "live" and self.cap is not None:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                try:
                    self._frame_pil = _frame_to_pil(frame)
                    self._show_image(self._frame_pil)
                    self._kick_detect()
                    self._kick_grade()
                except Exception:
                    pass
        elif self._mode == "frozen":
            self._kick_grade()
        self._tick_id = self.root.after(33, self._tick)

    def _kick_detect(self) -> None:
        if self._detect_busy or self._frame_pil is None or self.finder is None:
            return
        if self._mode != "live":
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
        if gen != self._scan_gen or self._mode != "live":
            return
        self._tracks = match_tracks(self._tracks, boxes)
        if self._picked_tid is not None and not any(t.get("tid") == self._picked_tid for t in self._tracks):
            self._picked_tid = None
        n = len(self._tracks)
        if n == 0:
            self._clear_findings()
            self.hint.config(text="No plant boxed. Fill the screen with a plant.", fg=WARN)
        elif not any(t.get("stage") == "graded" for t in self._tracks):
            self._set_chips([("Plant", CHIP, TEXT), (f"{n} boxed", CHIP, TEXT)])
            self.hint.config(text="Naming the crop…", fg=MUTED)
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
        focus = self._focus_track()
        selected = focus.get("tid") if focus else None
        framed = draw_boxes(img, self._tracks, selected_tid=selected) if self._tracks else img
        show = framed.copy()
        max_w = max(320, self._view_w)
        max_h = max(180, self._view_h)
        show.thumbnail((max_w, max_h))
        self.photo = ImageTk.PhotoImage(show)
        sw, sh = show.size
        ow, oh = img.size
        scale = sw / max(1, ow)
        self._disp = {
            "ox": self._view_w / 2 - sw / 2,
            "oy": self._view_h / 2 - sh / 2,
            "scale": scale,
        }
        self.view.delete("frame")
        self.view.delete("empty")
        self.view.create_image(self._view_w // 2, self._view_h // 2, image=self.photo, tags="frame")
        self.view.tag_lower("frame")
        self._sync_shutter()

    def snap(self) -> None:
        if self._mode == "frozen":
            self._resume_live()
            return
        if self._frame_pil is None:
            self.hint.config(text="No camera frame yet.", fg=WARN)
            return
        img = self._frame_pil.copy()
        self._scan_gen += 1
        self._tracks = []
        self._picked_tid = None
        self._mode = "frozen"
        self._save_snapshot(img)
        self._clear_findings()
        self._paint_chrome()
        self._set_chips([("Hold", FREEZE_BG, TEXT), ("Saved", CHIP, TEXT)])
        self.hint.config(text="Snapshot saved. Scanning this frame…", fg=MUTED)
        self._show_image(img)
        gen = self._scan_gen
        finder = self.finder

        def work() -> None:
            try:
                boxes = finder.find(img) if finder is not None else []
            except Exception:
                boxes = []
            self.root.after(0, lambda b=boxes, g=gen, i=img: self._after_snap_boxes(g, i, b))

        threading.Thread(target=work, daemon=True).start()

    def _after_snap_boxes(self, gen: int, img: Image.Image, boxes: list[dict]) -> None:
        if gen != self._scan_gen or self._mode != "frozen":
            return
        self._frame_pil = img
        self._tracks = match_tracks([], boxes)
        self._picked_tid = None
        self._show_image(img)
        if not self._tracks:
            self._set_chips([("No plant", CHIP, WARN)])
            self.hint.config(text="No plant boxed. Tap the button to retake.", fg=WARN)
        else:
            n = len(self._tracks)
            self._set_chips([("Plant", CHIP, TEXT), (f"{n} boxed", CHIP, TEXT)])
            self.hint.config(text="Naming the crop…", fg=MUTED)

    def _clear_findings(self) -> None:
        self._set_chips([])
        self.hint.config(text="")

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
        marker: list[tuple[str, str, str]] = []
        if n > 1:
            hbg, hfg = HEALTH_CHIP.get(track.get("health") or "", (CHIP, TEXT))
            marker.append((f"Box {self._box_number(track)}", hbg, hfg))
        crop = track.get("crop")
        health = track.get("health")
        named = track.get("named_plant")
        tip = track.get("tip") or ""
        extra = "Tap another box to inspect it." if n > 1 else ""
        if track.get("stage") != "graded":
            self._set_chips(marker + [("Naming…", CHIP, TEXT)])
            self.hint.config(text="Naming this box." + ("  " + extra if extra else ""), fg=MUTED)
            return
        if crop and crop != "unknown":
            chips = marker + [(crop, CHIP, TEXT)]
            if health and health != "unknown":
                hbg, hfg = HEALTH_CHIP.get(health, (CHIP, TEXT))
                chips.append((health, hbg, hfg))
            self._set_chips(chips)
            line = tip if tip else extra
            if tip and extra:
                line = f"{tip}  ·  {extra}"
            self.hint.config(text=line, fg=MUTED)
            return
        if named:
            short = str(named).split(" (")[0]
            self._set_chips(marker + [(short, CHIP, TEXT), ("Not farm crop", CHIP, WARN)])
            self.hint.config(text=(tip or "Cannot name this crop yet.") + ("  ·  " + extra if extra else ""), fg=WARN)
            return
        self._set_chips(marker + [("Unknown crop", CHIP, WARN)])
        self.hint.config(text=(tip or "Cannot name this crop yet.") + ("  ·  " + extra if extra else ""), fg=WARN)

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
