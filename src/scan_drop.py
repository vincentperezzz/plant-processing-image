import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from src.detect import PlantFinder, _iou, crop_xyxy, draw_boxes, match_tracks
from src.infer import Scanner
from src.paths import CKPT

BG = "#142016"
PANEL = "#1e2e22"
ACCENT = "#8bc34a"
TEXT = "#f1f8e9"
MUTED = "#a5b89a"
WARN = "#ef9a9a"


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    import cv2

    buf = np.ascontiguousarray(frame)
    if buf.dtype != np.uint8:
        buf = cv2.convertScaleAbs(buf)
    if buf.ndim == 2:
        rgb = cv2.cvtColor(buf, cv2.COLOR_GRAY2RGB)
    elif buf.shape[2] == 2:
        rgb = cv2.cvtColor(buf, cv2.COLOR_YUV2RGB_YUY2)
    elif buf.shape[2] == 4:
        rgb = cv2.cvtColor(buf, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.ascontiguousarray(rgb).copy(), "RGB")


def _open_capture():
    from src.camera import open_capture

    return open_capture("auto")


def _track_label(track: dict) -> str:
    crop = track.get("crop")
    health = track.get("health")
    named = track.get("named_plant")
    if crop and health and health != "unknown":
        return f"{crop}  {health}"
    if crop:
        return str(crop)
    if named:
        return str(named).split(" (")[0]
    return "plant"


def _box_area(track: dict) -> int:
    x1, y1, x2, y2 = track["xyxy"]
    return max(0, x2 - x1) * max(0, y2 - y1)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Plant Health Scanner")
        self.root.geometry("800x820")
        self.root.configure(bg=BG)
        self.scanner = None
        self.finder = PlantFinder()
        self.photo = None
        self.cap = None
        self._frame_pil = None
        self._mode = "idle"
        self._tick_id = None
        self._scan_gen = 0
        self._tracks: list[dict] = []
        self._detect_busy = False
        self._grade_busy = False
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._load_model()
        self._start_camera()

    def _build(self) -> None:
        tk.Label(
            self.root,
            text="Plant Health Scanner",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 18, "bold"),
        ).pack(pady=(18, 6))
        self.drop = tk.Label(
            self.root,
            text="Starting camera…",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 12),
            width=64,
            height=14,
            relief="ridge",
            bd=2,
        )
        self.drop.pack(padx=24, pady=16, fill="both", expand=True)
        modes = tk.Frame(self.root, bg=BG)
        modes.pack(pady=(0, 4))
        self.browse_btn = tk.Button(
            modes,
            text="Browse",
            command=self.browse,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=16,
            pady=6,
        )
        self.browse_btn.pack(side="left", padx=6)
        self.snap_btn = tk.Button(
            self.root,
            text="Snap",
            command=self.snap,
            bg=ACCENT,
            fg="#102010",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            padx=28,
            pady=8,
        )
        self.snap_btn.pack(pady=(2, 8))
        self.status = tk.Label(self.root, text="", bg=BG, fg=ACCENT, font=("Segoe UI", 11, "bold"), justify="left")
        self.status.pack(padx=28, pady=(4, 2), anchor="w")
        self.scores = tk.Label(
            self.root,
            text="",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9),
            justify="left",
            wraplength=740,
        )
        self.scores.pack(padx=28, pady=(0, 4), anchor="w")
        self.tip = tk.Label(
            self.root,
            text="",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 10),
            wraplength=720,
            justify="left",
        )
        self.tip.pack(padx=28, pady=(0, 18), anchor="w")
        self._paint_mode_buttons()

    def _paint_mode_buttons(self) -> None:
        def paint(btn: tk.Button, on: bool) -> None:
            if on:
                btn.config(bg=ACCENT, fg="#102010")
            else:
                btn.config(bg=PANEL, fg=TEXT)

        paint(self.browse_btn, self._mode == "file")
        if self._mode == "frozen":
            self.snap_btn.config(text="Retake", state="normal")
        elif self._mode == "live":
            self.snap_btn.config(text="Snap", state="normal")
        else:
            self.snap_btn.config(text="Snap", state="disabled")

    def _load_model(self) -> None:
        try:
            self.scanner = Scanner(CKPT)
            self.status.config(text="Ready.")
        except FileNotFoundError:
            self.status.config(text="No trained model yet. Run python training/train.py")

    def _start_camera(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.drop.config(text="Install opencv-python for camera\nor click Browse")
            self.status.config(text="No OpenCV. Browse or drop a photo.", fg=WARN)
            return
        cap = _open_capture()
        if cap is None:
            self.drop.config(text="No camera found\nClick Browse or drop a photo")
            self.status.config(text="No camera. Browse or drop a photo.", fg=WARN)
            return
        self.cap = cap
        self._mode = "live"
        self.status.config(text="Looking for plants… red box first, details later.", fg=ACCENT)
        self._paint_mode_buttons()
        self._tick()

    def _tick(self) -> None:
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
        elif self._mode in ("frozen", "file"):
            self._kick_grade()
        self._tick_id = self.root.after(33, self._tick)

    def _kick_detect(self) -> None:
        if self._detect_busy or self._frame_pil is None:
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
        n = len(self._tracks)
        if n == 0:
            self._clear_findings()
            self.status.config(text="No plant boxed yet. Fill the frame with leaves or a plant.", fg=WARN)
        else:
            self.status.config(text=f"{n} plant box{'es' if n != 1 else ''}  ·  naming next", fg=ACCENT)

    def _kick_grade(self) -> None:
        if self._grade_busy or self.scanner is None or self._frame_pil is None:
            return
        found = [t for t in self._tracks if t.get("stage") == "found"]
        named = [t for t in self._tracks if t.get("stage") == "named"]
        if found:
            track = max(found, key=_box_area)
            detail = False
        elif named:
            track = max(named, key=_box_area)
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
        self._grade_busy = False
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
        if not result["unknown"]:
            target["crop"] = result["crop"]
            target["health"] = result["health"]
            target["stage"] = "graded"
            target["tip"] = result.get("tip") or ""
            target["label"] = _track_label(target)
            self._paint_result(result)
        elif detail:
            target["named_plant"] = result.get("named_plant")
            target["crop"] = None if result["crop"] == "unknown" else result["crop"]
            target["health"] = None if result["health"] == "unknown" else result["health"]
            target["stage"] = "graded"
            target["tip"] = result.get("tip") or ""
            target["label"] = _track_label(target)
            self._paint_result(result)
        else:
            target["stage"] = "named"
            target["label"] = "plant"
        if self._mode in ("frozen", "file") and self._frame_pil is not None:
            self._show_image(self._frame_pil)

    def _show_image(self, img: Image.Image) -> None:
        framed = draw_boxes(img, self._tracks) if self._tracks else img
        show = framed.copy()
        show.thumbnail((640, 360))
        self.photo = ImageTk.PhotoImage(show)
        self.drop.config(image=self.photo, text="")

    def go_live(self) -> None:
        if self.cap is None:
            self._start_camera()
            if self.cap is None:
                return
        self._scan_gen += 1
        self._tracks = []
        self._mode = "live"
        self.status.config(text="Looking for plants… red box first, details later.", fg=ACCENT)
        self.scores.config(text="")
        self.tip.config(text="")
        self._paint_mode_buttons()

    def snap(self) -> None:
        if self._mode == "frozen":
            self.go_live()
            return
        if self._frame_pil is None:
            self.status.config(text="No camera frame yet.", fg=WARN)
            return
        img = self._frame_pil.copy()
        self._scan_gen += 1
        self._mode = "frozen"
        self._paint_mode_buttons()
        self.status.config(text="Frozen. Boxing plants, then grading…", fg=ACCENT)
        self._show_image(img)
        gen = self._scan_gen

        def work() -> None:
            try:
                boxes = self.finder.find(img)
            except Exception:
                boxes = []
            self.root.after(0, lambda b=boxes, g=gen, i=img: self._after_snap_boxes(g, i, b))

        threading.Thread(target=work, daemon=True).start()

    def _after_snap_boxes(self, gen: int, img: Image.Image, boxes: list[dict]) -> None:
        if gen != self._scan_gen or self._mode != "frozen":
            return
        self._frame_pil = img
        self._tracks = match_tracks([], boxes)
        self._show_image(img)
        if not self._tracks:
            self.status.config(text="No plant boxed. Retake and fill the frame.", fg=WARN)
            self._clear_findings()

    def browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Plant photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")],
        )
        if path:
            self.scan_path(path)

    def scan_path(self, path: str) -> None:
        if self.scanner is None:
            messagebox.showerror("Scanner", "Train the model first: python training/train.py")
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Scan failed", str(exc))
            return
        self._scan_gen += 1
        self._mode = "file"
        self._frame_pil = img
        self._tracks = []
        self._show_image(img)
        self.status.config(text="Boxing plants…", fg=ACCENT)
        self._paint_mode_buttons()
        gen = self._scan_gen

        def work() -> None:
            try:
                boxes = self.finder.find(img)
            except Exception:
                boxes = []
            self.root.after(0, lambda b=boxes, g=gen: self._after_file_boxes(g, b))

        threading.Thread(target=work, daemon=True).start()

    def _after_file_boxes(self, gen: int, boxes: list[dict]) -> None:
        if gen != self._scan_gen or self._mode != "file":
            return
        self._tracks = match_tracks([], boxes)
        if self._frame_pil is not None:
            self._show_image(self._frame_pil)
        if not self._tracks:
            self.status.config(text="No plant boxed in this photo.", fg=WARN)
            self._clear_findings()

    def _clear_findings(self) -> None:
        self.scores.config(text="")
        self.tip.config(text="")

    def _paint_result(self, result: dict) -> None:
        ranked = sorted(result["crop_scores"].items(), key=lambda kv: kv[1], reverse=True)
        bars = "   ".join(
            f"{'not in list' if name == 'other' else name} {pct:.0%}" for name, pct in ranked
        )
        bars = f"in-list {result.get('in_list', 1.0):.0%}   {bars}"
        if result["unknown"]:
            named = result.get("named_plant")
            if result["reason"] == "not_a_leaf":
                line = "Cannot detect a plant"
            elif named:
                line = (
                    f"Looks like {named}\n"
                    f"Not a farm crop I grade   (in-list {result.get('in_list', 0):.0%})"
                )
            elif result["reason"] == "not_in_list":
                line = (
                    f"Not one of my crops   (in-list {result.get('in_list', 0):.0%}, "
                    f"{result['other_probability']:.0%} something else)\n"
                    f"Closest known crop: {result['guess']}   ({result['crop_confidence']:.0%})"
                )
            else:
                line = (
                    f"Cannot detect crop\n"
                    f"Closest guess: {result['guess']}   ({result['crop_confidence']:.0%})"
                )
            self.status.config(text=line, fg=WARN)
        else:
            view = result.get("view") or "leaf"
            line = (
                f"Crop: {result['crop']}   ({result['crop_confidence']:.0%})   [{view}]\n"
                f"Health: {result['health']}   ({result['health_confidence']:.0%})"
            )
            self.status.config(text=line, fg=ACCENT)
        dict_bits = ""
        guesses = result.get("dictionary_guesses") or []
        if guesses:
            dict_bits = "   Dictionary  " + "   ".join(
                f"{g.get('local') or g['name']} {g['score']:.0%}" for g in guesses
            )
        self.scores.config(text=f"Confidence  {bars}{dict_bits}")
        self.tip.config(text=result.get("tip") or "")

    def _close(self) -> None:
        self._scan_gen += 1
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        self._mode = "idle"
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.root.destroy()


def _enable_drop(app: App) -> None:
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
    except ImportError:
        return
    if not isinstance(app.root, TkinterDnD.Tk):
        return

    def on_drop(event) -> None:
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        path = raw.split("} {")[0].split()[0] if raw else ""
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        if path:
            app.scan_path(path)

    app.drop.drop_target_register(DND_FILES)
    app.drop.dnd_bind("<<Drop>>", on_drop)


def main() -> None:
    try:
        from tkinterdnd2 import TkinterDnD

        root = TkinterDnD.Tk()
        dnd = True
    except ImportError:
        root = tk.Tk()
        dnd = False
    app = App(root)
    if dnd:
        _enable_drop(app)
    root.mainloop()


if __name__ == "__main__":
    main()
