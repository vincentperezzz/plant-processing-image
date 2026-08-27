from __future__ import annotations

import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.infer import looks_like_nonplant, plant_look
from src.paths import MODELS

RED = (229, 57, 53)
DEAD = (127, 0, 0)
GREEN = (76, 175, 80)
AMBER = (255, 179, 0)
GRAY = (176, 190, 197)
WHITE = (255, 255, 255)
INK = (16, 32, 16)
MAX_BOXES = 5
YOLO_KEEP = ("plant", "leaf")
YOLO_DROP = (
    "person",
    "human",
    "curtain",
    "drape",
    "t-shirt",
    "shirt",
    "clothing",
    "fabric",
    "wall",
    "furniture",
    "window",
)
YOLO_CLASSES = list(YOLO_KEEP) + list(YOLO_DROP)

HEALTH_RGB = {
    "healthy": GREEN,
    "mild": AMBER,
    "critical": RED,
    "dead": DEAD,
}


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (aa + bb - inter + 1e-6)


def _area_xyxy(xyxy: tuple) -> int:
    x1, y1, x2, y2 = xyxy
    return max(0, int(x2 - x1)) * max(0, int(y2 - y1))


def _center_xyxy(xyxy: tuple) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _union_xyxy(a: tuple, b: tuple) -> tuple[int, int, int, int]:
    return (int(min(a[0], b[0])), int(min(a[1], b[1])), int(max(a[2], b[2])), int(max(a[3], b[3])))


def _containment(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    small = max(1, min(_area_xyxy(a), _area_xyxy(b)))
    return inter / small


def _should_merge(a: tuple, b: tuple) -> bool:
    if _containment(a, b) >= 0.55:
        return True
    if _iou(a, b) >= 0.32:
        return True
    aa, bb = _area_xyxy(a), _area_xyxy(b)
    if min(aa, bb) < 1:
        return False
    larger, smaller = (a, b) if aa >= bb else (b, a)
    cx, cy = _center_xyxy(smaller)
    padx = 0.14 * max(1, larger[2] - larger[0])
    pady = 0.14 * max(1, larger[3] - larger[1])
    if (larger[0] - padx) <= cx <= (larger[2] + padx) and (larger[1] - pady) <= cy <= (larger[3] + pady):
        return True
    dx = cx - _center_xyxy(larger)[0]
    dy = cy - _center_xyxy(larger)[1]
    dist = (dx * dx + dy * dy) ** 0.5
    reach = 0.38 * (((larger[2] - larger[0]) ** 2 + (larger[3] - larger[1]) ** 2) ** 0.5)
    return dist < reach and _containment(a, b) >= 0.22


def _canopy_mask(img: Image.Image) -> tuple:
    import cv2

    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    scale = max(h, w) / 420.0
    if scale > 1.05:
        small = cv2.resize(arr, (max(1, int(w / scale)), max(1, int(h / scale))), interpolation=cv2.INTER_AREA)
    else:
        small = arr
        scale = 1.0
    rgb = small.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 1e-6)
    green = (g > r * 1.02) & (g > b * 1.02) & (g > 0.10) & (sat > 0.05)
    yellow = (r > 0.16) & (g > 0.16) & (g > b + 0.012) & (r > b + 0.012) & (sat > 0.05)
    brown = (r > 0.12) & (g > 0.07) & (r >= g * 0.82) & (r > b + 0.03) & (sat > 0.07) & (b < 0.50)
    fruit = ((r > 0.40) & (r > g + 0.06) & (sat > 0.18)) | ((r > 0.42) & (g > 0.22) & (b < g * 0.75) & (sat > 0.22))
    mask = (green | yellow | brown | fruit).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    count, labels = cv2.connectedComponents(mask)
    return labels, scale, count


def _box_component(labels: np.ndarray, scale: float, xyxy: tuple) -> int:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(0, int(x1 / scale))
    y1 = max(0, int(y1 / scale))
    x2 = min(labels.shape[1], max(x1 + 1, int(x2 / scale)))
    y2 = min(labels.shape[0], max(y1 + 1, int(y2 / scale)))
    patch = labels[y1:y2, x1:x2]
    if patch.size == 0:
        return 0
    vals = patch[patch > 0]
    if vals.size == 0:
        return 0
    return int(np.bincount(vals.ravel()).argmax())


def _component_bbox(labels: np.ndarray, cid: int, scale: float, iw: int, ih: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(labels == cid)
    if xs.size == 0:
        return None
    x1 = max(0, int(xs.min() * scale) - 4)
    y1 = max(0, int(ys.min() * scale) - 4)
    x2 = min(iw, int((xs.max() + 1) * scale) + 4)
    y2 = min(ih, int((ys.max() + 1) * scale) + 4)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return (x1, y1, x2, y2)


def _merge_same_plant(img: Image.Image, boxes: list[dict]) -> list[dict]:
    if not boxes:
        return []
    if len(boxes) == 1:
        return boxes[:MAX_BOXES]
    try:
        labels, scale, count = _canopy_mask(img)
    except Exception:
        return _merge_close_boxes(boxes)
    iw, ih = img.size
    groups: dict[int, list[dict]] = {}
    leftovers: list[dict] = []
    for box in boxes:
        cid = _box_component(labels, scale, box["xyxy"])
        if cid <= 0:
            leftovers.append(dict(box))
            continue
        groups.setdefault(cid, []).append(dict(box))
    out: list[dict] = []
    img_a = max(1, iw * ih)
    for cid, group in groups.items():
        acc = dict(group[0])
        for other in group[1:]:
            acc["xyxy"] = _union_xyxy(acc["xyxy"], other["xyxy"])
            acc["score"] = max(float(acc.get("score") or 0), float(other.get("score") or 0))
        acc["name"] = "plant"
        canopy = _component_bbox(labels, cid, scale, iw, ih)
        if canopy is not None:
            union_a = _area_xyxy(acc["xyxy"])
            canopy_a = _area_xyxy(canopy)
            if canopy_a < 0.62 * img_a and canopy_a <= max(union_a * 3.2, union_a + 1):
                acc["xyxy"] = _union_xyxy(acc["xyxy"], canopy)
        out.append(acc)
    out.extend(leftovers)
    return _merge_close_boxes(out)


def _merge_close_boxes(boxes: list[dict]) -> list[dict]:
    if len(boxes) < 2:
        return boxes[:MAX_BOXES]
    items = [dict(box) for box in boxes]
    changed = True
    while changed:
        changed = False
        items.sort(key=lambda b: _area_xyxy(b["xyxy"]), reverse=True)
        out: list[dict] = []
        used: set[int] = set()
        for i, box in enumerate(items):
            if i in used:
                continue
            acc = dict(box)
            for j in range(i + 1, len(items)):
                if j in used:
                    continue
                other = items[j]
                if _should_merge(acc["xyxy"], other["xyxy"]):
                    acc["xyxy"] = _union_xyxy(acc["xyxy"], other["xyxy"])
                    acc["score"] = max(float(acc.get("score") or 0), float(other.get("score") or 0))
                    used.add(j)
                    changed = True
            out.append(acc)
        items = out
    return items[:MAX_BOXES]


def _nms(boxes: list[dict], thr: float = 0.45, cap: int | None = None) -> list[dict]:
    boxes = sorted(boxes, key=lambda b: b.get("score", 0), reverse=True)
    keep: list[dict] = []
    limit = MAX_BOXES if cap is None else cap
    for box in boxes:
        if all(_iou(box["xyxy"], k["xyxy"]) < thr for k in keep):
            keep.append(box)
        if len(keep) >= limit:
            break
    return keep


def find_plants_exg(img: Image.Image) -> list[dict]:
    import cv2

    arr = np.asarray(img.convert("RGB"))
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    exg = 2 * g - r - b
    h, w = exg.shape
    thr = max(20, int(np.percentile(exg, 68)))
    mask = ((exg > thr) & (g > 36)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.018 * h * w
    max_area = 0.90 * h * w
    boxes: list[dict] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        if bh < 24 or bw < 24:
            continue
        score = min(0.92, 0.35 + area / (h * w))
        boxes.append({"xyxy": (x, y, x + bw, y + bh), "score": score, "name": "plant"})
    return _nms(boxes)


def _plant_pixel_score(crop: Image.Image) -> float:
    arr = np.asarray(crop.convert("RGB").resize((96, 96)), dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 1e-6)
    yellow = (r > 0.16) & (g > 0.16) & (g > b + 0.012) & (r > b + 0.012) & (sat > 0.05)
    brown = (r > 0.12) & (g > 0.07) & (r >= g * 0.82) & (r > b + 0.03) & (sat > 0.07) & (b < 0.50)
    green = (g > r * 1.03) & (g > b * 1.03) & (g > 0.10) & (sat > 0.05)
    return float((yellow | brown | green).mean())


def _shape_plant_score(crop: Image.Image) -> tuple[float, float]:
    import cv2

    arr = np.asarray(crop.convert("RGB").resize((128, 128)))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ax = float(np.mean(np.abs(gx)))
    ay = float(np.mean(np.abs(gy)))
    stripe = ay / (ax + 1e-6)
    edges = cv2.Canny(gray, 40, 130)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    jagged = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        peri = float(cv2.arcLength(contour, True))
        if area > 24:
            jagged = peri * peri / (12.566 * area)
    shape = 0.0
    if stripe < 1.9:
        shape += 0.45
    if jagged > 1.3:
        shape += 0.45
    if 0.65 < stripe < 1.55:
        shape += 0.2
    return min(1.0, shape), stripe


def _looks_like_ui(crop: Image.Image) -> bool:
    import cv2

    look = plant_look(crop)
    if look["fruit"] >= 0.02 or look["vegetation"] >= 0.16:
        return False
    arr = np.asarray(crop.convert("RGB").resize((128, 96)))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    horiz = float(cv2.morphologyEx(edges, cv2.MORPH_OPEN, hk).mean())
    vert = float(cv2.morphologyEx(edges, cv2.MORPH_OPEN, vk).mean())
    return horiz > 4.5 and horiz > vert * 1.55 and look["vegetation"] < 0.12 and look["fruit"] < 0.012


def _box_looks_like_plant(img: Image.Image, xyxy: tuple, score: float = 0.0) -> bool:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    iw, _ih = img.size
    aspect = bh / bw
    if aspect > 2.3 and bw / iw < 0.38:
        return False
    if aspect > 2.8:
        return False
    crop = crop_xyxy(img, xyxy, pad=0.02)
    if _looks_like_ui(crop):
        return False
    if looks_like_nonplant(crop):
        return False
    look = plant_look(crop)
    plantish = max(look["vegetation"], look["fruit"], _plant_pixel_score(crop))
    shape, stripe = _shape_plant_score(crop)
    if stripe > 1.85 and aspect > 1.8:
        return False
    if stripe > 2.2:
        return False
    if look["skin"] > 0.25 and plantish < 0.08 and shape < 0.4:
        return False
    if score >= 0.25 and plantish >= 0.03 and shape >= 0.45 and stripe < 1.8:
        return True
    if plantish >= 0.08 and stripe < 1.8 and aspect < 2.2:
        return True
    if shape >= 0.7 and stripe < 1.7 and aspect < 2.2:
        return True
    return False


def _keep_plant_boxes(img: Image.Image, boxes: list[dict]) -> list[dict]:
    kept = [b for b in boxes if _box_looks_like_plant(img, b["xyxy"], float(b.get("score") or 0))]
    return _merge_same_plant(img, kept)


def crop_xyxy(img: Image.Image, xyxy: tuple, pad: float = 0.08) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    pw = int((x2 - x1) * pad)
    ph = int((y2 - y1) * pad)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return img
    return img.crop((x1, y1, x2, y2))


def match_tracks(old: list[dict], boxes: list[dict]) -> list[dict]:
    used: set[int] = set()
    out: list[dict] = []
    nxt = max((int(t.get("tid") or 0) for t in old), default=0)
    for box in boxes:
        best, idx = 0.0, None
        for i, track in enumerate(old):
            if i in used:
                continue
            score = _iou(track["xyxy"], box["xyxy"])
            if score > best:
                best, idx = score, i
        if idx is not None and best >= 0.35:
            track = dict(old[idx])
            if not track.get("tid"):
                nxt += 1
                track["tid"] = nxt
            track["xyxy"] = box["xyxy"]
            track["score"] = box.get("score", track.get("score", 0))
            used.add(idx)
            out.append(track)
        else:
            nxt += 1
            out.append(
                {
                    "tid": nxt,
                    "xyxy": box["xyxy"],
                    "score": box.get("score", 0.5),
                    "name": "plant",
                    "label": "plant",
                    "crop": None,
                    "health": None,
                    "named_plant": None,
                    "stage": "found",
                    "tip": "",
                }
            )
    return out[:MAX_BOXES]


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _health_rgb(track: dict) -> tuple[int, int, int]:
    health = track.get("health")
    if health in HEALTH_RGB:
        return HEALTH_RGB[health]
    return GRAY


def _dash_line(draw: ImageDraw.ImageDraw, p0: tuple, p1: tuple, fill: tuple, width: int, on: int = 14, off: int = 8) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    t = 0.0
    paint = True
    while t < length:
        step = on if paint else off
        t2 = min(length, t + step)
        if paint:
            draw.line(
                (x0 + ux * t, y0 + uy * t, x0 + ux * t2, y0 + uy * t2),
                fill=fill,
                width=width,
            )
        t = t2
        paint = not paint


def _outline_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple, width: int, dashed: bool) -> None:
    x1, y1, x2, y2 = box
    if not dashed:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        return
    _dash_line(draw, (x1, y1), (x2, y1), color, width)
    _dash_line(draw, (x2, y1), (x2, y2), color, width)
    _dash_line(draw, (x2, y2), (x1, y2), color, width)
    _dash_line(draw, (x1, y2), (x1, y1), color, width)
    r = max(1, width)
    for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        draw.rectangle([px - r, py - r, px + r, py + r], fill=color)


def draw_boxes(
    img: Image.Image,
    tracks: list[dict],
    selected_tid: int | None = None,
    *,
    captions: bool = True,
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _font(16)
    ranked = sorted(tracks, key=lambda t: _area_xyxy(t["xyxy"]), reverse=True)
    numbers = {t.get("tid"): i + 1 for i, t in enumerate(ranked)}
    focus = selected_tid
    if focus is None and tracks:
        focus = max(tracks, key=lambda t: _area_xyxy(t["xyxy"])).get("tid")
    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track["xyxy"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out.size[0] - 1, x2), min(out.size[1] - 1, y2)
        tid = track.get("tid")
        picked = tid == focus
        color = _health_rgb(track)
        _outline_rect(draw, (x1, y1, x2, y2), color, 5 if picked else 2, dashed=not picked)
        if not captions:
            continue
        n = numbers.get(tid, 1)
        label = str(track.get("label") or "plant")
        tag = f"{n}  {label}" if len(tracks) > 1 else label
        box = draw.textbbox((0, 0), tag, font=font)
        tw, th = box[2] - box[0] + 10, box[3] - box[1] + 6
        ty = y1 - th if y1 - th >= 2 else y1 + 2
        tx2 = min(out.size[0] - 1, x1 + tw)
        draw.rectangle([x1, ty, tx2, ty + th], fill=color)
        ink = INK if track.get("health") == "mild" else WHITE
        draw.text((x1 + 5, ty + 2), tag, fill=ink, font=font)
    return out


class PlantFinder:
    def __init__(self, backend: str = "auto") -> None:
        self._yolo = None
        self._lock = threading.Lock()
        self.backend = "color"
        if backend == "color":
            return
        threading.Thread(target=self._load_yolo, daemon=True).start()

    def _load_yolo(self) -> None:
        try:
            from ultralytics import YOLO

            MODELS.mkdir(parents=True, exist_ok=True)
            weights = MODELS / "yolov8s-worldv2.pt"
            model = YOLO(str(weights) if weights.exists() else "yolov8s-worldv2.pt")
            model.set_classes(YOLO_CLASSES)
            with self._lock:
                self._yolo = model
                self.backend = "yolo"
        except Exception:
            with self._lock:
                self._yolo = None
                self.backend = "color"

    def find(self, img: Image.Image) -> list[dict]:
        with self._lock:
            yolo = self._yolo
        if yolo is not None:
            try:
                raw = self._find_yolo(yolo, img)
                plants = [b for b in raw if b.get("name") in YOLO_KEEP]
                junk = [b for b in raw if b.get("name") not in YOLO_KEEP]
                plants = [
                    b
                    for b in plants
                    if not any(_iou(b["xyxy"], j["xyxy"]) >= 0.35 for j in junk)
                ]
                boxed = _keep_plant_boxes(img, plants)
                if boxed:
                    return boxed
                if raw:
                    return []
            except Exception:
                pass
            extra = _keep_plant_boxes(img, find_plants_exg(img))
            return extra
        return _keep_plant_boxes(img, find_plants_exg(img))

    def _find_yolo(self, model, img: Image.Image) -> list[dict]:
        arr = np.ascontiguousarray(np.asarray(img.convert("RGB")))
        results = model.predict(
            arr,
            imgsz=320,
            conf=0.22,
            iou=0.45,
            max_det=20,
            verbose=False,
        )
        if not results:
            return []
        hit = results[0]
        if hit.boxes is None or len(hit.boxes) == 0:
            return []
        xyxy = hit.boxes.xyxy.cpu().numpy()
        conf = hit.boxes.conf.cpu().numpy()
        cls = hit.boxes.cls.cpu().numpy() if hit.boxes.cls is not None else None
        names = YOLO_CLASSES
        boxes = []
        for i, (row, score) in enumerate(zip(xyxy, conf)):
            x1, y1, x2, y2 = [int(v) for v in row]
            if (x2 - x1) * (y2 - y1) < 24 * 24:
                continue
            kind = "plant"
            if cls is not None:
                ci = int(cls[i])
                if 0 <= ci < len(names):
                    kind = names[ci]
            boxes.append({"xyxy": (x1, y1, x2, y2), "score": float(score), "name": kind})
        return _nms(boxes, cap=12)
