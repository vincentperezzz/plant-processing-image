import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.dictionary import PlantDictionary
from src.labels import crops, health_levels, load_spec
from src.model import TwoHeadNet
from src.paths import CKPT, META
from src.wording import word_tip

_PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


FARM_CROPS = ("palay", "sili", "tomato", "eggplant", "lettuce")


def _shape_masses(mask: np.ndarray) -> tuple[float, float]:
    try:
        import cv2
    except ImportError:
        frac = float(mask.mean()) if mask.any() else 0.0
        return frac, 0.0
    raw = np.ascontiguousarray(mask.astype(np.uint8) * 255)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    img_a = float(mask.shape[0] * mask.shape[1])
    round_a = 0.0
    skinny_a = 0.0
    for i in range(1, count):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 28:
            continue
        bw = float(max(1, stats[i, cv2.CC_STAT_WIDTH]))
        bh = float(max(1, stats[i, cv2.CC_STAT_HEIGHT]))
        aspect = max(bw, bh) / min(bw, bh)
        fill = area / (bw * bh)
        if aspect <= 1.55 and fill >= 0.42:
            round_a += area
        elif aspect <= 1.8 and fill >= 0.35:
            round_a += area * 0.65
        elif aspect >= 2.05:
            skinny_a += area
    return round_a / img_a, skinny_a / img_a


def plant_look(img: Image.Image) -> dict:
    arr = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 1e-6)
    green = (g > r * 1.06) & (g > b * 1.04) & (g > 0.16) & (sat > 0.12)
    yellow = (r > 0.22) & (g > 0.22) & (b < 0.48) & (g >= r * 0.72) & (g > b + 0.06) & (sat > 0.12)
    red = (r > 0.40) & (r > g + 0.08) & (r > b + 0.08) & (sat > 0.22)
    orange = (r > 0.46) & (g > 0.22) & (g < r * 0.92) & (b < g * 0.70) & (sat > 0.28) & ~red
    lime = (
        (r > 0.34)
        & (g > 0.36)
        & (b < 0.34)
        & (np.abs(r - g) < 0.18)
        & (g > b + 0.10)
        & (sat > 0.18)
        & ~green
    )
    purple = ((r > 0.14) & (b > 0.14) & (g < r * 0.82) & (g < b * 0.82) & (((r + b) * 0.5) > g + 0.04) & (sat > 0.12)) | (
        (mx < 0.38) & (r > g + 0.03) & (b > g + 0.02) & (sat > 0.14)
    )
    produce = red | orange | lime | purple
    leaf = float((green | yellow).mean())
    fruit_s = float(produce.mean())
    skin = float(((r > 0.32) & (g > 0.18) & (b > 0.12) & (r > g) & (r > b) & ((r - g) > 0.04) & (sat < 0.62)).mean())
    round_m, skinny_m = _shape_masses(produce)
    purple_round, purple_skinny = _shape_masses(purple)
    red_f = float(red.mean())
    orange_f = float(orange.mean())
    lime_f = float(lime.mean())
    purple_f = float(purple.mean())
    color_fruit = red_f + orange_f + lime_f
    hint = None
    if purple_f >= 0.018 or purple_round + purple_skinny >= 0.008:
        hint = "eggplant"
    elif round_m >= 0.0045 and round_m >= skinny_m * 0.8 and color_fruit >= 0.0035:
        hint = "tomato"
    elif skinny_m >= 0.0055 and skinny_m > round_m * 1.2 and color_fruit >= 0.0028:
        hint = "sili"
    elif round_m >= 0.01 and (orange_f + red_f + lime_f) >= 0.0025:
        hint = "tomato"
    return {
        "leaf": round(leaf, 4),
        "skin": round(skin, 4),
        "fruit": round(fruit_s, 4),
        "vegetation": round(leaf + fruit_s, 4),
        "fruit_hint": hint,
        "fruit_round": round(round_m, 4),
        "fruit_skinny": round(skinny_m, 4),
        "fruit_red": round(red_f, 4),
        "fruit_orange": round(orange_f, 4),
        "fruit_lime": round(lime_f, 4),
        "fruit_purple": round(purple_f, 4),
    }


def looks_like_nonplant(img: Image.Image) -> bool:
    look = plant_look(img)
    if look["fruit"] >= 0.02:
        return False
    if look["skin"] > 0.16 and look["leaf"] < 0.32:
        return True
    try:
        import cv2
    except ImportError:
        return False
    arr = np.asarray(img.convert("RGB").resize((128, 128)))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray_f = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_f, 40, 120)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
    horiz = float(cv2.morphologyEx(edges, cv2.MORPH_OPEN, hk).mean())
    vert = float(cv2.morphologyEx(edges, cv2.MORPH_OPEN, vk).mean())
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    jagged = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        peri = float(cv2.arcLength(contour, True))
        if area > 24:
            jagged = peri * peri / (12.566 * area)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    col = np.mean(np.abs(gx), axis=0)
    col = col - float(col.mean())
    denom = float(np.dot(col, col)) + 1e-6
    auto = 0.0
    if denom > 1.0:
        corr = np.correlate(col, col, mode="full")
        mid = corr.size // 2
        band = corr[mid + 6 : mid + 36]
        if band.size:
            auto = float(band.max()) / denom
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_std = float(hsv[:, :, 0].std())
    sat_std = float(hsv[:, :, 1].std())
    parallel = (vert > 3.0 and vert > horiz * 1.2) or (horiz > 3.0 and horiz > vert * 1.2)
    if jagged > 1.55 and look["vegetation"] > 0.18:
        return False
    if auto > 0.38 and parallel:
        return True
    if parallel and hue_std < 16 and sat_std < 38 and jagged < 1.35:
        return True
    if look["leaf"] >= 0.12 and jagged < 1.12 and hue_std < 13 and sat_std < 32 and auto < 0.22:
        return True
    return False


class Scanner:
    def __init__(self, ckpt_path: Path | None = None, *, use_dictionary: bool = True):
        path = ckpt_path or CKPT
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run training/finetune_other.py first.")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.crop_names = blob.get("crops") or crops()
        self.health_names = blob.get("health") or health_levels()
        spec = load_spec()
        self.crop_thr = float(spec.get("crop_unknown_threshold", 0.80))
        self.margin_thr = float(spec.get("crop_margin_threshold", 0.18))
        self.leaf_min = float(spec.get("leaf_min_score", 0.10))
        self.skin_max = float(spec.get("skin_max_score", 0.14))
        self.health_thr = float(spec.get("health_unknown_threshold", 0.45))
        self.other_i = self.crop_names.index("other") if "other" in self.crop_names else None
        self.other_thr = float(spec.get("other_threshold", 0.35))
        self.gate_thr = float(spec.get("in_list_threshold", 0.40))
        self.has_gate = "gate_head.weight" in blob["model"]
        self.dict_min = float(spec.get("dictionary_min_score", 0.24))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TwoHeadNet(len(self.crop_names), len(self.health_names))
        self.model.load_state_dict(blob["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()
        self._dictionary = None
        self.use_dictionary = use_dictionary

    @property
    def dictionary(self) -> PlantDictionary:
        if self._dictionary is None:
            self._dictionary = PlantDictionary(device=self.device)
        return self._dictionary

    def scan(
        self,
        image: str | Path | Image.Image,
        *,
        detail: bool = True,
        assume_plant: bool = False,
    ) -> dict:
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            img = Image.open(image).convert("RGB")
        look = plant_look(img)
        x = _PREPROCESS(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            crop_logits, health_logits, gate_logits = self.model(x)
            crop_p = torch.softmax(crop_logits, dim=1)[0]
            health_p = torch.softmax(health_logits, dim=1)[0]
            in_p = float(torch.sigmoid(gate_logits[0, 0])) if self.has_gate else 1.0
        health_i = int(health_p.argmax())
        other_p = float(crop_p[self.other_i]) if self.other_i is not None else 0.0
        real = crop_p.clone()
        if self.other_i is not None:
            real[self.other_i] = -1.0
        ordered = torch.topk(real, k=min(2, real.numel()))
        crop_i = int(ordered.indices[0])
        crop_conf = float(crop_p[crop_i])
        second = float(crop_p[int(ordered.indices[1])]) if ordered.values.numel() > 1 else 0.0
        margin = crop_conf - second
        health_conf = float(health_p[health_i])
        crop_scores = {self.crop_names[i]: round(float(crop_p[i]), 4) for i in range(len(self.crop_names))}
        health_scores = {self.health_names[i]: round(float(health_p[i]), 4) for i in range(len(self.health_names))}
        reason = None
        if looks_like_nonplant(img):
            reason = "not_a_leaf"
        elif (
            look["skin"] > self.skin_max
            and look["leaf"] < 0.22
            and look["fruit"] < 0.03
        ):
            reason = "not_a_leaf"
        elif self.has_gate and in_p < self.gate_thr:
            reason = "not_in_list"
        elif self.other_i is not None and (other_p >= crop_conf or other_p >= self.other_thr):
            reason = "not_in_list"
        elif crop_conf < self.crop_thr:
            reason = "low_confidence"
        elif margin < self.margin_thr:
            reason = "low_margin"
        unknown = reason is not None
        crop = None if unknown else self.crop_names[crop_i]
        if crop == "other":
            crop = None
            unknown = True
            reason = reason or "not_in_list"
        health = None if unknown or health_conf < self.health_thr else self.health_names[health_i]
        hint = look.get("fruit_hint")
        if hint in FARM_CROPS and hint in self.crop_names:
            strong = (look.get("fruit_round") or 0) >= 0.0045 or (look.get("fruit") or 0) >= 0.016
            if strong or (hint != crop and (look.get("fruit_round") or 0) >= 0.003):
                crop = hint
                crop_i = self.crop_names.index(hint)
                crop_conf = max(crop_conf, 0.72)
                margin = max(margin, 0.22)
                unknown = False
                reason = None
                health = self.health_names[health_i] if health_conf >= self.health_thr else health
        trust_cnn = (not unknown) and crop in FARM_CROPS
        closeup = look["leaf"] >= 0.20 and look["fruit"] < 0.08
        view = "leaf" if trust_cnn and closeup else "plant"
        guesses = []
        named = None
        if detail and not trust_cnn and self.use_dictionary:
            try:
                guesses = self.dictionary.match(img, min_score=self.dict_min)
            except Exception:
                guesses = []
            plants = [g for g in guesses if g.get("kind") == "plant"]
            junk = [g for g in guesses if g.get("kind") == "junk"]
            plant_hit = plants[0] if plants else None
            junk_hit = junk[0] if junk else None
            if plant_hit and (junk_hit is None or plant_hit["score"] >= junk_hit["score"]):
                named = plant_hit
                view = "plant"
                if plant_hit["id"] in FARM_CROPS:
                    crop = plant_hit["id"]
                    unknown = False
                    reason = "ok"
                    graded = self.dictionary.grade_health(img, crop)
                    if graded["health"] and graded["margin"] >= 0.04:
                        health = graded["health"]
                        health_conf = graded["confidence"]
                        health_scores = graded["scores"]
                    else:
                        health = None
                    crop_conf = plant_hit["score"]
            elif junk_hit and (plant_hit is None or junk_hit["score"] > plant_hit["score"]):
                reason = "not_a_leaf"
                unknown = True
                crop = None
                health = None
                view = "junk"
            elif look["vegetation"] < 0.06:
                reason = "not_a_leaf"
                unknown = True
                crop = None
                health = None
                view = "junk"
        facts = {
            "reject": reason == "not_a_leaf",
            "reason": reason or "ok",
            "crop": crop,
            "health": health,
            "view": view,
            "dictionary_guesses": guesses,
        }
        named_label = None
        if named is not None:
            named_label = named["name"] + (f" ({named['local']})" if named.get("local") else "")
        return {
            "crop": crop or "unknown",
            "health": health or "unknown",
            "crop_confidence": round(crop_conf, 4),
            "health_confidence": round(health_conf, 4),
            "crop_margin": round(margin, 4),
            "other_probability": round(other_p, 4),
            "in_list": round(in_p, 4),
            "leaf_score": look["leaf"],
            "skin_score": look["skin"],
            "fruit_score": look["fruit"],
            "fruit_hint": look.get("fruit_hint"),
            "fruit_round": look.get("fruit_round"),
            "fruit_skinny": look.get("fruit_skinny"),
            "view": view,
            "unknown": unknown,
            "reason": reason or "ok",
            "guess": self.crop_names[crop_i],
            "crop_scores": crop_scores,
            "health_scores": health_scores,
            "dictionary_guesses": guesses,
            "named_plant": named_label,
            "tip": word_tip(facts) if detail else "",
        }


def save_meta(payload: dict) -> None:
    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(payload, indent=2), encoding="utf-8")
