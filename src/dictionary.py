from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import yaml
from PIL import Image

from src.paths import DICTIONARY, DICTIONARY_CACHE, ROOT

HF_HOME = ROOT / "data" / ".cache" / "hf"


class PlantDictionary:
    def __init__(self, path: Path | None = None, device: torch.device | None = None):
        self.path = path or DICTIONARY
        spec = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.clip_id = spec.get("clip_model", "openai/clip-vit-base-patch32")
        self.min_score = float(spec.get("min_score", 0.24))
        self.top_k = int(spec.get("top_k", 3))
        self.entries = list(spec.get("entries") or [])
        self.health_phrases = list(spec.get("health_phrases") or [])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._text_z = None
        self._index = []
        self._health_z = None
        self._health_index = []

    def _fingerprint(self) -> str:
        raw = self.path.read_bytes()
        return hashlib.sha256(raw + self.clip_id.encode("utf-8")).hexdigest()

    def _load_clip(self) -> None:
        if self._model is not None:
            return
        import os

        os.environ.setdefault("HF_HOME", str(HF_HOME))
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.clip_id)
        self._model = CLIPModel.from_pretrained(self.clip_id)
        self._model.to(self.device)
        self._model.eval()

    def _encode_texts(self) -> None:
        key = self._fingerprint()
        if DICTIONARY_CACHE.exists():
            blob = torch.load(DICTIONARY_CACHE, map_location="cpu", weights_only=False)
            if blob.get("key") == key:
                self._text_z = blob["text_z"].to(self.device)
                self._index = blob["index"]
                hz = blob.get("health_z")
                self._health_z = hz.to(self.device) if hz is not None else None
                self._health_index = blob.get("health_index") or []
                return
        self._load_clip()
        phrases = []
        index = []
        for entry in self.entries:
            for phrase in entry.get("phrases") or []:
                phrases.append(str(phrase))
                index.append(
                    {
                        "id": entry["id"],
                        "kind": entry.get("kind", "plant"),
                        "name": entry.get("name", entry["id"]),
                        "local": entry.get("local", ""),
                    }
                )
        if not phrases:
            self._text_z = torch.zeros((0, 512), device=self.device)
            self._index = []
            return
        inputs = self._processor(text=phrases, padding=True, truncation=True, return_tensors="pt")
        inputs = {name: val.to(self.device) for name, val in inputs.items()}
        with torch.no_grad():
            z = self._model.get_text_features(**inputs).pooler_output
            z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self._text_z = z
        self._index = index
        health_texts = [str(row["text"]) for row in self.health_phrases]
        self._health_index = [
            {"level": row["level"], "crop": row.get("crop") or ""}
            for row in self.health_phrases
        ]
        if health_texts:
            h_in = self._processor(text=health_texts, padding=True, truncation=True, return_tensors="pt")
            h_in = {name: val.to(self.device) for name, val in h_in.items()}
            with torch.no_grad():
                hz = self._model.get_text_features(**h_in).pooler_output
                hz = hz / hz.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self._health_z = hz
        else:
            self._health_z = None
        DICTIONARY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "key": key,
                "text_z": z.cpu(),
                "index": index,
                "health_z": None if self._health_z is None else self._health_z.cpu(),
                "health_index": self._health_index,
            },
            DICTIONARY_CACHE,
        )

    def _image_z(self, img: Image.Image) -> torch.Tensor:
        self._load_clip()
        inputs = self._processor(images=img.convert("RGB"), return_tensors="pt")
        inputs = {name: val.to(self.device) for name, val in inputs.items()}
        with torch.no_grad():
            z = self._model.get_image_features(**inputs).pooler_output
            z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return z

    def match(self, img: Image.Image, min_score: float | None = None, top_k: int | None = None) -> list[dict]:
        self._encode_texts()
        if self._text_z is None or self._text_z.numel() == 0:
            return []
        k = top_k or self.top_k
        floor = self.min_score if min_score is None else min_score
        z = self._image_z(img)
        sim = (z @ self._text_z.T)[0]
        best: dict[str, dict] = {}
        for i, score in enumerate(sim.tolist()):
            meta = self._index[i]
            prev = best.get(meta["id"])
            if prev is None or score > prev["score"]:
                best[meta["id"]] = {
                    "id": meta["id"],
                    "kind": meta["kind"],
                    "name": meta["name"],
                    "local": meta["local"],
                    "score": round(float(score), 4),
                }
        ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)
        return [r for r in ranked if r["score"] >= floor][:k]

    def grade_health(self, img: Image.Image, crop_id: str) -> dict:
        self._encode_texts()
        levels = ["healthy", "mild", "critical", "dead"]
        empty = {lv: 0.0 for lv in levels}
        if self._health_z is None or self._health_z.numel() == 0:
            return {"health": None, "confidence": 0.0, "scores": empty, "margin": 0.0}
        z = self._image_z(img)
        sim = (z @ self._health_z.T)[0]
        best = {lv: -1.0 for lv in levels}
        for i, score in enumerate(sim.tolist()):
            meta = self._health_index[i]
            crop = meta.get("crop") or ""
            if crop not in ("", crop_id):
                continue
            lv = meta["level"]
            if score > best[lv]:
                best[lv] = score
        logits = torch.tensor([best[lv] for lv in levels], dtype=torch.float32)
        if float(logits.max()) < 0:
            return {"health": None, "confidence": 0.0, "scores": empty, "margin": 0.0}
        p = torch.softmax(logits * 25.0, dim=0)
        ordered = torch.topk(p, k=2)
        conf = float(ordered.values[0])
        margin = float(ordered.values[0] - ordered.values[1])
        health = levels[int(ordered.indices[0])]
        scores = {lv: round(float(p[i]), 4) for i, lv in enumerate(levels)}
        return {"health": health, "confidence": round(conf, 4), "scores": scores, "margin": round(margin, 4)}
