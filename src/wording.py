from __future__ import annotations

import json
import os
from typing import Any

from src.tips import tip_for

SYSTEM = (
    "You write one or two short sentences of advice for a Filipino farmer. "
    "Use only the JSON facts. Do not invent a crop, disease, or treatment. "
    "If reject is true, tell them this is not a plant and to photograph the crop. "
    "If crop and health are set, grade that plant in plain English. "
    "If view is plant, talk about the whole plant, not only a leaf. English only."
)


def _facts_payload(facts: dict[str, Any]) -> dict:
    return {
        "reject": bool(facts.get("reject")),
        "reason": facts.get("reason") or "ok",
        "crop": facts.get("crop"),
        "health": facts.get("health"),
        "view": facts.get("view") or "leaf",
        "dictionary_guesses": facts.get("dictionary_guesses") or [],
    }


def _ollama(facts: dict[str, Any]) -> str | None:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
    try:
        import urllib.request

        body = json.dumps(
            {
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(_facts_payload(facts), ensure_ascii=False)},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            host.rstrip("/") + "/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("message") or {}).get("content") or ""
        text = text.strip()
        return text or None
    except Exception:
        return None


def _hf(facts: dict[str, Any]) -> str | None:
    model_id = os.environ.get("LLM_MODEL")
    if not model_id:
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        prompt = SYSTEM + "\n\n" + json.dumps(_facts_payload(facts), ensure_ascii=False) + "\n\nTip:"
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        return text or None
    except Exception:
        return None


def _template(facts: dict[str, Any]) -> str:
    reason = facts.get("reason")
    guesses = facts.get("dictionary_guesses") or []
    plants = [g for g in guesses if g.get("kind") == "plant"]
    junk = [g for g in guesses if g.get("kind") == "junk"]
    if reason == "not_a_leaf" or (junk and (not plants or junk[0]["score"] >= plants[0]["score"])):
        hit = junk[0] if junk else None
        if hit:
            return (
                f"This looks like {hit['name']}, not a crop. "
                "Point the camera at the plant so leaves or fruit fill the frame, and scan again."
            )
        return tip_for(None, None, True, "not_a_leaf")
    crop = facts.get("crop")
    health = facts.get("health")
    view = facts.get("view") or "leaf"
    if crop not in {None, "unknown"} and health not in {None, "unknown"}:
        return tip_for(crop, health, False, "ok")
    if crop not in {None, "unknown"}:
        local = ""
        if plants:
            local = f" ({plants[0]['local']})" if plants[0].get("local") else ""
        return (
            f"This looks like {crop}{local}. "
            "I cannot grade health from this photo with enough confidence. "
            "Move closer so leaves or the whole plant fill the frame, then scan again."
        )
    if plants:
        top = plants[0]
        local = f" ({top['local']})" if top.get("local") else ""
        extra = ""
        if len(plants) > 1:
            extra = " Also close: " + ", ".join(
                (p["local"] or p["name"]) for p in plants[1:3]
            ) + "."
        return (
            f"This looks like {top['name']}{local}, not one of the farm crops I grade "
            f"(palay, sili, tomato, eggplant, lettuce).{extra}"
        )
    unknown = bool(facts.get("reject")) or crop in {None, "unknown"}
    return tip_for(None if unknown else crop, None if unknown else health, unknown, reason)


def word_tip(facts: dict[str, Any]) -> str:
    return _ollama(facts) or _hf(facts) or _template(facts)
