from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.paths import SCANS_CSV, SCANS_DB, SCANS_DIR


def _connect() -> sqlite3.Connection:
    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(SCANS_DB)
    con.row_factory = sqlite3.Row
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            crop TEXT,
            health TEXT,
            named_plant TEXT,
            tip TEXT,
            image_path TEXT
        )
        """
    )
    con.commit()
    return con


def write_jpeg(img: Image.Image) -> str:
    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = SCANS_DIR / f"{stamp}.jpg"
    img.convert("RGB").save(path, quality=85)
    return str(path)


def save_snapshot(img: Image.Image) -> int:
    path = write_jpeg(img)
    return add_scan(crop=None, health=None, named_plant=None, tip="", image_path=path)


def add_scan(
    *,
    crop: str | None,
    health: str | None,
    named_plant: str | None,
    tip: str,
    image_path: str,
) -> int:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO scans (created_at, crop, health, named_plant, tip, image_path) VALUES (?, ?, ?, ?, ?, ?)",
            (created, crop or "", health or "", named_plant or "", tip or "", image_path),
        )
        con.commit()
        return int(cur.lastrowid)


def list_scans(limit: int = 200) -> list[dict]:
    with _connect() as con:
        rows = con.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def export_csv(dest: Path | None = None) -> Path:
    path = dest or SCANS_CSV
    rows = list_scans(limit=10000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "created_at", "crop", "health", "named_plant", "tip", "image_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path
