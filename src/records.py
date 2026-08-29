from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.paths import SCANS_DB, SCANS_DIR


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
    cols = {row["name"] for row in con.execute("PRAGMA table_info(scans)")}
    if "confidence" not in cols:
        con.execute("ALTER TABLE scans ADD COLUMN confidence REAL")
    con.commit()
    return con


def write_png(img: Image.Image) -> str:
    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = SCANS_DIR / f"{stamp}.png"
    img.convert("RGB").save(path, format="PNG", optimize=True)
    return str(path)


def add_scan(
    *,
    crop: str | None,
    health: str | None,
    named_plant: str | None,
    tip: str,
    image_path: str,
    confidence: float | None = None,
) -> int:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO scans (created_at, crop, health, named_plant, tip, image_path, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (created, crop or "", health or "", named_plant or "", tip or "", image_path, confidence),
        )
        con.commit()
        return int(cur.lastrowid)


def list_scans(limit: int = 200) -> list[dict]:
    with _connect() as con:
        rows = con.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def list_photos(limit: int = 200) -> list[Path]:
    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        p
        for p in SCANS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def delete_scan(scan_id: int, image_path: str) -> None:
    with _connect() as con:
        con.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        con.commit()
    try:
        Path(image_path).unlink(missing_ok=True)
    except OSError:
        pass
