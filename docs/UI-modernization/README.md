# Handoff: Plant Health Kiosk — modernized UI

## Overview
A modernized redesign of the Plant Health kiosk's two screens — **Scan** (live camera + grading HUD) and **Gallery** (past scans) — for the 7" 1024×600 Raspberry Pi touchscreen. Goal: easier to read at a glance, in varying indoor/outdoor light, for student users.

## About the design files
The bundled file (`reference/Plant Health Kiosk.dc.html`) is an **HTML/CSS design reference**, built to preview the look and layout — it is not code to run on the Pi. **The target runtime stays Python: Tkinter + Pillow**, exactly as today's `src/pi_sim.py`. That file already composites rounded HUD cards, badges and the shutter onto the camera frame with `PIL.ImageDraw` — this redesign is a new set of values (colors, radii, type, spacing, layout) fed into that same drawing pipeline, not a rewrite of the architecture and not an embedded browser.

## Fidelity
**High-fidelity.** Colors below are exact (hex, converted from the design system's OKLCH values so they drop straight into Pillow/Tkinter, which don't support `oklch()`). Typography, radii and spacing are exact. Copy text is exact.

## Screens

### 1. Scan (`_build_scan` / `_compose_frame` / `_draw_card` in `pi_sim.py`)
**Purpose:** live camera view; grades whatever plant is framed and shows a shutter to save it.

**Layout** (1024×600):
- Top nav bar, full width, 64px tall, `--color-surface` background, 1px bottom border in `--color-divider`.
  - Left: leaf icon (24px, stroke 2.75, `#036819`) + "Plant Health" wordmark, Caprasimo 18px.
  - Right-aligned group: "Scan" / "Gallery" tab buttons (pill, 44px min height, active tab = solid accent fill `#c67139` with cream text, inactive = outlined `--color-divider` border), then "Exit" button (outlined, text/border in `#a5292b`).
- Below nav: camera viewfinder fills the remaining 1024×536.
  - **LIVE badge**: top-left, 16px inset. Pill, background `#a5292b`, white text, 12px bold, small white dot.
  - **Detection box(es)**: rounded-rect outline over the detected plant, 3px, `--color-accent-2-400` (`#aebf92`) for the primary box; a second box (dashed, `--color-neutral-400` `#c0b6a5`) when more than one plant is in frame.
  - **Box picker chips** (only when multiple plants detected): centered top, two 40px circular pill buttons numbered "1"/"2". Active = solid accent fill; inactive = dark translucent fill (`rgba(20,20,20,0.55)`) with a white 1.5px border and white text — inactive chips must NOT reuse the light-theme "secondary" button style, since they sit on the dark camera feed and need their own dark-surface treatment to stay legible.
  - **Plant type card**: floating top-left (16px inset from the edge, right under the LIVE badge), 236px wide. Cream card (`--color-surface`), label "Plant type" (11px, uppercase-ish caption, accent color), value in Caprasimo 26px.
  - **Plant health card**: directly under the type card, same width, 10px gap. Background/border/text swap by grade — see Design Tokens → Health colors. Includes an outlined confidence pill, e.g. "96% confident", 11px, nowrap (must not wrap — wrapping pushes text outside the card).
  - **Notes card**: bottom-left, 16px inset, fixed width 560px, max-height 110px (clip/ellipsis rather than grow — a full-width, unbounded-height notes card was tried first and it overwhelmed the screen). Cream card, "Notes" caption, 14px body tip text. When multiple plants are in frame, appends " Box 1 of 2 — tap the other box to inspect it." at 65% opacity.
  - **Shutter button**: vertically centered, 24px from the right edge. Classic camera shutter, not a filled pill button: 88px circle, transparent fill, 5px white ring (`rgba(255,255,255,0.92)`), with a smaller solid white 70px disk centered inside.

### 2. Gallery (`_build_gallery` / `_fill_gallery` / `_paint_gallery` in `pi_sim.py`)
**Purpose:** browse past scans, inspect one, delete it.

**Layout:**
- Same top nav as Scan, with "Gallery" as the active tab.
- Left rail, 260px wide, full height, `--color-surface` background, scrollable, 1px right border in `--color-divider`. One row per photo, 12px padding, min-height 68px, containing: crop name (16px) + health tag pill (top row), then a caption line "{time} · {confidence}% confidence" (11px, muted). The selected row gets a raised shadow (`--shadow-md`).
- Right pane: the selected photo, letterboxed with 24px inset, rounded corners (`--radius-lg`).
  - **Info + notes card**: bottom-left, 44px inset, 420px wide. Row of crop name (Caprasimo 20px) + health tag + timestamp, then the saved tip text (13px, 85% opacity) underneath — the tip must be shown here, not only on the live Scan screen, so a grade's reasoning is still visible when reviewing later.
  - **Delete button**: bottom-right, 44px inset, 48px icon button, trash icon in `#a5292b`. (No download/export/share action — deliberately left out.)

## Design tokens

### Color — health grades (exact hex, source: OKLCH via design system)
| Grade | Background | Text/value | Border/accent |
|---|---|---|---|
| Healthy | `#cdf0cd` | `#004906` | `#43a047`-ish mid step (use `oklch(55% 0.14 145)` if porting via a color lib, else `#3fae4a`) |
| Mild | `#fae2b0` | `#6e3800` | `#c58d04` |
| Critical | `#ffe4e1` | `#83000d` | `#c74b47` |
| Dead | `#dcd3c4` (neutral-300) | `#2e2b25` (neutral-900) | `#474238` (neutral-800) |

### Other colors
| Token | Hex | Use |
|---|---|---|
| Cream ground | `#f5ead8` | page/viewfinder-adjacent background |
| Card surface | `#ebddc5` | all floating cards, nav bar |
| Text | `#201e1d` | default text |
| Accent (terracotta) | `#c67139` | primary buttons, active tab |
| Accent-2 (sage) | `#7a8a5e` | secondary accents |
| Alert red | `#a52929` | LIVE badge, Exit button, delete icon |
| Neutral-900 (viewfinder well) | `#2e2b25` | camera area background before/behind the live feed |
| Leaf icon green | `#036819` | wordmark icon |

### Typography
- Headings / values: **Caprasimo** (single weight 400) — needs its `.ttf` bundled under `vendor/fonts/` for fully offline use (it's a Google Font; the Pi has no internet at runtime). Load via `ImageFont.truetype()` the same way `_load_font()` already does for DejaVu/Liberation.
- Body / labels: **Figtree**, weights 400/600/700 — same bundling requirement.
- Sizes used: 26px card values, 22px/20px photo captions, 18px wordmark, 16px/15px buttons, 14px notes body, 13px meta, 12px/11px badges and captions.

### Spacing & radius
- Card radius: 16px (`--radius-md`). Larger surfaces (dialog-scale): 28px (`--radius-lg`).
- Buttons/pills/chips: fully round, `border-radius: 999px`.
- Nav height: 64px. Standard inset from screen edges: 16px (Scan) / 24–44px (Gallery, more breathing room since there's no live feed motion to fight).

### Icons
Lucide, stroke-width 2.75, `currentColor`. Used: `leaf` (wordmark), `camera` (unused now — shutter is a drawn ring, not an icon), `trash-2` (gallery delete).

## Interactions & behavior
- Scan ⇄ Gallery: tab click swaps the visible screen; no animation needed to port first.
- Box picker chips: tapping "2" re-targets which detected plant the health card/notes describe (mirrors existing `_focus_track()` / `_picked_tid` logic in `pi_sim.py` — no new state machine needed, just re-skin the existing selection).
- Shutter tap: unchanged from today's `snap()` — flash + shutter "punch" animation, write PNG, append gallery record.
- Gallery row tap: swaps the right-pane preview + its info/notes card (mirrors existing `_open_photo()`).
- Delete tap: not implemented in current `pi_sim.py` (today's gallery is read-only) — this redesign specifies a delete affordance; wire it to remove the record/file (see `src/records.py`) and refresh `_fill_gallery()`.

## State management
No new state beyond what `pi_sim.py` already tracks (`_page`, `_tracks`, `_picked_tid`, `_hud_*`, `_gal_src`/`_gal_path`). Add one boolean-ish action: delete-current-photo, which should invalidate `_gal_src`/refresh the thumbnail list.

## Assets
- Leaf, trash-2 icons — Lucide, redrawn as plain SVG paths in the reference file; convert to whatever the Tkinter/Pillow side draws icons with today (looks like none currently — this introduces the first icon use, so decide once: hand-draw with `ImageDraw` primitives, or rasterize small PNGs from the Lucide paths at build time and cache them).
- Caprasimo + Figtree font files — not included in this handoff; download once and vendor the `.ttf`s for offline Pi installs (same pattern as `deploy/fetch-pi-offline.py` / `vendor/` already used for Python wheels).

## Files
- `reference/Plant Health Kiosk.dc.html` — the full HTML design reference (both screens, tweakable health-grade demo).
- `reference/design-tokens.css` — the source design-system stylesheet the hex values above were derived from (OKLCH ramps, spacing/radius/shadow scale).
- Existing repo files this replaces the values in: `src/pi_sim.py` (`_build_scan`, `_build_gallery`, `_draw_card`, `_hud_boxes`, `_sync_shutter`, `_fill_gallery`, `_paint_gallery`, and the color constants at the top of the file).
