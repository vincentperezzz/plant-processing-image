# Study: faster crop name after the red box

**Project:** Plant Health Scanner (PC v0, Pi 4B 8 GB later)  
**Date:** 2026-08-14  
**Scope:** Time from **red box on screen** → **crop name on the label** (tomato, palay, sili, …). Not box-drawing speed. Not the English tip.  
**Status:** study only — **no code changes in this slice**

---

## Executive recommendation

The red box is the host seating you. The **crop name is the kitchen**. After a box appears, we still cook the name in a second brain — and often a third.

| If the photo is… | Who names it today | Why it feels slow |
| --- | --- | --- |
| Leaf close-up of a farm crop | CNN (`models/best.pt`, MobileNetV3-Small) | Usually already fast |
| Whole plant / greenhouse / fruit | CLIP dictionary (ViT-B/32) | This is the lag you feel |
| Anything, while Live is running | Same, **plus** YOLO-World still chewing the camera | Two heavy models share the GPU/CPU |

**Do this, in order, when we are allowed to touch code:**

1. **Measure** the three waits on this PC (CNN, CLIP, YOLO-World) with a stopwatch in the log. Guessing is how we pick the wrong fight.
2. **Live after box = CNN only.** CLIP (extra PH names) waits for **Snap**. Farm crop name should land in tens of milliseconds, not seconds.
3. **Name the biggest box first. Never re-CLIP a box that already has a name.**
4. **Pause YOLO-World while the CNN names a crop** so they do not fistfight on the GPU.
5. Only if that is still not enough: **put the five farm names on a YOLO-nano detector** so the box *is* the name — no second homework pass.

Do **not** reach for VGG19, a bigger CLIP, or Live CLIP on five boxes. That makes naming slower.

---

## What happens today (after the box)

Source: `src/scan_drop.py` (`_kick_grade`), `src/infer.py` (`scan(..., detail=)`), `src/detect.py`.

```text
red box appears, label = "plant"
        │
        ▼
  CNN on that crop  (detail=False)     ← pop quiz, five farm crops
        │
        ├─ sure it’s palay/sili/tomato/eggplant/lettuce
        │     → label becomes the crop name (+ health if CNN is sure)
        │
        └─ not sure (typical whole plant)
              → CLIP dictionary           ← essay question
              → label becomes a name (or stays vague)
```

Only **one box is named at a time**. Live also keeps running YOLO-World / the color finder on new frames.

Everyday version: we already circled the plant with a marker. Then we ask a student (CNN) “which of my five crops?” If they shrug, we walk to the library (CLIP). The celebrity box-drawer (YOLO-World) keeps talking in the hallway.

---

## Where the seconds go

These are **typical PC ranges**, not timings from this machine. Step 0 of the build is to replace this table with real numbers.

| Step after the box | Typical wait | Notes |
| --- | --- | --- |
| Cut the box out of the frame | < 5 ms | Cheap |
| CNN name (224 px, MobileNetV3-Small) | ~20–150 ms | Fast if it is trusted |
| CLIP name (ViT-B/32) | ~0.4–2+ s on CPU; less on CUDA | Dominant when CNN shrugs |
| First CLIP load in the session | several seconds, once | Feels like “the first Snap is frozen” |
| YOLO-World still running on Live | extra 0.2–1 s per camera look | Steals the same GPU the CNN wants |

**Rule of thumb:** if the label sticks on `plant` then jumps to a name, you waited on **CLIP**, not the CNN. If it never leaves `plant`, CLIP is busy or the CNN never got a turn.

Greenhouse / whole-plant shots (the tomato-in-a-room case) usually **fail the CNN confidence gate**, so they take the slow door on purpose. That is a product choice, not a bug — unless Live is supposed to name farm crops without the library.

---

## Options (impact vs cost)

| # | Change | Faster crop name? | Cost | Pi-safe later? |
| --- | --- | --- | --- | --- |
| A | Live: CNN name only; CLIP on Snap | **High** for farm crops | Extra PH names wait for shutter | Yes |
| B | Do not re-name a box that already has a crop | Medium (stops repeat CLIP) | Tiny | Yes |
| C | Grade largest box first; cap 1 on Live | Medium | Other plants wait | Yes |
| D | Pause detector while CNN runs | Medium if CUDA contention | Boxes update a bit less often | Yes |
| E | Lower CNN “I’m sure” bar **only inside a red box** | High if CNN is actually right on plants | More wrong names on junk-in-a-box | Yes, if we test it |
| F | YOLO-nano with 5 crop classes (+ plant) | **Highest** — name arrives with the box | Need box-labeled photos; AGPL if Ultralytics | Best Pi path |
| G | Smaller CLIP (MobileCLIP) or skip CLIP on Live forever | High for non-farm names | Weaker extra-species names | Maybe |
| H | Export CNN to TensorRT / ONNX | Low–medium; CNN is not the villain | Extra export pipeline | TFLite INT8 is the Pi version |
| I | Keep YOLO-World + Live CLIP | None | Status quo | **No** |

**F** is the “street YOLO” endgame: the red box already says `tomato`. **A–D** are the cheap week. **E** is a judgment call (speed vs lying).

---

## Recommended build sequence (when coding is allowed)

### 0. Stopwatch (half day)

Log three clocks per box: `t_cnn`, `t_clip`, `t_detect`. Ten Live frames of (1) a tomato leaf, (2) a whole tomato plant, (3) a desk with no plant. Write the medians into this doc. Do not skip this — A vs F depends on it.

### 1. Pipeline diet (1–2 days) — likely the win you want

- After a box: run CNN, write the farm name as soon as CNN is sure.
- Do **not** start CLIP on Live.
- Snap: then CLIP + health detail + tip.
- Skip CLIP if the box already has `crop`.
- Live: name **one** box (largest).
- Optionally pause YOLO-World during that CNN call.

Expected feel: leaf close-ups name almost instantly; whole plants show `plant` on Live and get a real name on Snap — unless we also do E or F.

### 2. Only if Live whole-plants must name without Snap

Pick **one**:

- **E** — trust CNN more when `assume_plant=True` (already boxed). Validate on junk-in-frame so we do not label a watering can “sili”.
- **F** — teach a nano detector the five crops with boxes (YOLO-World can auto-label a pile of photos, then we train nano). License: Ultralytics AGPL.

### 3. Pi (later, not this PC demo)

YOLO-World + CLIP Live will not fit a Pi 4 the way you want. Target: nano detector INT8 + MobileNet INT8, CLIP off the live path.

---

## What not to do for this goal

- Bigger backbone (VGG19) for naming — heavier, not faster.
- CLIP on every Live frame for every box.
- Waiting to draw the red box until the name is ready — that makes **boxes** feel slow, which you already solved.

---

## Decisions we still need

1. **Live crop name** — five farm crops only (fast CNN), or still try kape/mais/etc. while the camera moves (CLIP tax)?
2. **Wrong-name risk** — OK to guess faster inside a box (option E), or only speak when the CNN is very sure?
3. **Success number** — e.g. “farm crop name on the label within 200 ms after the box on this PC.” Without a number we cannot know if step 1 was enough.
