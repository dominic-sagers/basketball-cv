# Pretrained Basketball Models

This document covers pretrained model options and Roboflow Universe datasets
relevant to the basketball-cv pipeline.

The base `yolo11m.pt` weights (COCO-trained) detect `person` and `sports ball`
out of the box — enough to start testing. Fine-tuned basketball weights will
improve ball detection at distance, hoop detection, and reduce false positives
from non-player people (coaches, spectators).

---

## Model architecture options

### YOLOv11 (primary choice)

Already integrated in the pipeline. See `src/detector.py` and `src/tracker.py`.

- Inference: 60+ FPS on RTX 4080 Super at 1280px input
- ByteTrack built in via `model.track()`
- Pose estimation available via `yolo11m-pose.pt`
- Fine-tuning: straightforward with Ultralytics CLI or Python API

**Variants by speed/accuracy tradeoff:**

| Model | Speed (RTX 4080S) | Use case |
|---|---|---|
| `yolo11n.pt` | ~120+ FPS | CPU fallback / very tight margin |
| `yolo11s.pt` | ~90 FPS | Good if multi-camera headroom is needed |
| `yolo11m.pt` | ~60 FPS | Default — best balance for single stream |
| `yolo11l.pt` | ~40 FPS | Better accuracy if FPS allows |

### RF-DETR (alternative worth evaluating)

RF-DETR is Roboflow's real-time detection transformer, released March 2025.
Built on a DINOv2 vision transformer backbone rather than CNN, which gives
it notably better occlusion handling — relevant for 10-player pickup games
where bodies regularly overlap.

**Why it matters for basketball-cv:**
- Heavy occlusion (screens, pile-ups) is where YOLO-family models degrade most
- Transformer attention can better isolate players in crowded frames
- May reduce the "player disappears behind screen" tracking drop that causes missed events

**Performance on T4 GPU (slower than RTX 4080 Super):**

| Model | COCO mAP | FPS (T4) | Notes |
|---|---|---|---|
| RF-DETR-N | ~48 | ~45 FPS | Nano — fast but lower accuracy |
| RF-DETR-S | ~52 | ~33 FPS | Small — closest to YOLOv11m territory |
| RF-DETR-M | ~56 | ~25 FPS | May hit 40-50 FPS on RTX 4080 Super |
| RF-DETR-L | 60.5 | ~20 FPS | Best accuracy; benchmark before committing |

RTX 4080 Super is roughly 2.5-3x faster than T4, so RF-DETR-M/L may be viable.
**Must benchmark before building event logic on top of it.**

**License:** Nano through Large are Apache 2.0 (free, open source).
XL and 2XL are Roboflow proprietary (PML 1.0).

**Install:**
```bash
pip install rfdetr
```

**GitHub:** https://github.com/roboflow/rf-detr

**Decision:** Use YOLOv11 as default (already integrated, ByteTrack built in).
Evaluate RF-DETR on your test footage once you have it — if occlusion causes
frequent track loss, it's the first alternative to try.

---

## Pretrained basketball weights

### SwishAI (recommended starting point)

The most directly usable open-weights basketball model found.
Trained on Roboflow basketball datasets, `yolo11s` base.

**Classes:** `Ball`, `Ball in Basket`, `Player`, `Basket`, `Player Shooting`

This is the only public repo found that packages trained basketball `.pt` weights
without requiring a Roboflow paid plan.

**GitHub:** https://github.com/sPappalard/SwishAI
**Weights location:** `basketball_training/weights/best.pt`

To use in this pipeline, download `best.pt` and update `config.yaml`:
```yaml
model:
  weights: "store/weights/swishai_best.pt"
  class_map:
    0: "Ball"
    1: "Ball in Basket"
    2: "Player"
    3: "Basket"
    4: "Player Shooting"
```

**Caveat:** Trained on yolo11s, so slightly lower accuracy than yolo11m.
Worth testing as a starting point before fine-tuning on your own footage.

---

## Roboflow Universe datasets

These are the most useful public datasets for fine-tuning or evaluation.
All require a free Roboflow account to download (API key needed).

### Combined: players + ball + hoop

**Basketball Players (Roboflow Universe Projects)**
- URL: https://universe.roboflow.com/roboflow-universe-projects/basketball-players-fy4c2
- Classes: `Ball`, `Hoop`, `Period`, `Player`, `Ref`, `Shot Clock`, `Team Name`, `Team Points`, `Time Remaining`
- Most widely forked basketball dataset on Universe
- Useful for scoreboard OCR groundwork too (Phase 2)

**basketball-players-and-ball1 (tickstrike)**
- URL: https://universe.roboflow.com/tickstrike/basketball-players-and-ball1
- Classes: `team1`, `team2`, `ball`, `ref`, `time`, `hoop`
- 511 images — smaller but has team separation labels

### Ball + hoop (no players)

**Basketball and Net Detection**
- URL: https://universe.roboflow.com/sc-xqmxu/basketball-and-net-detection/dataset/7
- 5,482 images — largest basketball detection dataset found
- Classes: ball + net/hoop
- Best dataset for fine-tuning ball and hoop detection accuracy

**Basketball Detection (ball + hoop)**
- URL: https://universe.roboflow.com/basketball-detection-b977c/basketball-detection-sskux/dataset/7
- ~1,300 images
- Updated December 2025

### Players only

**basketball-player-detection-3 (Roboflow official)**
- URL: https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo/dataset/6
- 654 images, player class only

### Court segmentation

**Basketball Courts (baskemtball)**
- URL: https://universe.roboflow.com/baskemtball/basketball-courts
- Instance segmentation of court regions
- Useful for Phase 1 court homography calibration

---

## Downloading datasets from Roboflow Universe

Requires a free Roboflow account and API key (no paid plan needed for dataset download).

```bash
pip install roboflow
```

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_FREE_API_KEY")

# Example: largest ball+hoop dataset
project = rf.workspace("sc-xqmxu").project("basketball-and-net-detection")
dataset = project.version(7).download("yolov11")
# Downloads to ./basketball-and-net-detection-7/ with images + labels in YOLOv11 format
```

**To get your API key:**
1. Create free account at https://roboflow.com
2. Go to Settings → API Keys
3. Copy the key into the snippet above

---

## Fine-tuning on your own footage

Once you have game footage and a baseline model working, the most valuable
thing you can do is label a few hundred frames from your specific gym and
fine-tune on them. Gym fluorescent lighting, your camera angles, and your
court markings are all distribution-shifted from the training data.

**Recommended workflow:**
1. Run `pipeline_test.py --save-output` to record annotated clips
2. Review frames where detections are wrong or missing
3. Label those frames in Roboflow (free tier: up to 1,000 images/month)
4. Export as YOLOv11 format and fine-tune:

```bash
yolo detect train \
  data=path/to/dataset/data.yaml \
  model=yolo11m.pt \
  epochs=50 \
  imgsz=1280 \
  device=0
```

5. Put the resulting `best.pt` in `store/weights/` and update `config.yaml`

Even 200-300 labeled frames from your gym will meaningfully improve detection
quality for your specific setup.

---

## Recommended path forward

1. **Start now:** Run `yolo11m.pt` (base COCO weights) against your test footage.
   Good enough to see players and ball, assess tracking stability.

2. **Quick win:** Download SwishAI's `best.pt` — already trained on basketball,
   adds hoop detection and `Ball in Basket` class for shot detection.

3. **Best ball/hoop accuracy:** Download the `basketball-and-net-detection` dataset
   (5,482 images) and fine-tune `yolo11m.pt` on it. This is the highest-leverage
   fine-tuning target before you have your own footage.

4. **Occlusion issues?** Benchmark RF-DETR-S or RF-DETR-M on your test footage.
   If track loss during screens/pile-ups is causing missed events, RF-DETR's
   transformer attention is the strongest available alternative.

5. **Endgame:** Label ~300 frames from your actual gym with your actual cameras.
   Fine-tune on that. Your mileage from any of the above will plateau without
   in-distribution training data.
