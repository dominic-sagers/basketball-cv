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

### basketball-ft (current — fine-tuned in-project)

Fine-tuned `yolo11m` on the Roboflow basketball-detection-srfkd dataset (9,599 images, 50 epochs).
Weights are DVC-tracked at `store/weights/basketball-ft.pt` — pull with `dvc pull`.

| Metric | Value |
|---|---|
| Ball detection rate | 27.1% (↑ from 14.8% COCO baseline) |
| Overall mAP50 | 0.956 |
| Ball mAP50 | 0.943 |
| Basket mAP50 | 0.979 |
| Inference FPS | ~48 at imgsz=960 on RTX 4080 Super |

**Classes:** `Ball`, `Ball_in_Basket`, `Player`, `Basket`, `Player_Shooting`

This is the active model in `config.yaml`. Reproduce it with `python src/train.py`.

---

### SwishAI (alternative starting point)

Open-weights basketball model trained on Roboflow datasets, `yolo11s` base.

**GitHub:** https://github.com/sPappalard/SwishAI

**Note:** The `.pt` weights are not included in the repo — only the training code
is published. Would need to be trained from their scripts before use.

**Classes:** `Ball`, `Ball in Basket`, `Player`, `Basket`, `Player Shooting`

**Caveat:** `yolo11s` base — lower accuracy than our `yolo11m` fine-tune.

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

Place the dataset under `store/dataset/<name>/` and update `config.yaml`:
```yaml
training:
  dataset: "store/dataset/<name>/data.yaml"
  output_name: "basketball-ft-v2"
```
Then run:
```bash
python src/train.py
```

5. After training, version the new weights: `dvc add store && dvc push && git add store.dvc && git commit`

Even 200-300 labeled frames from your gym will meaningfully improve detection
quality for your specific setup.

---

## Current state and path forward

**Done:**
- Fine-tuned `yolo11m` on basketball-detection-srfkd (9,599 imgs) — 27.1% ball detection, mAP50=0.956
- Scoring logic live: `Ball_in_Basket` detections drive score with 45-frame debounce
- Weights DVC-tracked on the remote — `dvc pull` to restore

**Next:**
1. **More data — your gym:** Run `pipeline_test.py --save-output` on real game footage.
   Label frames where detection fails in Roboflow (free tier: 1,000 images/month).
   Even 200–300 frames from your court will push ball detection well past 50%.
   Use `python src/train.py` with the new dataset — fine-tuning from `basketball-ft.pt`
   converges faster than starting from COCO.

2. **More data — additional datasets:** The `basketball-and-net-detection` dataset
   (5,482 images, ball + hoop) on Roboflow Universe is the highest-leverage public
   dataset to add next. Merge it with `basketball-srfkd` and retrain.

3. **Occlusion issues?** If ByteTrack loses players during screens and causes missed
   events, benchmark RF-DETR-S or RF-DETR-M. Transformer attention handles heavy
   occlusion better than YOLO's CNN backbone.
