# Model Training Guide

Fine-tuning YOLOv11 on basketball-specific data to improve detection of the
ball, hoop, and shooting events beyond what the base COCO model provides.

---

## Why fine-tune?

The base `yolo11m.pt` model (COCO-trained) detects `person` and `sports ball`
but was not trained on basketball footage. Benchmarking on our test clip showed:

| Config | Ball detection rate | Avg FPS |
|---|---|---|
| yolo11m.pt, imgsz=1280, conf=0.4 | 14.8% | 26.5 |
| yolo11m.pt, imgsz=960, conf=0.4  | ~14%  | 48.5 |

Key failure modes of the base model:
- Ball not detected when held by a player (overlapping bounding boxes)
- Ball lost mid-arc due to motion blur and small pixel footprint
- No hoop/basket detection at all (not in COCO)
- No "Player Shooting" class for shot attempt detection

Fine-tuning on basketball-specific data adds all five classes and teaches the
model what a basketball looks like from gym camera angles.

---

## Dataset

**Source:** [Roboflow Universe — basketball-detection-srfkd](https://universe.roboflow.com/basketball-6vyfz/basketball-detection-srfkd)

| Split | Images |
|---|---|
| Train | 9,599 |
| Val | 873 |
| Test | — |

**Classes (5):**

| ID | Name | Relevance |
|---|---|---|
| 0 | Ball | Core tracking target |
| 1 | Ball_in_Basket | Made shot detection |
| 2 | Player | Person tracking |
| 3 | Basket | Hoop zone definition |
| 4 | Player_Shooting | Shot attempt detection |

Local path: `store/dataset/basketball-srfkd/` (DVC-tracked — pull via `dvc pull`)

---

## Get the dataset

### Option A — DVC pull (recommended)

If you have been granted access to the DVC remote, this restores the full
`store/` directory including the dataset, weights, and footage in one command:

```bash
dvc pull
```

See `docs/dvc-setup.md` for Tailscale access setup.

### Option B — Download from Roboflow

If you don't have DVC access, download directly. Requires a free
[Roboflow](https://roboflow.com) account and API key.

```bash
pip install roboflow
```

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("basketball-6vyfz").project("basketball-detection-srfkd")
dataset = project.version(1).download("yolov11", location="store/dataset/basketball-srfkd")
```

The `data.yaml` paths are already configured relative to the dataset directory —
no edits needed.

---

## Dataset layout

All datasets live under `store/dataset/`, one named subdirectory each:

```
store/dataset/
    basketball-srfkd/          ← Roboflow basketball-detection-srfkd (current)
        data.yaml
        train/images/
        valid/images/
        test/images/
    my-gym-footage/            ← custom-labelled footage (example future dataset)
        data.yaml
        train/images/
        ...
```

To switch the active dataset, change one line in `config.yaml`:

```yaml
training:
  dataset: "store/dataset/my-gym-footage/data.yaml"
```

---

## Run training

All training parameters come from `config.yaml → training:`. Run from the project root:

```bash
python src/train.py
```

To resume an interrupted run:

```bash
python src/train.py --resume
```

This reads `training.dataset`, `training.base_model`, `training.epochs`, etc. from
config.yaml and calls the Ultralytics trainer. Weights are saved to
`store/weights/<output_name>/`.

### Parameter notes

| config.yaml key | Default | Reason |
|---|---|---|
| `base_model` | `yolo11m.pt` | Medium — best speed/accuracy on RTX 4080S |
| `epochs` | 50 | Enough for a clean dataset; increase to 100 if mAP plateaus |
| `imgsz` | 960 | Matches inference config; smaller than 1280 for FPS headroom |
| `batch` | 8 | Safe for 16GB VRAM at imgsz=960; batch=16 causes OOM on RTX 4080 Super |
| `device` | 0 | First GPU (RTX 4080 Super) |
| `output_name` | `basketball-ft` | Weights saved to `store/weights/<output_name>/` |

Expected training time: **~1 hr/epoch** on RTX 4080 Super at batch=8, imgsz=960
(50 epochs ≈ 50 hours for a 9,599-image dataset).

### Monitoring training

Ultralytics prints a live progress table per epoch:

```
Epoch  GPU_mem   box_loss  cls_loss  dfl_loss  Instances  Size
1/50   4.21G     1.842     2.341     1.187     142        960
...
50/50  4.21G     0.923     0.814     0.981     138        960
```

Watch `box_loss` and `cls_loss` — both should trend down. If they plateau
before epoch 50, training is complete. If still dropping at epoch 50, rerun
with `epochs=100 resume=True`.

Results and plots saved to `store/weights/basketball-ft/`:
- `store/weights/basketball-ft/weights/best.pt` — best checkpoint by val mAP
- `store/weights/basketball-ft/weights/last.pt` — final epoch checkpoint
- `store/weights/basketball-ft/results.png` — loss and mAP curves

---

## Use the fine-tuned model

Update `config.yaml`:

```yaml
model:
  weights: "store/weights/basketball-ft/weights/best.pt"
  device: 0
  confidence: 0.4
  iou_threshold: 0.5
  input_size: 960
  class_map:
    0: "Ball"
    1: "Ball_in_Basket"
    2: "Player"
    3: "Basket"
    4: "Player_Shooting"
```

Then run the pipeline as normal:

```powershell
python src/pipeline_test.py `
  --file "store/footage/basketballcv-sample-1.mov" `
  --no-preview `
  --save-output "store/output/sample-1-basketball-ft.mp4" `
  --save-log "store/output/sample-1-basketball-ft-log.json"
```

Compare the `ball_detection_rate_pct` in the log against the 14.8% baseline.
Target: **≥50%** before moving on to `ball_tracker.py`.

---

## Adding a new dataset

1. Download or export the dataset in YOLOv11 format into a new named subdirectory:

   ```
   store/dataset/<your-dataset-name>/
       data.yaml
       train/images/
       valid/images/
       test/images/
   ```

2. Make sure `data.yaml` paths are relative to the `data.yaml` file itself
   (i.e. `train: train/images`, not absolute paths).

3. Update `config.yaml`:

   ```yaml
   training:
     dataset: "store/dataset/<your-dataset-name>/data.yaml"
     output_name: "basketball-ft-v2"   # keep weights from different runs separate
   ```

4. Run `python src/train.py`.

5. After training, version and push the new weights:

   ```bash
   dvc add store
   dvc push
   git add store.dvc
   git commit -m "Add weights: <dataset-name>, <result summary>"
   ```

---

## Re-train with your own footage (next step)

Once we have enough labelled game footage, fine-tuning on your specific gym
will give the biggest accuracy boost. The Roboflow dataset is a general
basketball dataset — your gym's lighting, camera angles, and court markings
are all different.

Workflow:
1. Use `pipeline_test.py --save-output` to capture annotated clips from real games
2. Label frames where detection fails in [Roboflow](https://app.roboflow.com) (free tier: 1,000 images/month)
3. Export as YOLOv11 format and merge with the existing dataset
4. Re-run training with `model="store/weights/basketball-ft/weights/best.pt"` as starting point
   (fine-tune the fine-tune — faster convergence than starting from COCO)

Even 200–300 labelled frames from your gym will meaningfully improve results.
