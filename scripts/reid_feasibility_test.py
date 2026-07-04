"""
Re-ID feasibility test using existing game footage + track log.

Uses a pretrained ResNet50 (ImageNet) as a proxy for a real re-ID backbone.
If a generic backbone can separate players, a re-ID-specific model (OSNet) will do better.

Outputs:
  - Intra-track vs inter-track cosine similarity stats
  - t-SNE plot of embeddings colored by track ID
  - Similarity matrix heatmap for top tracks
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CROP_H, CROP_W = 256, 128  # standard re-ID crop size
SAMPLE_EVERY = 30           # sample one frame per second (30fps)
MIN_TRACK_FRAMES = 500      # only consider well-established tracks
MAX_TRACKS = 20             # top N tracks to analyze
CROPS_PER_TRACK = 30        # crops sampled per track


def build_model(device: torch.device) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()  # 2048-dim embedding
    model.eval().to(device)
    return model


def get_transform() -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize((CROP_H, CROP_W)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_track_index(log_path: Path) -> dict[int, list[tuple[int, list[int]]]]:
    """Returns {track_id: [(frame_num, bbox), ...]} for Player tracks."""
    with open(log_path) as f:
        data = json.load(f)

    index: dict[int, list] = defaultdict(list)
    for frame in data["frames"]:
        for obj in frame.get("objects", []):
            if obj["class"] == "Player":
                index[obj["track_id"]].append((frame["frame"], obj["bbox"]))

    return index


def sample_crops(
    video_path: Path,
    track_index: dict[int, list],
    top_tracks: list[int],
    transform: T.Compose,
) -> dict[int, list[torch.Tensor]]:
    """Extract and transform crops for each top track."""
    # Build frame -> [(track_id, bbox)] lookup for efficient seeking
    needed_frames: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for tid in top_tracks:
        entries = track_index[tid]
        sampled = entries[::SAMPLE_EVERY][:CROPS_PER_TRACK]
        for frame_num, bbox in sampled:
            needed_frames[frame_num].append((tid, bbox))

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info(f"Video: {total} frames. Seeking {len(needed_frames)} frames.")

    crops: dict[int, list[torch.Tensor]] = defaultdict(list)
    sorted_frames = sorted(needed_frames.keys())

    for frame_num in sorted_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ok, img = cap.read()
        if not ok:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        for tid, bbox in needed_frames[frame_num]:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            crop = img_rgb[y1:y2, x1:x2]
            try:
                crops[tid].append(transform(crop))
            except Exception:
                pass

    cap.release()
    log.info(f"Extracted crops: { {tid: len(v) for tid, v in crops.items()} }")
    return crops


@torch.no_grad()
def compute_embeddings(
    crops: dict[int, list[torch.Tensor]],
    model: nn.Module,
    device: torch.device,
) -> dict[int, np.ndarray]:
    embeddings: dict[int, np.ndarray] = {}
    for tid, tensors in crops.items():
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        emb = model(batch).cpu().numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # L2 normalize
        embeddings[tid] = emb
    return embeddings


def compute_similarity_stats(embeddings: dict[int, np.ndarray]) -> dict:
    intra_sims, inter_sims = [], []
    track_ids = list(embeddings.keys())

    for tid in track_ids:
        emb = embeddings[tid]
        if len(emb) < 2:
            continue
        sim = cosine_similarity(emb)
        upper = sim[np.triu_indices(len(sim), k=1)]
        intra_sims.extend(upper.tolist())

    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            sim = cosine_similarity(embeddings[track_ids[i]], embeddings[track_ids[j]])
            inter_sims.extend(sim.flatten().tolist())

    return {
        "intra_mean": float(np.mean(intra_sims)),
        "intra_std": float(np.std(intra_sims)),
        "inter_mean": float(np.mean(inter_sims)),
        "inter_std": float(np.std(inter_sims)),
        "separation": float(np.mean(intra_sims) - np.mean(inter_sims)),
    }


def plot_tsne(embeddings: dict[int, np.ndarray], out_path: Path) -> None:
    all_emb, all_labels = [], []
    for tid, emb in embeddings.items():
        all_emb.append(emb)
        all_labels.extend([tid] * len(emb))

    X = np.vstack(all_emb)
    labels = np.array(all_labels)
    unique_ids = list(embeddings.keys())

    log.info(f"Running t-SNE on {len(X)} embeddings...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(X) // 4), random_state=42)
    X2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
    for color, tid in zip(colors, unique_ids):
        mask = labels == tid
        ax.scatter(X2d[mask, 0], X2d[mask, 1], c=[color], label=f"track {tid}", s=20, alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    ax.set_title("t-SNE of player appearance embeddings (ResNet50 ImageNet)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info(f"t-SNE saved → {out_path}")


def plot_similarity_matrix(embeddings: dict[int, np.ndarray], out_path: Path) -> None:
    track_ids = list(embeddings.keys())
    mean_embs = np.vstack([emb.mean(axis=0) for emb in embeddings.values()])
    mean_embs /= np.linalg.norm(mean_embs, axis=1, keepdims=True)
    sim_matrix = cosine_similarity(mean_embs)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(track_ids)))
    ax.set_yticks(range(len(track_ids)))
    ax.set_xticklabels([f"t{tid}" for tid in track_ids], rotation=90, fontsize=7)
    ax.set_yticklabels([f"t{tid}" for tid in track_ids], fontsize=7)
    ax.set_title("Per-track mean embedding cosine similarity")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info(f"Similarity matrix saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_json", type=Path)
    parser.add_argument("raw_video", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("store/output/reid_test"))
    parser.add_argument("--min-track-frames", type=int, default=MIN_TRACK_FRAMES)
    parser.add_argument("--max-tracks", type=int, default=MAX_TRACKS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    log.info("Loading track index...")
    track_index = load_track_index(args.log_json)

    top_tracks = sorted(
        [tid for tid, entries in track_index.items() if len(entries) >= args.min_track_frames],
        key=lambda tid: len(track_index[tid]),
        reverse=True,
    )[: args.max_tracks]
    log.info(f"Top {len(top_tracks)} tracks (≥{args.min_track_frames} frames): {top_tracks}")

    transform = get_transform()
    model = build_model(device)
    log.info("Model loaded (ResNet50 ImageNet)")

    log.info("Extracting crops from video...")
    crops = sample_crops(args.raw_video, track_index, top_tracks, transform)

    log.info("Computing embeddings...")
    embeddings = compute_embeddings(crops, model, device)

    stats = compute_similarity_stats(embeddings)
    log.info("=== Similarity stats ===")
    log.info(f"  Intra-track (same player):  {stats['intra_mean']:.3f} ± {stats['intra_std']:.3f}")
    log.info(f"  Inter-track (diff players): {stats['inter_mean']:.3f} ± {stats['inter_std']:.3f}")
    log.info(f"  Separation (intra - inter): {stats['separation']:.3f}")
    if stats["separation"] > 0.05:
        log.info("  ✓ Promising — players are separable with a generic backbone")
    else:
        log.info("  ✗ Low separation — may need re-ID-specific model or more tuning")

    plot_tsne(embeddings, args.out_dir / "reid_tsne.png")
    plot_similarity_matrix(embeddings, args.out_dir / "reid_similarity_matrix.png")

    with open(args.out_dir / "reid_stats.json", "w") as f:
        json.dump({**stats, "tracks_analyzed": len(embeddings)}, f, indent=2)
    log.info(f"Stats saved → {args.out_dir / 'reid_stats.json'}")


if __name__ == "__main__":
    main()
