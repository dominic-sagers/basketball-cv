"""
weekly_benchmark.py — summarise store/benchmarks.jsonl for the past week.

Prints a table grouped by machine/GPU showing avg, min, and max FPS across
all runs recorded in the last 7 days, plus a per-run breakdown.

Usage:
    python scripts/weekly_benchmark.py            # last 7 days
    python scripts/weekly_benchmark.py --days 30  # last 30 days
    python scripts/weekly_benchmark.py --all      # all time
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BENCHMARK_LOG = Path("store/benchmarks.jsonl")


def _load(days: int | None) -> list[dict]:
    if not BENCHMARK_LOG.exists():
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat() if days else None
    records = []
    for line in BENCHMARK_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if cutoff and r.get("timestamp", "") < cutoff:
                continue
            records.append(r)
        except json.JSONDecodeError:
            continue
    return records


def _table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly FPS benchmark summary")
    parser.add_argument("--days", type=int, default=7, metavar="N", help="Look back N days (default: 7)")
    parser.add_argument("--all", action="store_true", help="Show all-time records")
    args = parser.parse_args()

    days = None if args.all else args.days
    records = _load(days)

    if not records:
        period = "all time" if days is None else f"the last {days} days"
        print(f"No benchmark records found in {period}.")
        print(f"  Log file: {BENCHMARK_LOG.resolve()}")
        return

    period_label = "all time" if days is None else f"last {days} days"
    print(f"\n=== FPS Benchmark Summary ({period_label}) ===\n")

    # Per-machine summary
    by_machine: dict[str, list[float]] = defaultdict(list)
    for r in records:
        key = f"{r.get('hostname', '?')}  [{r.get('gpu', '?')}  {r.get('gpu_vram_gb', '?')}GB]"
        by_machine[key].append(r["avg_fps"])

    summary_rows = []
    for machine, fps_list in sorted(by_machine.items()):
        summary_rows.append([
            machine,
            str(len(fps_list)),
            f"{sum(fps_list)/len(fps_list):.1f}",
            f"{min(fps_list):.1f}",
            f"{max(fps_list):.1f}",
        ])
    print(_table(summary_rows, ["Machine", "Runs", "Avg FPS", "Min FPS", "Max FPS"]))

    # Per-run breakdown
    print(f"\n--- Per-run detail ---\n")
    detail_rows = []
    for r in sorted(records, key=lambda x: x.get("timestamp", "")):
        detail_rows.append([
            r.get("timestamp", "?")[:16],
            r.get("hostname", "?"),
            r.get("gpu", "?"),
            r.get("video") or r.get("source") or "?",
            f"{r.get('avg_fps', 0):.1f}",
            f"{r.get('total_frames', 0):,}",
            f"{r.get('duration_s', 0)/60:.0f}m",
            r.get("weights") or "?",
        ])
    print(_table(detail_rows, ["Timestamp", "Host", "GPU", "Video", "FPS", "Frames", "Time", "Weights"]))
    print()


if __name__ == "__main__":
    main()
