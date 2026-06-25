"""
game_archive.py — archive raw full-game videos to store/footage/games/ and push to DVC.

Called automatically at the end of each game session. Can also be used manually:

    python src/game_archive.py --list           # show archived games
    python src/game_archive.py --push           # push latest store state to DVC remote
    python src/game_archive.py --replay <file>  # print the pipeline_test replay command

After a session, DVC archives are committed to git with:
    git add store.dvc && git commit -m "archive: game <run_id>"

DVC remote: configured in .dvc/config (currently Dominic's Tailscale drive).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

GAMES_DIR = Path("store/footage/games")


# ---------------------------------------------------------------------------
# Core archive logic
# ---------------------------------------------------------------------------

def archive_raw_game(
    raw_video_path: str,
    run_id: str,
    camera_team: str,
) -> str | None:
    """
    Copy a completed raw game video into store/footage/games/ for DVC archiving.

    Uses a hardlink when source and destination are on the same filesystem
    (zero disk cost, instant). Falls back to a byte-copy otherwise.

    Args:
        raw_video_path: Path to the concatenated raw game video.
        run_id:         Session timestamp string, e.g. "2026-06-29_194500".
        camera_team:    "A" or "B".

    Returns the archive path on success, None on failure.
    """
    src = Path(raw_video_path)
    if not src.exists():
        logger.warning("Raw video not found — skipping archive: %s", raw_video_path)
        return None

    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    dest = GAMES_DIR / f"{run_id}_cam{camera_team.upper()}.mp4"

    if dest.exists():
        logger.info("Archive already exists: %s", dest)
        return str(dest)

    try:
        dest.hardlink_to(src)
        logger.info("Archived (hardlink) %s → %s", src.name, dest)
    except OSError:
        shutil.copy2(str(src), str(dest))
        logger.info("Archived (copy) %s → %s", src.name, dest)
    except Exception as exc:
        logger.error("Archive failed %s → %s: %s", src, dest, exc)
        return None

    return str(dest)


def dvc_push(project_root: str = ".") -> bool:
    """
    Update store.dvc and push newly archived game files to the DVC remote.

    Runs `dvc add store` (re-hashes the store directory to pick up new files),
    then `dvc push`. Logs the `git add store.dvc` command the user needs to
    commit so the updated pointer is tracked in git.

    Returns True on success. Errors are logged but never raised.
    """
    dvc = _find_dvc(project_root)
    if not dvc:
        logger.warning(
            "dvc binary not found — skipping push. "
            "Activate .venv or install dvc (pip install dvc[ssh])."
        )
        return False

    try:
        logger.info("Updating store.dvc (dvc add store) — may take a moment for large archives …")
        r = subprocess.run(
            [dvc, "add", "store"],
            capture_output=True, text=True, cwd=project_root,
        )
        if r.returncode != 0:
            logger.error("dvc add store failed:\n%s", (r.stderr or r.stdout)[-600:])
            return False

        logger.info("Pushing to DVC remote …")
        r = subprocess.run(
            [dvc, "push"],
            capture_output=True, text=True, cwd=project_root,
        )
        if r.returncode != 0:
            logger.error("dvc push failed:\n%s", (r.stderr or r.stdout)[-600:])
            return False

        logger.info("DVC push complete.")
        logger.info(
            "Track in git:  git add store.dvc && git commit -m 'archive: game session'"
        )
        return True

    except Exception as exc:
        logger.error("DVC push error: %s", exc)
        return False


def list_games() -> list[Path]:
    """Return archived game files in store/footage/games/, newest first."""
    if not GAMES_DIR.exists():
        return []
    return sorted(GAMES_DIR.glob("*.mp4"), reverse=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_dvc(project_root: str) -> str | None:
    """Prefer .venv/bin/dvc (matches DVC remote memory), fall back to PATH."""
    venv_dvc = Path(project_root) / ".venv" / "bin" / "dvc"
    if venv_dvc.exists():
        return str(venv_dvc)
    return shutil.which("dvc")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Archive raw game videos and manage DVC remote"
    )
    parser.add_argument("--list",   action="store_true", help="List archived games")
    parser.add_argument("--push",   action="store_true", help="Push store to DVC remote")
    parser.add_argument("--replay", metavar="FILE",       help="Print replay command for a game file")
    args = parser.parse_args()

    if args.list:
        games = list_games()
        if not games:
            print("No archived games in store/footage/games/")
        else:
            print(f"{'File':<55}  {'Size':>8}")
            print("-" * 66)
            for g in games:
                size_mb = g.stat().st_size / 1024 / 1024
                print(f"  {g.name:<53}  {size_mb:>6.0f} MB")
        return

    if args.push:
        ok = dvc_push()
        sys.exit(0 if ok else 1)

    if args.replay:
        p = Path(args.replay)
        print(f"python src/pipeline_test.py --file {p} --save-log store/output/{p.stem}_replay_log.json")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
