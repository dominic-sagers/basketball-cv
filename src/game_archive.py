"""
game_archive.py — DVC push helper for post-session archiving.

Called automatically at the end of each session (by archiver.py and app.py).
Can also be used manually:

    python src/game_archive.py --list           # show archived sessions
    python src/game_archive.py --push           # push latest store state to DVC remote
    python src/game_archive.py --replay <file>  # print the detect_track_and_log replay command

After a session, commit the DVC pointer to git:
    git add store.dvc && git commit -m "archive: game <run_id>" && git push
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("store/output")


# ---------------------------------------------------------------------------
# DVC push
# ---------------------------------------------------------------------------

def dvc_push(project_root: str = ".") -> bool:
    """
    Re-hash store/ and push newly archived files to the DVC remote.

    Runs `dvc add store` (picks up any new files), then `dvc push`.
    Logs the git command needed to commit the updated store.dvc pointer.

    Returns True on success.
    """
    dvc = _find_dvc(project_root)
    if not dvc:
        logger.warning(
            "dvc not found — skipping push. "
            "Activate .venv or install dvc (pip install dvc[ssh])."
        )
        return False

    try:
        logger.info("Updating store.dvc (dvc add store) …")
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
            "Commit the updated pointer:  "
            "git add store.dvc && git commit -m 'archive: game session' && git push"
        )
        return True

    except Exception as exc:
        logger.error("DVC push error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Session listing
# ---------------------------------------------------------------------------

def list_sessions() -> list[Path]:
    """Return session directories under store/output/, newest first."""
    if not OUTPUT_DIR.exists():
        return []
    return sorted(
        [p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.name != "stream-chunks"],
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_dvc(project_root: str) -> str | None:
    """Prefer .venv/bin/dvc, fall back to PATH."""
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

    parser = argparse.ArgumentParser(description="DVC archive helper for basketball-cv")
    parser.add_argument("--list",   action="store_true", help="List archived sessions")
    parser.add_argument("--push",   action="store_true", help="Push store to DVC remote")
    parser.add_argument("--replay", metavar="FILE",      help="Print replay command for a raw game file")
    args = parser.parse_args()

    if args.list:
        sessions = list_sessions()
        if not sessions:
            print("No sessions in store/output/")
        else:
            for s in sessions:
                files = sorted(s.glob("*.mp4"))
                size_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
                print(f"  {s.name:<35}  {len(files)} file(s)  {size_mb:.0f} MB")
        return

    if args.push:
        ok = dvc_push()
        sys.exit(0 if ok else 1)

    if args.replay:
        p = Path(args.replay)
        run_id = p.parent.name
        print(
            f"python src/detect_track_and_log.py --file {p} "
            f"--save-log store/output/{run_id}/{p.stem}_replay_log.json"
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
