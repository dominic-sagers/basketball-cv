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
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("store/output")

# Tracks the one background `dvc push` that may be running, so a local `dvc add`
# (which must win immediately — see dvc_add_local) can terminate it rather than
# block behind it. DVC's own lock file makes `add` and `push` mutually exclusive;
# confirmed 2026-07-02 when a running push made a concurrent add fail outright
# with "Unable to acquire lock" instead of queueing.
_push_lock = threading.Lock()
_push_proc: subprocess.Popen | None = None

# Push state, exposed via get_push_status() (and the server's /api/v1/push-status
# route) so the archive-server operator/app can tell whether data reaching DONE
# locally has *also* made it to the remote — the push itself is only visible in
# server logs otherwise, which is invisible from the Android side.
PUSH_NEVER_RUN = "NEVER_RUN"
PUSH_PENDING = "PENDING"
PUSH_PUSHING = "PUSHING"
PUSH_PUSHED = "PUSHED"
PUSH_FAILED = "FAILED"


@dataclass
class PushStatus:
    state: str = PUSH_NEVER_RUN
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


_push_status = PushStatus()


def get_push_status() -> dict[str, str | None]:
    """
    Current state of the background dvc push to the remote.

    This is global, not per-session — dvc push covers the whole local DVC
    cache in one shot, not one session's chunks at a time. A session reaching
    DONE only guarantees its data is safe in the local cache on this machine;
    this is how to check whether that data has also reached the remote.
    """
    with _push_lock:
        return asdict(_push_status)


# ---------------------------------------------------------------------------
# DVC add (local, fast — gates archive completion)
# ---------------------------------------------------------------------------

def dvc_add_local(project_root: str = ".") -> bool:
    """
    Re-hash store/ into the local DVC cache on this machine (bb-1).

    This is the part that must complete for a session to count as archived —
    it's disk-bound, not network-bound, so it should take seconds regardless
    of remote push speed. If a background dvc_push_background() is currently
    holding DVC's lock, it gets killed so this can proceed immediately; the
    push resumes (incrementally, nothing already sent is re-sent) on the next
    dvc_push_background() call.

    Returns True on success.
    """
    dvc = _find_dvc(project_root)
    if not dvc:
        logger.warning(
            "dvc not found — skipping add. "
            "Activate .venv or install dvc (pip install dvc[ssh])."
        )
        return False

    _kill_in_flight_push()

    try:
        logger.info("Updating store.dvc (dvc add store) …")
        r = subprocess.run(
            [dvc, "add", "store"],
            capture_output=True, text=True, cwd=project_root,
            timeout=300,
        )
        if r.returncode != 0:
            logger.error("dvc add store failed:\n%s", (r.stderr or r.stdout)[-600:])
            return False

        logger.info("dvc add complete — data is safe in the local cache on this machine.")
        logger.info(
            "Commit the updated pointer:  "
            "git add store.dvc && git commit -m 'archive: game session' && git push"
        )
        return True

    except subprocess.TimeoutExpired as exc:
        logger.error("dvc add timed out after %ss: %s", exc.timeout, exc.cmd)
        return False
    except Exception as exc:
        logger.error("dvc add error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# DVC push (remote, slow — best-effort, does not gate anything)
# ---------------------------------------------------------------------------

def dvc_push_background(project_root: str = ".") -> None:
    """
    Push the local DVC cache to the remote (Dominic's tailnet machine) in the
    background. Not timed — over the current relay-bound link this can take
    hours for a full game; that's fine, since nothing waits on it. Call this
    from a daemon thread after dvc_add_local() succeeds.

    If dvc_add_local() kills this mid-push, the next call resumes: DVC only
    sends objects the remote doesn't already have.
    """
    dvc = _find_dvc(project_root)
    if not dvc:
        return

    global _push_proc
    with _push_lock:
        if _push_proc is not None and _push_proc.poll() is None:
            logger.info("dvc push already in progress — this session's data will ride along.")
            return
        logger.info("Starting background dvc push to remote …")
        _push_status.state = PUSH_PUSHING
        _push_status.started_at = datetime.now().isoformat()
        _push_status.error = None
        _push_proc = subprocess.Popen(
            [dvc, "push"], cwd=project_root,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        proc = _push_proc

    out, _ = proc.communicate()
    with _push_lock:
        if _push_proc is proc:
            _push_proc = None
        if proc.returncode == 0:
            _push_status.state = PUSH_PUSHED
            _push_status.finished_at = datetime.now().isoformat()
        elif proc.returncode < 0:
            # Killed by _kill_in_flight_push() to let a newer dvc add proceed — a
            # fresh push covering this session's data starts right after that add,
            # so this isn't a failure, just queued behind the add.
            _push_status.state = PUSH_PENDING
        else:
            _push_status.state = PUSH_FAILED
            _push_status.finished_at = datetime.now().isoformat()
            _push_status.error = (out or "")[-600:]

    if proc.returncode == 0:
        logger.info("Background dvc push complete.")
    elif proc.returncode < 0:
        logger.info("Background dvc push interrupted (signal %d) — will resume next time.", -proc.returncode)
    else:
        logger.error("Background dvc push failed:\n%s", (out or "")[-600:])


def _kill_in_flight_push() -> None:
    """Terminate the background push if one is running, so a local add can proceed now."""
    global _push_proc
    with _push_lock:
        proc = _push_proc
        _push_proc = None
    if proc is not None and proc.poll() is None:
        logger.info("A background dvc push is running — stopping it so dvc add can proceed now.")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


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
        ok = dvc_add_local()
        if ok:
            dvc_push_background()
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
