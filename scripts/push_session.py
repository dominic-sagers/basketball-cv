"""
Post-session DVC push script.
Run this after every game session to push new clips to the server
so they appear on the clip viewer site.

Usage:
    python scripts/push_session.py
    python scripts/push_session.py --message "Monday game Apr 29"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, **kwargs)
    if result.returncode != 0:
        print(f"  [FAILED] exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Push session outputs to DVC remote and update store.dvc")
    parser.add_argument("--message", "-m", default=None, help="Git commit message (default: auto-generated)")
    parser.add_argument("--no-git-push", action="store_true", help="Skip git push (DVC push only)")
    args = parser.parse_args()

    msg = args.message or f"session {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    print("\n=== Post-session push ===\n")

    # 1. Re-index store/ and update store.dvc
    print("[1/4] Indexing store/ with DVC...")
    run(["dvc", "add", "store"])

    # 2. Commit the updated store.dvc pointer
    print(f"\n[2/4] Committing store.dvc ({msg})...")
    run(["git", "add", "store.dvc"])
    # Check if there's actually a change to commit
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=REPO_ROOT
    )
    if diff.returncode == 0:
        print("  store.dvc unchanged — nothing to commit")
    else:
        run(["git", "commit", "-m", f"dvc: {msg}"])

    # 3. Push file contents to the DVC remote (server cache)
    print("\n[3/4] Pushing files to DVC remote (server cache)...")
    run(["dvc", "push"])

    # 4. Push the updated store.dvc pointer to git remote
    if not args.no_git_push:
        print("\n[4/4] Pushing store.dvc to git remote...")
        run(["git", "push"])
    else:
        print("\n[4/4] Skipping git push (--no-git-push)")

    print("\n=== Done ===")
    print("  New clips will appear on the site once the server pulls the updated store.dvc.")
    print("  On the site: click 'Refresh' or the server can run: git -C /mnt/nvme1/personal_projects/basketball-cv pull")


if __name__ == "__main__":
    main()
