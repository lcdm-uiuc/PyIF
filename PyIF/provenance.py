
from pathlib import Path
import subprocess


def _run_git(root, *args):
    """Run a Git command, returning None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def get_git_provenance():
    """Return (commit hash, working-tree dirty status)."""
    root = Path(__file__).resolve().parent.parent

    # Ensure this package belongs to the detected repository.
    git_root = _run_git(root, "rev-parse", "--show-toplevel")

    if git_root is None or Path(git_root).resolve() != root:
        return None, None

    revision = _run_git(root, "rev-parse", "HEAD")

    status = _run_git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=normal",
    )

    dirty = None if status is None else bool(status)

    return revision, dirty
