
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
    """Return source commit and working-tree/build dirty status."""
    root = Path(__file__).resolve().parent.parent

    git_root = _run_git(root, "rev-parse", "--show-toplevel")

    if git_root is not None and Path(git_root).resolve() == root:
        revision = _run_git(root, "rev-parse", "HEAD")
        status = _run_git(
            root,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        )

        dirty = None if status is None else bool(status)
        return revision, dirty

    # Fall back to provenance embedded during packaging.
    try:
        from ._build_info import SOURCE_COMMIT, SOURCE_DIRTY
    except ImportError:
        return None, None

    return SOURCE_COMMIT, SOURCE_DIRTY

