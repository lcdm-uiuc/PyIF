
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_installed_wheel_retains_git_commit(tmp_path):
    """An installed wheel should retain its source revision."""

    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()

    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()

    subprocess.run(
        [
            sys.executable, "-m", "pip", "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir", str(wheel_dir),
            str(ROOT),
        ],
        check=True,
    )

    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1

    installed = tmp_path / "installed"

    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--no-deps",
            "--target", str(installed),
            str(wheels[0]),
        ],
        check=True,
    )

    outside = tmp_path / "outside"
    outside.mkdir()

    script = """
import sys
from pathlib import Path
import PyIF
from PyIF.provenance import get_git_provenance

installed = Path(sys.argv[1]).resolve()
expected = sys.argv[2]

assert Path(PyIF.__file__).resolve().is_relative_to(installed)

revision, dirty = get_git_provenance()

assert revision == expected
assert dirty is None or isinstance(dirty, bool)
"""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(installed)

    subprocess.run(
        [sys.executable, "-c", script, str(installed), expected],
        cwd=outside,
        env=env,
        check=True,
    )
