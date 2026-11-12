
import runpy
import subprocess
from pathlib import Path

import setuptools
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


ROOT = Path(__file__).resolve().parent


def _git(*args):
    """Run Git in the source directory if available."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    return result.stdout.strip() if result.returncode == 0 else None


def _source_provenance():
    """Get Git metadata or metadata retained in a source archive."""
    git_root = _git("rev-parse", "--show-toplevel")

    if git_root is not None and Path(git_root).resolve() == ROOT:
        commit = _git("rev-parse", "HEAD")
        status = _git(
            "status", "--porcelain", "--untracked-files=normal"
        )

        if commit:
            return commit, None if status is None else bool(status)

    # Building from a source distribution without .git:
    info_file = ROOT / "PyIF" / "_build_info.py"

    if info_file.is_file():
        info = runpy.run_path(str(info_file))
        return info.get("SOURCE_COMMIT"), info.get("SOURCE_DIRTY")

    return None, None


def _build_info_content():
    commit, dirty = _source_provenance()

    return (
        '"""Source provenance recorded at build time."""\n'
        f"SOURCE_COMMIT = {commit!r}\n"
        f"SOURCE_DIRTY = {dirty!r}\n"
    )


class BuildPy(build_py):
    """Embed provenance inside the built package."""

    def run(self):
        super().run()

        destination = Path(self.build_lib) / "PyIF" / "_build_info.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            _build_info_content(), encoding="utf-8"
        )

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode=include_bytecode)
        return outputs + [
            str(Path(self.build_lib) / "PyIF" / "_build_info.py")
        ]


class SourceDist(sdist):
    """Preserve provenance when packaging source without .git."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)

        destination = Path(base_dir) / "PyIF" / "_build_info.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            _build_info_content(), encoding="utf-8"
        )


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='PyIF',  
     version='0.2.0.dev0',
     python_requires=">=3.10",
     author="Kelechi Ikegwu, Jacob Trauger, Robert Brunner",
     author_email="ikegwu2@illinois.edu, jtt2@illinois.edu, bigdog@illinois.edu",
     description="An open source implementation to compute bi-variate Transfer Entropy.",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/lcdm-uiuc/PyIF",
     packages=setuptools.find_packages(),
     install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "numba",
     ],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: University of Illinois/NCSA Open Source License",
         "Operating System :: OS Independent",
     ],
     cmdclass={"build_py": BuildPy, "sdist": SourceDist},
 )
