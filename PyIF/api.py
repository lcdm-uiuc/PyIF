
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from numbers import Integral

import numpy as np

from . import te_compute as legacy
from .config import KSGConfig
from .result import TEResult
from .provenance import get_git_provenance


def _data_hash(array):
    """Fingerprint input data for reproducibility."""
    data = np.ascontiguousarray(array)
    header = f"{data.shape}|{data.dtype.str}".encode()
    return sha256(header + b"\0" + data.tobytes()).hexdigest()


def estimate_te(
    source,
    target,
    *,
    estimator="ksg",
    target_history=1,
    source_history=1,
    delay=1,
    conditioning=None,
    config=None,
):
    """Estimate transfer entropy from source to target."""

    if estimator != "ksg":
        raise NotImplementedError(
            f"Estimator '{estimator}' is not implemented yet"
        )

    for name, value in (
        ("target_history", target_history),
        ("source_history", source_history),
        ("delay", delay),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    if source_history != 1 or delay != 1:
        raise NotImplementedError(
            "The legacy KSG estimator supports only "
            "source_history=1 and delay=1"
        )

    if conditioning is not None:
        raise NotImplementedError(
            "Conditional transfer entropy is not implemented yet"
        )

    if config is None:
        config = KSGConfig()

    if not isinstance(config, KSGConfig):
        raise TypeError("KSG estimation requires a KSGConfig")

    if not isinstance(source, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("source and target must be NumPy arrays")

    raw_te = legacy.te_compute(
        X=target,
        Y=source,
        k=config.k_neighbors,
        embedding=target_history,
        safetyCheck=config.safety_check,
        GPU=(config.backend == "gpu"),
    )

    if raw_te is None:
        raise ValueError("The legacy estimator did not return an estimate")

    settings = {
        "estimator": estimator,
        "target_history": int(target_history),
        "source_history": int(source_history),
        "delay": int(delay),
        "conditioning": None,
        "config": asdict(config),
        "inference_method": "none",
        "n_surrogates": 0,
        "random_seed": None,
    }

    code_revision, working_tree_dirty = get_git_provenance()

    return TEResult(
        raw_te=float(raw_te),
        units="nats",
        n_input=len(target),
        n_history_rows=len(target) - target_history,
        resolved_settings=settings,
        software_version=version("PyIF"),
        data_hashes={
            "source": _data_hash(source),
            "target": _data_hash(target),
        },
        code_revision=code_revision,
        working_tree_dirty=working_tree_dirty,
        random_seed=None,
    )
