
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from numbers import Integral

import numpy as np

from .estimators import ESTIMATORS, TEContext
from .result import TEResult
from .provenance import get_git_provenance
from .preprocessing import count_tied_observations


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
    
    spec = ESTIMATORS.get(estimator)

    if spec is None:
        raise NotImplementedError(
            f"Estimator '{estimator}' is not implemented yet"
        )

    for name, value in (
        ("target_history", target_history),
        ("source_history", source_history),
        ("delay", delay),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")

    if conditioning is not None:
        raise NotImplementedError(
            "Conditional transfer entropy is not implemented yet"
        )

    if config is None:
        config = spec.config_type()

    if not isinstance(config, spec.config_type):
        raise TypeError(
            f"{estimator} estimation requires "
            f"{spec.config_type.__name__}"
        )

    if not isinstance(source, np.ndarray) or not isinstance(
        target, np.ndarray
    ):
        raise TypeError("source and target must be NumPy arrays")

    context = TEContext(
        source=source,
        target=target,
        target_history=int(target_history),
        source_history=int(source_history),
        delay=int(delay),
    )

    raw_te = spec.compute(context, config)

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
        n_history_rows=len(target) - context.first_row,
        resolved_settings=settings,
        software_version=version("PyIF"),
        data_hashes={
            "source": _data_hash(source),
            "target": _data_hash(target),
        },
        code_revision=code_revision,
        working_tree_dirty=working_tree_dirty,
        random_seed=None,
        tie_counts={
            "source": count_tied_observations(source),
            "target": count_tied_observations(target),
        },
    )
