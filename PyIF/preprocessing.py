
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedTarget:
    xk: np.ndarray
    k: np.ndarray
    embedding: int


@dataclass(frozen=True)
class PreparedSource:
    lag: np.ndarray
    embedding: int


def prepare_target(target, embedding=1):
    """Prepare target-only spaces for the legacy KSG estimator."""
    target = np.asarray(target)

    if target.ndim != 1:
        raise ValueError("target must be one-dimensional")

    if type(embedding) is not int or not 1 <= embedding < len(target):
        raise ValueError("embedding must be between 1 and len(target)-1")

    n = len(target) - embedding

    xk = np.empty((n, embedding + 1), dtype=np.float64)

    for i in range(embedding + 1):
        j = embedding - i
        xk[:, i] = target[j:j + n]

    k = np.ascontiguousarray(xk[:, 1:])

    return PreparedTarget(xk=xk, k=k, embedding=embedding)


def prepare_source(source, embedding=1):
    """Prepare the source lag aligned with target history."""
    source = np.asarray(source)

    if source.ndim != 1:
        raise ValueError("source must be one-dimensional")

    if type(embedding) is not int or not 1 <= embedding < len(source):
        raise ValueError("embedding must be between 1 and len(source)-1")

    lag = np.array(
        source[embedding - 1:-1],
        dtype=np.float64,
        copy=True
    )

    return PreparedSource(lag=lag, embedding=embedding)


def combine_spaces(target, source):
    """Combine prepared series into the four legacy KSG spaces."""
    if target.embedding != source.embedding:
        raise ValueError("Embedding settings must match")

    if len(target.xk) != len(source.lag):
        raise ValueError("Prepared series must have matching lengths")

    xky = np.column_stack((target.xk, source.lag))
    ky = np.column_stack((target.k, source.lag))

    return xky, ky, target.xk, target.k, len(target.xk)
