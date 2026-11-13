
from dataclasses import dataclass

import numpy as np

from . import te_compute as legacy
from .config import KSGConfig


@dataclass(frozen=True)
class TEContext:
    """Inputs and requested history settings for an estimator."""

    source: np.ndarray
    target: np.ndarray
    target_history: int
    source_history: int
    delay: int

    @property
    def first_row(self):
        return max(
            self.target_history,
            self.delay + self.source_history - 1,
        )

    def histories(self):
        """Return aligned future, target past, and source past."""
        if self.source.ndim != 1 or self.target.ndim != 1:
            raise ValueError("History preparation requires 1D series")

        if len(self.source) != len(self.target):
            raise ValueError("Source and target lengths must match")

        n = len(self.target)
        start = self.first_row

        if start >= n:
            raise ValueError("Not enough observations for histories")

        rows = np.arange(start, n)

        future = self.target[rows]

        target_past = np.column_stack([
            self.target[rows - lag]
            for lag in range(1, self.target_history + 1)
        ])

        source_past = np.column_stack([
            self.source[rows - self.delay - lag]
            for lag in range(self.source_history)
        ])

        return future, target_past, source_past


@dataclass(frozen=True)
class EstimatorSpec:
    """Contract for one estimator implementation."""

    config_type: type
    compute: object


def _compute_ksg(context, config):
    """Preserve the existing legacy KSG calculation."""

    if context.source_history != 1 or context.delay != 1:
        raise NotImplementedError(
            "The legacy KSG estimator supports only "
            "source_history=1 and delay=1"
        )

    result = legacy.te_compute(
        X=context.target,
        Y=context.source,
        k=config.k_neighbors,
        embedding=context.target_history,
        safetyCheck=config.safety_check,
        GPU=(config.backend == "gpu"),
    )

    if result is None:
        raise ValueError(
            "The legacy estimator did not return an estimate"
        )

    return float(result)


ESTIMATORS = {
    "ksg": EstimatorSpec(
        config_type=KSGConfig,
        compute=_compute_ksg,
    ),
}
