
import numpy as np
from sklearn.neighbors import KDTree

from .preprocessing import (
    prepare_target,
    prepare_source,
    combine_spaces,
)


class TEWorkspace:
    """Store aligned series and reuse KSG preprocessing across pairs."""

    def __init__(self, series, window_id=None):
        if not series:
            raise ValueError("At least one series is required")

        self.window_id = window_id
        self._series = {}
        self._length = None

        # Take private snapshots of the input series.
        for name, values in series.items():
            arr = np.asarray(values)

            if arr.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")

            if not np.issubdtype(arr.dtype, np.number) or not np.isrealobj(arr):
                raise TypeError(f"{name} must contain real numeric values")

            data = np.array(arr, dtype=np.float64, copy=True)

            if len(data) < 2 or not np.all(np.isfinite(data)):
                raise ValueError(f"{name} has invalid observations")

            if self._length is None:
                self._length = len(data)
            elif len(data) != self._length:
                raise ValueError("All series must have equal lengths")

            data.setflags(write=False)
            self._series[name] = data

        self._targets = {}
        self._sources = {}
        self._target_trees = {}

    def _check_embedding(self, embedding):
        if type(embedding) is not int or not 1 <= embedding < self._length:
            raise ValueError("Invalid embedding")

    def get_target(self, name, embedding=1):
        """Return cached target histories."""
        self._check_embedding(embedding)
        key = (name, embedding)

        if key not in self._targets:
            prepared = prepare_target(self._series[name], embedding)

            prepared.xk.setflags(write=False)
            prepared.k.setflags(write=False)

            self._targets[key] = prepared

        return self._targets[key]

    def get_source(self, name, embedding=1):
        """Return cached aligned source observations."""
        self._check_embedding(embedding)
        key = (name, embedding)

        if key not in self._sources:
            prepared = prepare_source(self._series[name], embedding)

            prepared.lag.setflags(write=False)
            self._sources[key] = prepared

        return self._sources[key]

    def get_target_trees(self, name, embedding=1):
        """Build target-only KD-trees once and reuse them."""
        prepared = self.get_target(name, embedding)
        key = (name, embedding)

        if key not in self._target_trees:
            xk_tree = KDTree(prepared.xk, metric="chebyshev")
            k_tree = KDTree(prepared.k, metric="chebyshev")

            self._target_trees[key] = (xk_tree, k_tree)

        return self._target_trees[key]

    def make_spaces(self, source_name, target_name, embedding=1):
        """Construct the four legacy KSG spaces from cached histories."""
        target = self.get_target(target_name, embedding)
        source = self.get_source(source_name, embedding)

        return combine_spaces(target, source)
