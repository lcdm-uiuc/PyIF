
import numpy as np
from sklearn.neighbors import KDTree
from . import CPU_TE as cte
from numbers import Integral

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
        self._sorted_targets = {}

    
    def _check_embedding(self, embedding):
        if (
            isinstance(embedding, bool)
            or not isinstance(embedding, Integral)
            or not 1 <= embedding < self._length
        ):
            raise ValueError("Invalid embedding")

        return int(embedding)


    def get_target(self, name, embedding=1):
        """Return cached target histories."""
        embedding = self._check_embedding(embedding)
        key = (name, embedding)

        if key not in self._targets:
            prepared = prepare_target(self._series[name], embedding)

            prepared.xk.setflags(write=False)
            prepared.k.setflags(write=False)

            self._targets[key] = prepared

        return self._targets[key]

    def get_source(self, name, embedding=1):
        """Return cached aligned source observations."""
        embedding = self._check_embedding(embedding)
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
    

    def get_sorted_target(self, name, embedding=1):
        """Return target observations sorted once per configuration."""
        embedding = self._check_embedding(embedding)
        key = (name, embedding)

        if key not in self._sorted_targets:
            # Matches the legacy loop: j = embedding ... len(X)-1.
            values = np.sort(self._series[name][embedding:])
            values.setflags(write=False)
            self._sorted_targets[key] = values

        return self._sorted_targets[key]


    def make_spaces(self, source_name, target_name, embedding=1):
        """Construct the four legacy KSG spaces from cached histories."""
        target = self.get_target(target_name, embedding)
        source = self.get_source(source_name, embedding)

        return combine_spaces(target, source)
    
    def estimate_ksg(
        self,
        source_name,
        target_name,
        *,
        k=1,
        embedding=1,
    ):
        """Calculate CPU KSG TE using cached target structures."""

        embedding = self._check_embedding(embedding)

        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k must be a positive integer")

        k = int(k)

        # Construct pair-specific spaces from cached histories.
        xky, ky, xk, history, n_points = self.make_spaces(
            source_name,
            target_name,
            embedding
        )

        if k >= n_points:
            raise ValueError(
                "k must be less than the number of embedded rows"
            )

        # These trees depend on the particular source-target pair.
        xky_tree = KDTree(xky, metric="chebyshev")
        ky_tree = KDTree(ky, metric="chebyshev")

        # These trees depend only on the target and are reusable.
        xk_tree, history_tree = self.get_target_trees(
            target_name,
            embedding
        )

        # Call the original CPU estimator without changing its math.
        te = cte.compute(
            xky_tree,
            ky_tree,
            xk_tree,
            history_tree,
            xky,
            ky,
            xk,
            history,
            n_points,
            self._series[target_name],
            embedding=embedding,
            k=k,
            sorted_target=self.get_sorted_target(target_name, embedding),
        )

        return float(te)