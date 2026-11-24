"""Signed Mesner--Shalizi CMI, arXiv:1912.03387v1, equations 18--19.

Max norm; inclusive balls; counts exclude the query observation. The logarithmic
branch applies whenever the joint-ball count exceeds k, including ties at a
positive radius. Equation 20's final clipping is intentionally not applied.
Numeric discrete/continuous data only; no jitter or automatic standardization.
"""
from numbers import Integral
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree
from .numerics import prepare_collection

def conditional_mi(x, y, z, *, k=16, return_details=False):
    """Estimate I(x; y | z) for equal-length vectors or matrices of sample rows.

    The scalar result is signed, in nats. ``return_details`` exposes the counts
    and local estimates for reproducible tests; no inference is performed.
    A finite score is not a test of the paper's regularity/i.i.d. assumptions.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)
    arrays = {}
    for name, values in (("x", x), ("y", y), ("z", z)):
        data = np.asarray(values)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2 or data.shape[1] < 1:
            raise ValueError(f"{name} must have one or more columns of sample rows")
        arrays[name] = data
    n = len(arrays['x'])
    if any(len(a) != n for a in arrays.values()):
        raise ValueError("x, y and z must have equal sample counts")
    if k >= n:
        raise ValueError("k must be smaller than the number of history rows")
    columns = {f"{name}_{j}": values[:,j]
               for name, values in arrays.items() for j in range(values.shape[1])}
    prepared, numerical = prepare_collection(columns)
    x, y, z = [np.column_stack([prepared[f"{name}_{j}"]
                               for j in range(arrays[name].shape[1])])
               for name in ('x', 'y', 'z')]
    joint = np.column_stack([x, y, z])
    tree = KDTree(joint, metric='chebyshev')
    radius = tree.query(joint, k=k+1, return_distance=True)[0][:, -1]
    if not np.isfinite(radius).all():
        raise FloatingPointError("Non-finite MS neighbor distances")
    # Exact inclusive radius: no blanket one-ULP inflation, which can change ties.
    joint_count = tree.query_radius(joint, radius, count_only=True) - 1
    counts = []
    for points in (np.column_stack([x,z]), np.column_stack([y,z]), z):
        counts.append(KDTree(points, metric='chebyshev').query_radius(
            points, radius, count_only=True)-1)
    nxz, nyz, nz = counts
    if np.any(joint_count < k) or any(np.any(c < joint_count) for c in counts):
        raise FloatingPointError("Inconsistent MS inclusive neighbor counts")
    tied = joint_count > k
    local = digamma(joint_count)-digamma(nxz)-digamma(nyz)+digamma(nz)
    local[tied] = (np.log(joint_count[tied])-np.log(nxz[tied])
                   -np.log(nyz[tied])+np.log(nz[tied]))
    value = float(local.mean())
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite MS estimate")
    if return_details:
        return dict(value=value, local=local, radius=radius, joint_count=joint_count,
                    nxz=nxz, nyz=nyz, nz=nz, log_branch=tied,
                    numerical_preprocessing=numerical)
    return value
