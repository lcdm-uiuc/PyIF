"""Loss-checked conversion to float64 and common binary scaling."""
import numpy as np

def checked_series(values, name="series"):
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional or have shape (N, 1)")
    if not np.issubdtype(arr.dtype, np.number) or not np.isrealobj(arr):
        raise TypeError(f"{name} must contain real numeric values")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    info = {"integer_offset": None}
    with np.errstate(over="ignore", invalid="ignore"):
        out = arr.astype(np.float64, copy=True)
    if np.issubdtype(arr.dtype, np.integer):
        # Compare Python integers, avoiding a second lossy NumPy coercion.
        if any(int(a) != int(b) for a, b in zip(arr, out)):
            offset = int(arr.min())
            differences = [int(a) - offset for a in arr]
            if max(differences) > 2**53:
                raise ValueError(f"{name}: integer range cannot be preserved exactly in float64")
            out = np.asarray(differences, dtype=np.float64)
            info["integer_offset"] = str(offset)
    elif not np.isfinite(out).all() or not np.array_equal(out.astype(arr.dtype), arr):
        raise ValueError(f"{name}: conversion to float64 loses precision or overflows")
    return out, info


def prepare_collection(series):
    """Translate integers exactly if needed; scale ALL coordinates together.

    Scaling uses a power of two only outside the ordinary magnitude range.
    It preserves relative scales, unlike separately standardizing each series.
    Reject any scaling that cannot be reversed exactly in floating point.
    """
    prepared, details = {}, {}
    for name, values in series.items():
        prepared[name], details[name] = checked_series(values, name)
    maximum = max((float(np.max(np.abs(a))) if a.size else 0.
                   for a in prepared.values()), default=0.)
    exponent = 0
    if maximum and (maximum > 2.**400 or maximum < 2.**-400):
        exponent = int(np.frexp(maximum)[1])
        for name, values in prepared.items():
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                scaled = np.ldexp(values, -exponent)
                restored = np.ldexp(scaled, exponent)
            if not np.isfinite(scaled).all() or not np.array_equal(restored, values):
                raise ValueError("Common scaling loses information; dynamic range is too wide")
            prepared[name] = scaled
    return prepared, {"series": details, "common_binary_exponent": exponent}
