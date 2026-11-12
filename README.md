# PyIF

An open source implementation to compute bi-variate Transfer Entropy.


Then `from PyIF import te_compute` to import the needed functions. Then all that is needed then is to call the `te_compute` function.

## Installation

To install PyIF using pip run the following command:
```bash
pip install PyIF
```

To install a the development release of PyIF run the following command:
```bash
pip install -e .
```

## Example

``` python
from PyIF import te_compute as te
import numpy as np
rand = np.random.RandomState(seed=23)

X_1000 = rand.randn(1000, 1).flatten()
Y_1000 = rand.randn(1000, 1).flatten()

TE = te.te_compute(X_1000, Y_1000, k=1, embedding=1, safetyCheck=True, GPU=False)

print(TE)
```

## Arguments

PyIF has 2 required arguments `X` and `Y` which should be numpy arrays with dimensions of N x 1. The following arguments are optional:

- `k`: controls the number of neighbors used in KD-tree queries
- `embedding`: controls the number of past observations of `X` (the target series) used in the calculation. The current implementation uses only one lag of `Y` (the source series), regardless of the embedding value. `te_compute(X, Y)` estimates transfer entropy from Y to X.
- `GPU`: a boolean argument that indicates if CUDA compatible GPUs should be used to estimate transfer entropy instead of your computer's CPUs.
-  `safetyCheck`: a boolean argument can be used to check for duplicates rows in your dataset.

## Estimator architecture

PyIF uses a shared `estimate_te()` interface with estimator-specific configuration classes. The `PyIF.estimators` registry maps estimator names to their configuration types and calculation functions.

Each estimator receives a `TEContext` containing the source, target, and requested history settings. The context can construct aligned future observations, target histories, and source histories using `context.histories()`. An estimator returns a transfer entropy estimate in nats, which the public interface wraps in `TEResult` with resolved settings, input hashes, tie counts, and software provenance.

KSG is currently the only implemented production estimator. It retains its existing computational path. Additional estimators can use the shared histories while implementing their own numerical methods and tie handling.

Conditional TE, surrogate inference, and additional estimator implementations remain planned features.

## How to Cite PyIF

K. M. Ikegwu, J. Trauger, J. McMullin and R. J. Brunner, "PyIF: A Fast and Light Weight Implementation to Estimate Bivariate Transfer Entropy for Big Data," 2020 SoutheastCon, Raleigh, NC, USA, 2020, pp. 1-6, doi: 10.1109/SoutheastCon44009.2020.9249650.