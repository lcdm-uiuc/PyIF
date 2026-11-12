# PyIF

An open source implementation to compute bi-variate Transfer Entropy.


PyIF provides the original `te_compute()` function and a new `estimate_te()` interface. The refactored interface currently supports KSG estimation.

## Installation

To install PyIF using pip run the following command:
```bash
pip install PyIF
```

To install this development branch, run the following command from the repository directory:
```bash
pip install -e .
```

## Legacy example

``` python
from PyIF import te_compute as te
import numpy as np
rand = np.random.RandomState(seed=23)

X_1000 = rand.randn(1000, 1).flatten()
Y_1000 = rand.randn(1000, 1).flatten()

TE = te.te_compute(X_1000, Y_1000, k=1, embedding=1, safetyCheck=True, GPU=False)

print(TE)
```

## Refactored interface example

```python
import numpy as np
from PyIF import estimate_te
from PyIF.config import KSGConfig

rng = np.random.default_rng(23)
source = rng.normal(size=1000)
target = rng.normal(size=1000)

result = estimate_te(
    source=source,
    target=target,
    config=KSGConfig(k_neighbors=2),
)

print(result.raw_te)
```

The estimate is from `source` to `target`. The returned
`TEResult` also contains resolved settings, tie counts,
input hashes, and software provenance.

## Arguments

The legacy `te_compute(X, Y)` function accepts NumPy arrays with shape `(N,)` or `(N, 1)`. It estimates transfer entropy from `Y` (source) to `X` (target). The following arguments are optional:

- `k`: controls the number of neighbors used in KD-tree queries
- `embedding`: controls the number of past observations of `X` (the target series) used in the calculation. The current implementation uses only one lag of `Y` (the source series), regardless of the embedding value. `te_compute(X, Y)` estimates transfer entropy from Y to X.
- `GPU`: a boolean argument that indicates if CUDA compatible GPUs should be used to estimate transfer entropy instead of your computer's CPUs.
-  `safetyCheck`: a boolean argument can be used to check for duplicates rows in your dataset.

## Estimator architecture

PyIF uses a shared `estimate_te()` interface with estimator-specific configuration classes. The `PyIF.estimators` registry maps estimator names to their configuration types and calculation functions.

Each estimator receives a `TEContext` containing the source, target, and requested history settings. The context can construct aligned future observations, target histories, and source histories using `context.histories()`. An estimator returns a transfer entropy estimate in nats, which the public interface wraps in `TEResult` with resolved settings, input hashes, tie counts, and software provenance.

KSG is currently the only implemented production estimator. It retains its existing computational path. Additional estimators can use the shared histories while implementing their own numerical methods and tie handling.

Conditional TE, surrogate inference, and additional estimator implementations remain planned features.

`tested-commit.txt` records the original upstream baseline used
for comparison, not the current refactor revision. The refactored
package records its own source Git provenance.

## How to Cite PyIF

K. M. Ikegwu, J. Trauger, J. McMullin and R. J. Brunner, "PyIF: A Fast and Light Weight Implementation to Estimate Bivariate Transfer Entropy for Big Data," 2020 SoutheastCon, Raleigh, NC, USA, 2020, pp. 1-6, doi: 10.1109/SoutheastCon44009.2020.9249650.