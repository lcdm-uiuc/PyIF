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
- `embedding`: controls how many lagged periods are used to estimate transfer entropy
- `GPU`: a boolean argument that indicates if CUDA compatible GPUs should be used to estimate transfer entropy instead of your computer's CPUs.
-  `safetyCheck`: a boolean argument can be used to check for duplicates rows in your dataset.
