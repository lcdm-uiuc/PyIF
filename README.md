# PyTE

An open source implementation to compute bi-variate Transfer Entropy.


Then `from PyTE import te_compute` to import the needed functions. Then all that is needed then is to call the `te_compute` function.

## Installation

To install from pip all that is needed is run the line `pip install PyTE`.


To install a the development release of Py-TE run the following command
```bash
pip install -e .
```

## Example

``` python
from PyTE import te_compute

import numpy as np
rand = np.random.RandomState(seed=23)

X_1000 = rand.randn(1000, 1).flatten()
Y_1000 = rand.randn(1000, 1).flatten()

TE = te_compute(X_1000, Y_1000, k=1, embedding=1, safetyCheck=True, GPU=False)

print(TE)
```
