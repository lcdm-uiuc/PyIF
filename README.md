# PyTE

An open source implementation to compute bi-variate Transfer Entropy.


Then `from PyTE import te_compute` to import the needed functions. Then all that is needed then is to call the `te_compute` function.

## Installation

TODO: Pip installation. Put the name it is called in pip here


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
