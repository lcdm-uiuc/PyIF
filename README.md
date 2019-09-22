# transfer-entropy
An open source implementation to compute bi-variate Transfer Entropy.


TODO: Which License?
TODO: Examples.

To show how to use it. First install it from pip

TODO: Put the name it is called in pip here

Then `from te_compute import te_compute` to import the needed functions. Then all that is needed then is to call the `te_compute` function.

##Example
`from te_compute import te_compute

import numpy as np
rand = np.random.RandomState(seed=23)

    # Create a bi-variate datasets containing:
    # 1000; 10,000; 100,000; 1,000,000 rows

X_1000 = rand.randn(1000, 1).flatten()
Y_1000 = rand.randn(1000, 1).flatten()

TE = te_compute(X_1000, Y_1000, k=1, embedding=1, safetyCheck=True, GPU=False)
print(TE)
`

TODO: Pip installation?
