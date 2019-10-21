from te_compute import te_compute


import numpy as np

rand = np.random.RandomState(seed=23) # random_state for consistent results

# Create a bi-variate datasets containing:
# 1000; 10,000; 100,000; 1,000,000 rows

X_1000 = rand.randn(1000, 1).flatten()
Y_1000 = rand.randn(1000, 1).flatten()

TE = te_compute(X_1000, Y_1000, k=1, embedding=1, safetyCheck=False, GPU=False)

print(TE)
