import numpy as np

from PyIF import te_compute as te


def test_readme_cpu_example():
    """The README CPU example should run and return a finite number."""
    rng = np.random.RandomState(seed=23)
    x = rng.randn(1000)
    y = rng.randn(1000)

    result = te.te_compute(
        x,
        y,
        k=1,
        embedding=1,
        safetyCheck=True,
        GPU=False,
    )

    assert isinstance(result, (float, np.floating))
    assert np.isfinite(result)