
import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from PyIF import estimate_te, te_compute
from PyIF.config import KSGConfig
from importlib.metadata import version
from dataclasses import dataclass
from PyIF.estimators import ESTIMATORS, EstimatorSpec


def test_new_api_matches_legacy():
    rng = np.random.default_rng(123)
    target = rng.normal(size=100)
    source = rng.normal(size=100)

    original = te_compute.te_compute(
        target, source, k=1, embedding=1,
        safetyCheck=False, GPU=False
    )

    result = estimate_te(source, target)

    assert result.raw_te == original
    assert result.units == "nats"
    assert result.n_input == 100
    assert result.n_history_rows == 99


def test_resolved_defaults():
    rng = np.random.default_rng(456)
    source = rng.normal(size=100)
    target = rng.normal(size=100)

    result = estimate_te(source, target)

    assert result.resolved_settings["estimator"] == "ksg"
    assert result.resolved_settings["target_history"] == 1
    assert result.resolved_settings["source_history"] == 1
    assert result.resolved_settings["delay"] == 1

    assert result.resolved_settings["config"] == {
        "k_neighbors": 1,
        "backend": "cpu",
        "safety_check": False,
    }

    assert result.inference.status == "not_computed"
    assert result.inference.corrected_te is None
    assert len(result.data_hashes["source"]) == 64


def test_unsupported_features():
    rng = np.random.default_rng(789)
    source = rng.normal(size=100)
    target = rng.normal(size=100)

    with pytest.raises(NotImplementedError):
        estimate_te(source, target, estimator="gaussian")

    with pytest.raises(NotImplementedError):
        estimate_te(source, target, source_history=2)

    with pytest.raises(NotImplementedError):
        estimate_te(source, target, conditioning={"market": source})


def test_invalid_config():
    with pytest.raises(ValueError):
        KSGConfig(k_neighbors=0)


def test_nondefault_ksg_settings_match_legacy():
    rng = np.random.default_rng(1234)
    target = rng.normal(size=200)
    source = rng.normal(size=200)

    # Existing PyIF implementation
    original = te_compute.te_compute(
        X=target,
        Y=source,
        k=2,
        embedding=2,
        safetyCheck=False,
        GPU=False
    )

    # New interface with equivalent settings
    result = estimate_te(
        source=source,
        target=target,
        estimator="ksg",
        target_history=2,
        config=KSGConfig(
            k_neighbors=2,
            backend="cpu",
            safety_check=False
        )
    )

    # Verify numerical equivalence
    assert result.raw_te == original

    # Verify that nondefault parameters were recorded
    assert result.resolved_settings["target_history"] == 2
    assert result.resolved_settings["config"]["k_neighbors"] == 2

    # Verify that defaults were also recorded
    assert result.resolved_settings["source_history"] == 1
    assert result.resolved_settings["delay"] == 1
    assert result.resolved_settings["config"]["backend"] == "cpu"
    assert result.resolved_settings["config"]["safety_check"] is False

    # Verify the reported history matrix size
    assert result.n_history_rows == 198


def test_result_reproducibility_metadata():
    rng = np.random.default_rng(12345)

    source = rng.normal(size=100)
    target = rng.normal(size=100)

    result = estimate_te(source, target)

    assert result.software_version == "0.2.0.dev0"
    assert result.software_version == version("PyIF")

    assert result.random_seed is None
    assert result.resolved_settings["random_seed"] is None

    # In a Git checkout, record a 40-character commit hash.
    if result.code_revision is not None:
        assert len(result.code_revision) == 40
        assert all(c in "0123456789abcdef"
                   for c in result.code_revision)

    # The dirty flag is either known or unavailable.
    assert result.working_tree_dirty is None or isinstance(
        result.working_tree_dirty, bool
    )


@pytest.mark.parametrize(
    "value",
    [1, 2, np.int32(2), np.int64(2), np.uint64(2)]
)
def test_accepts_integer_types(value):
    config = KSGConfig(k_neighbors=value)

    assert config.k_neighbors == int(value)
    assert type(config.k_neighbors) is int


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, 1.5, "2", None]
)
def test_rejects_invalid_k(value):
    with pytest.raises(ValueError):
        KSGConfig(k_neighbors=value)


def test_result_reports_ties_with_safety_check_disabled():
    rng = np.random.default_rng(2026)

    source = rng.normal(size=200)
    target = rng.normal(size=200)

    # Three source observations belong to one tied group.
    source[1] = source[0]
    source[2] = source[0]

    # Two target observations belong to one tied group.
    target[1] = target[0]

    result = estimate_te(
        source=source,
        target=target,
        config=KSGConfig(
            k_neighbors=2,
            safety_check=False,
        ),
    )

    assert result.tie_counts == {
        "source": 3,
        "target": 2,
    }



def _generate_one_way_ar(n, coupling_x, coupling_y, seed=0):
    """One-way Gaussian AR(1) model from Matthew's validation framework."""
    rng = np.random.default_rng(seed)

    a = b = 0.5
    transition = np.array([
        [a, coupling_x],
        [coupling_y, b],
    ])

    # Match the stationary initialization in AR_system.py.
    covariance = solve_discrete_lyapunov(
        transition, np.eye(2)
    )

    x = np.zeros(n)
    y = np.zeros(n)

    x[0], y[0] = rng.multivariate_normal(
        [0, 0], covariance
    )

    for t in range(1, n):
        eps_x = rng.normal()
        eps_y = rng.normal()

        x[t] = (
            a * x[t - 1]
            + coupling_x * y[t - 1]
            + eps_x
        )
        y[t] = (
            b * y[t - 1]
            + coupling_y * x[t - 1]
            + eps_y
        )

    return x, y


@pytest.mark.parametrize(
    "coupling_x,coupling_y",
    [
        (0.0, 0.45),  # X drives Y
        (0.45, 0.0),  # Y drives X
    ],
)
def test_directional_ar_coupling(coupling_x, coupling_y):
    """TE should be greater in the known coupling direction."""
    x, y = _generate_one_way_ar(
        n=2500,
        coupling_x=coupling_x,
        coupling_y=coupling_y,
        seed=0,
    )

    config = KSGConfig(k_neighbors=2)

    te_xy = estimate_te(
        source=x,
        target=y,
        config=config,
    ).raw_te

    te_yx = estimate_te(
        source=y,
        target=x,
        config=config,
    ).raw_te

    if coupling_y > 0:
        assert te_xy > te_yx
    else:
        assert te_yx > te_xy


def test_second_estimator_uses_shared_interface(monkeypatch):
    """A second estimator works without modifying estimate_te."""

    @dataclass(frozen=True)
    class ExampleConfig:
        multiplier: float = 2.0

    source = np.arange(20, dtype=float)
    target = np.arange(20, dtype=float) * 0.5

    def example_estimator(context, config):
        future, target_past, source_past = context.histories()

        assert future.shape == (17,)
        assert target_past.shape == (17, 2)
        assert source_past.shape == (17, 2)

        np.testing.assert_array_equal(
            source_past[0], [source[1], source[0]]
        )

        return config.multiplier * 0.125

    monkeypatch.setitem(
        ESTIMATORS,
        "example",
        EstimatorSpec(
            config_type=ExampleConfig,
            compute=example_estimator,
        ),
    )

    result = estimate_te(
        source=source,
        target=target,
        estimator="example",
        target_history=2,
        source_history=2,
        delay=2,
        config=ExampleConfig(),
    )

    assert result.raw_te == 0.25
    assert result.n_history_rows == 17
    assert result.resolved_settings["estimator"] == "example"
    assert result.resolved_settings["config"]["multiplier"] == 2.0
