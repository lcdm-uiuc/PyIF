
import numpy as np
import pytest

from PyIF import estimate_te, te_compute
from PyIF.config import KSGConfig


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
