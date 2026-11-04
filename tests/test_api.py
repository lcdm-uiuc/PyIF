
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
