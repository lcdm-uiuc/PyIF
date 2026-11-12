
import numpy as np
import pytest

from PyIF import helper
from PyIF.workspace import TEWorkspace
from PyIF import te_compute
import PyIF.workspace as workspace_module


def test_workspace_spaces_match_legacy():
    rng = np.random.default_rng(123)

    target = rng.normal(size=100)
    source_a = rng.normal(size=100)
    source_b = rng.normal(size=100)

    workspace = TEWorkspace({
        "target": target,
        "A": source_a,
        "B": source_b,
    })

    for name, source in [("A", source_a), ("B", source_b)]:
        original = helper.make_spaces(target, source, embedding=2)
        cached = workspace.make_spaces(name, "target", embedding=2)

        for a, b in zip(original, cached):
            np.testing.assert_array_equal(a, b)


def test_target_trees_are_reused():
    rng = np.random.default_rng(456)

    workspace = TEWorkspace({
        "target": rng.normal(size=100),
        "A": rng.normal(size=100),
        "B": rng.normal(size=100),
    })

    first = workspace.get_target_trees("target", embedding=2)
    second = workspace.get_target_trees("target", embedding=2)

    # Both requests must reuse the exact same tree objects.
    assert first[0] is second[0]
    assert first[1] is second[1]

    # Different embedding settings need separate trees.
    different = workspace.get_target_trees("target", embedding=3)
    assert first[0] is not different[0]


def test_original_data_changes_do_not_affect_cache():
    target = np.arange(20, dtype=float)
    source = np.arange(20, dtype=float) * 2

    workspace = TEWorkspace({
        "target": target,
        "source": source,
    })

    before = workspace.make_spaces("source", "target", embedding=2)

    # Modify the original arrays after workspace creation.
    target[:] = -999
    source[:] = -999

    after = workspace.make_spaces("source", "target", embedding=2)

    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_rejects_misaligned_lengths():
    with pytest.raises(ValueError):
        TEWorkspace({
            "A": np.arange(20),
            "B": np.arange(19),
        })


@pytest.mark.parametrize(
    "embedding,k",
    [(1, 1), (2, 2), (3, 1)]
)
def test_workspace_te_matches_legacy(embedding, k):
    rng = np.random.default_rng(12345)

    target = rng.normal(size=200)
    source_a = rng.normal(size=200)
    source_b = rng.normal(size=200)

    workspace = TEWorkspace({
        "target": target,
        "A": source_a,
        "B": source_b,
    })

    for name, source in [
        ("A", source_a),
        ("B", source_b)
    ]:
        original = te_compute.te_compute(
            X=target,
            Y=source,
            k=k,
            embedding=embedding,
            safetyCheck=False,
            GPU=False
        )

        cached = workspace.estimate_ksg(
            source_name=name,
            target_name="target",
            k=k,
            embedding=embedding
        )

        np.testing.assert_allclose(
            cached,
            original,
            rtol=0,
            atol=1e-12
        )


def test_workspace_reduces_tree_construction(monkeypatch):
    rng = np.random.default_rng(789)

    workspace = TEWorkspace({
        "target": rng.normal(size=100),
        "A": rng.normal(size=100),
        "B": rng.normal(size=100),
    })

    original_kdtree = workspace_module.KDTree
    tree_count = {"value": 0}

    def counting_kdtree(*args, **kwargs):
        tree_count["value"] += 1
        return original_kdtree(*args, **kwargs)

    monkeypatch.setattr(
        workspace_module,
        "KDTree",
        counting_kdtree
    )

    workspace.estimate_ksg("A", "target", embedding=1)
    workspace.estimate_ksg("B", "target", embedding=1)

    # First pair builds 4 trees.
    # Second pair builds only 2 new trees.
    assert tree_count["value"] == 6


@pytest.mark.parametrize("embedding,k", [(1, 1), (2, 2)])
def test_full_directed_pair_matrix(embedding, k):
    rng = np.random.default_rng(2026)

    series = {
        name: rng.normal(size=200)
        for name in ("A", "B", "C", "D")
    }

    workspace = TEWorkspace(series)

    for target_name, target in series.items():
        for source_name, source in series.items():
            if source_name == target_name:
                continue

            original = te_compute.te_compute(
                X=target,
                Y=source,
                k=k,
                embedding=embedding,
                GPU=False,
                safetyCheck=False,
            )

            cached = workspace.estimate_ksg(
                source_name,
                target_name,
                k=k,
                embedding=embedding,
            )

            np.testing.assert_allclose(
                cached, original,
                rtol=0, atol=1e-12
            )

    # Each target has its own cached search trees.
    assert len(workspace._target_trees) == len(series)



def test_workspace_accepts_numpy_integers():
    rng = np.random.default_rng(123)
    workspace = TEWorkspace({
        "source": rng.normal(size=200),
        "target": rng.normal(size=200),
    })

    original = workspace.estimate_ksg(
        "source", "target", k=2, embedding=2
    )

    numpy_integer = workspace.estimate_ksg(
        "source", "target",
        k=np.int64(2),
        embedding=np.int32(2),
    )

    assert original == numpy_integer

    assert workspace.get_target(
        "target", np.int64(2)
    ) is workspace.get_target("target", 2)


def test_workspace_rejects_boolean_settings():
    rng = np.random.default_rng(456)
    workspace = TEWorkspace({
        "source": rng.normal(size=100),
        "target": rng.normal(size=100),
    })

    with pytest.raises(ValueError):
        workspace.estimate_ksg(
            "source", "target", k=True
        )

    with pytest.raises(ValueError):
        workspace.estimate_ksg(
            "source", "target", embedding=True
        )
