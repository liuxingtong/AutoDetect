from autodetect_repro.diversity import (
    behavioral_metrics,
    semantic_metrics,
    semantic_novelty,
)
import pytest


def test_semantic_novelty_uses_mean_knn_distance():
    assert semantic_novelty([0.8, 0.6]) == pytest.approx(0.3)


def test_semantic_metrics_freezes_history_and_counts_unique_gain():
    result = semantic_metrics(
        ["new-a", "new-b"],
        ["old"],
        embeddings={
            "new-a": [1.0, 0.0],
            "new-b": [0.0, 1.0],
            "old": [1.0, 0.0],
        },
        duplicate_ids={"new-b"},
    )
    assert result.mean_novelty == 0.5
    assert result.unique_question_gain == 1


def test_behavioral_metrics_counts_confirmed_clusters_and_gain():
    result = behavioral_metrics(previous_count=2, current_labels=["new", None, "newer"], target_calls=4)
    assert result.richness == 4
    assert result.gain == 2
    assert result.discovery_rate == 0.5
