from autodetect_repro.novelty import NoveltyFilter, NoveltyLabel


def test_filter_rejects_empty_and_exact_duplicates_before_model():
    filterer = NoveltyFilter(embedding=lambda text: [1.0, 0.0])
    result = filterer.filter(
        ["", "same", "fresh"],
        history=["same"],
        classify=lambda candidate, neighbors: NoveltyLabel.DISTINCT,
        budget=3,
    )
    assert [item.candidate for item in result.selected] == ["fresh"]
    assert {item.candidate for item in result.rejected} == {"", "same"}


def test_filter_uses_frozen_history_and_keeps_related_with_budget():
    vectors = {"old": [1.0, 0.0], "related": [0.9, 0.1], "distinct": [0.0, 1.0]}
    filterer = NoveltyFilter(embedding=lambda text: vectors[text], top_k=1)
    result = filterer.filter(
        ["related", "distinct"],
        history=["old"],
        classify=lambda candidate, neighbors: NoveltyLabel.RELATED,
        budget=1,
    )
    assert len(result.selected) == 1
    assert result.selected[0].candidate == "distinct"
    assert result.selected[0].semantic_novelty == 1.0
    assert result.history_snapshot == ("old",)
