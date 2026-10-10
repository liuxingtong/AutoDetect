from autodetect_repro.prompts import (
    adaptive_candidates_prompt,
    adaptive_prompt,
    behavioral_clustering_prompt,
    controller_prompt,
    failure_signature_prompt,
    promotion_prompt,
)
from autodetect_repro.types import KnowledgePoint


def test_adaptive_prompts_require_json_and_boundary_labels():
    assert "JSON" in controller_prompt({}, {}, {}, 10)
    assert "Different" in behavioral_clustering_prompt({}, [])
    assert "observed_behavior" in failure_signature_prompt("q", "a", "r", "e")
    assert "promote" in promotion_prompt({}, [])


def test_adaptive_prompt_includes_strategy_when_supplied():
    point = KnowledgePoint("g", "s", "knowledge-point")
    rendered = adaptive_prompt(
        point, history=(), iteration=1,
        strategy="exploit", objective="deepen the current failure pattern",
    )
    assert "exploit" in rendered
    assert "deepen the current failure pattern" in rendered


def test_adaptive_prompt_omits_strategy_clause_when_unspecified():
    point = KnowledgePoint("g", "s", "knowledge-point")
    rendered = adaptive_prompt(point, history=(), iteration=1)
    assert "Adopt the search strategy" not in rendered


def test_adaptive_candidates_prompt_includes_strategy_and_count():
    point = KnowledgePoint("g", "s", "knowledge-point")
    rendered = adaptive_candidates_prompt(
        point, history=(), iteration=2, count=4,
        strategy="verify", objective="controlled comparison",
    )
    assert "verify" in rendered
    assert "controlled comparison" in rendered
    assert "exactly 4" in rendered
