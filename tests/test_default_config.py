"""Pin the default small-scale config to the adaptive path.

Without this, the framework's adaptive features (analysis pipeline,
controller, branches, promotion, search_decision events, the new
report sections) are wired up but never triggered, because the
default ``search.mode`` is baseline.

If a future refactor changes the default back to baseline, this test
fails loudly so the operator notices before they reproduce a
"successful" run that produced no branches and no promoted points.
"""

from __future__ import annotations

from pathlib import Path

from autodetect_repro.config import load_config


def test_smallscale_config_runs_adaptive_not_baseline():
    config = load_config(
        Path("configs/siliconflow_smallscale.yaml"),
        {"SILICONFLOW_API_KEY": "unit-test-secret"},
    )
    assert config.search.mode != "baseline", (
        "configs/siliconflow_smallscale.yaml is still in baseline mode; "
        "the adaptive framework (analysis pipeline, controller, "
        "branches, promotion) will be silently bypassed on every run. "
        "Set search.mode to adaptive_rule or adaptive_llm."
    )


def test_smallscale_config_uses_rule_controller():
    """The reproduction uses adaptive_rule (not adaptive_llm) so it
    does not require an LLM-as-controller call.  Pin that."""
    config = load_config(
        Path("configs/siliconflow_smallscale.yaml"),
        {"SILICONFLOW_API_KEY": "unit-test-secret"},
    )
    assert config.search.mode == "adaptive_rule", (
        "expected adaptive_rule for the small-scale reproduction; "
        "adaptive_llm would double the LLM call rate per adaptive "
        "iteration. Got: {}".format(config.search.mode)
    )


def test_smallscale_config_has_promotion_knobs():
    """The adaptive path needs auto_promote and
    candidate_questions_per_round; without them the runner still
    runs but no PROMOTE gate ever fires."""
    config = load_config(
        Path("configs/siliconflow_smallscale.yaml"),
        {"SILICONFLOW_API_KEY": "unit-test-secret"},
    )
    assert config.search.candidate_questions_per_round >= 1
    # auto_promote defaults to True in the dataclass, but the
    # reproduction config should not flip it off.
    assert config.search.auto_promote is True, (
        "auto_promote is False in siliconflow_smallscale.yaml; "
        "branches will be recorded but never promote to discovered "
        "knowledge points, so the reproduction will not exercise the "
        "promotion chain."
    )
