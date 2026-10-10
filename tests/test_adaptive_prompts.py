from autodetect_repro.prompts import (
    behavioral_clustering_prompt,
    controller_prompt,
    failure_signature_prompt,
    promotion_prompt,
)


def test_adaptive_prompts_require_json_and_boundary_labels():
    assert "JSON" in controller_prompt({}, {}, {}, 10)
    assert "Different" in behavioral_clustering_prompt({}, [])
    assert "observed_behavior" in failure_signature_prompt("q", "a", "r", "e")
    assert "promote" in promotion_prompt({}, [])
