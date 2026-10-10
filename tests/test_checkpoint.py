import json
import hashlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from autodetect_repro.checkpoint import (
    CheckpointError,
    CheckpointStore,
    config_fingerprint,
    sample_id,
)
from autodetect_repro.config import load_config
from autodetect_repro.types import KnowledgePoint, SampleRecord


@pytest.fixture
def config():
    return load_config(
        Path("configs/siliconflow_smallscale.yaml"),
        {"SILICONFLOW_API_KEY": "unit-test-secret"},
    )


def make_record(identifier="record-1", prompt="Return a list"):
    return SampleRecord(
        sample_id=identifier,
        group="format",
        subtask="text format",
        knowledge_point="list format",
        phase="seed",
        iteration=0,
        ordinal=0,
        prompt=prompt,
        key_point="list constraint",
        answer="- one",
        reference_answer="- one\n- two",
        comparison="Rating: [[3]]",
        score=3.0,
        is_weakness=True,
        status="valid",
        error=None,
        request_count=3,
        prompt_tokens=30,
        completion_tokens=15,
        total_tokens=45,
    )


def test_sample_id_is_stable_across_processes():
    point = KnowledgePoint("format", "text format", "list format")
    expected = sample_id(point, "adaptive", 2, 0)
    script = (
        "from autodetect_repro.checkpoint import sample_id; "
        "from autodetect_repro.types import KnowledgePoint; "
        "print(sample_id(KnowledgePoint('format','text format','list format'),"
        "'adaptive',2,0))"
    )

    actual = subprocess.check_output([sys.executable, "-c", script], text=True).strip()

    assert actual == expected
    assert len(actual) == 64


def test_manifest_is_created_atomically_without_secret(tmp_path, config, monkeypatch):
    replacements = []
    real_replace = os.replace

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("autodetect_repro.checkpoint.os.replace", recording_replace)

    store = CheckpointStore(tmp_path / "run", config)

    manifest_text = store.manifest_path.read_text()
    assert len(replacements) == 1
    assert replacements[0][1] == store.manifest_path
    assert not replacements[0][0].exists()
    assert json.loads(manifest_text)["config_fingerprint"] == config_fingerprint(config)
    assert "unit-test-secret" not in manifest_text


def test_append_flushes_and_fsyncs_then_loads_record(tmp_path, config, monkeypatch):
    fsync_calls = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        fsync_calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr("autodetect_repro.checkpoint.os.fsync", recording_fsync)
    store = CheckpointStore(tmp_path / "run", config)

    store.append(make_record())

    assert fsync_calls
    assert store.records() == [make_record()]


def test_malformed_final_line_is_quarantined_and_prior_records_resume(
    tmp_path, config
):
    store = CheckpointStore(tmp_path / "run", config)
    store.append(make_record())
    with store.samples_path.open("a", encoding="utf-8") as handle:
        handle.write('{"partial":')

    records = store.records()

    assert records == [make_record()]
    assert store.quarantine_path.read_text() == '{"partial":'
    assert len(store.samples_path.read_text().splitlines()) == 1


def test_corruption_in_middle_of_checkpoint_is_rejected(tmp_path, config):
    store = CheckpointStore(tmp_path / "run", config)
    valid = json.dumps(make_record().__dict__, separators=(",", ":"))
    store.samples_path.write_text(valid + "\nnot-json\n" + valid + "\n")

    with pytest.raises(CheckpointError, match="line 2"):
        store.records()


def test_manifest_mismatch_is_rejected(tmp_path, config):
    run_dir = tmp_path / "run"
    CheckpointStore(run_dir, config)
    altered = replace(
        config,
        search=replace(config.search, random_seed=99),
    )

    with pytest.raises(CheckpointError, match="fingerprint"):
        CheckpointStore(run_dir, altered)


def test_legacy_manifest_without_embedding_is_resumable(tmp_path, config):
    run_dir = tmp_path / "run"
    store = CheckpointStore(run_dir, config)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["config"].pop("embedding", None)
    canonical = json.dumps(
        manifest["config"], ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    manifest["config_fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert CheckpointStore(run_dir, config).events() == []


def test_record_containing_secret_is_rejected_before_write(tmp_path, config):
    store = CheckpointStore(tmp_path / "run", config)

    with pytest.raises(CheckpointError, match="secret"):
        store.append(make_record(prompt="Do not leak unit-test-secret"))

    assert "unit-test-secret" not in store.samples_path.read_text()


def test_event_append_and_resume_without_secret(tmp_path, config):
    store = CheckpointStore(tmp_path / "run", config)
    store.append_event("diversity_metrics", {"gain": 2})
    assert store.events() == [{"event_type": "diversity_metrics", "gain": 2}]
    assert "unit-test-secret" not in store.events_path.read_text()
