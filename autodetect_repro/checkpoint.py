"""Atomic manifests and append-only JSONL experiment checkpoints."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import RunConfig
from .types import KnowledgePoint, SampleRecord


class CheckpointError(RuntimeError):
    """Raised when checkpoint state is unsafe or incompatible."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def sample_id(
    point: KnowledgePoint,
    phase: str,
    iteration: int,
    ordinal: int,
) -> str:
    payload = {
        "group": point.group,
        "subtask": point.subtask,
        "knowledge_point": point.name,
        "phase": phase,
        "iteration": iteration,
        "ordinal": ordinal,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _public_config(config: RunConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["models"].pop("api_key", None)
    payload["output_root"] = str(config.output_root)
    return payload


def config_fingerprint(config: RunConfig) -> str:
    return hashlib.sha256(
        _canonical_json(_public_config(config)).encode("utf-8")
    ).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(path.name), dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


class CheckpointStore:
    """Own the immutable manifest and recoverable sample checkpoint."""

    def __init__(self, run_dir: Path, config: RunConfig) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.json"
        self.samples_path = self.run_dir / "samples.jsonl"
        self.quarantine_path = self.run_dir / "samples.jsonl.quarantine"
        self._secret = config.models.api_key
        fingerprint = config_fingerprint(config)

        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CheckpointError("manifest is unreadable") from exc
            if manifest.get("config_fingerprint") != fingerprint:
                raise CheckpointError("config fingerprint does not match manifest")
        else:
            manifest = {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config_fingerprint": fingerprint,
                "config": _public_config(config),
            }
            _atomic_write(
                self.manifest_path,
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
            )
        self.samples_path.touch(exist_ok=True)

    def append(self, record: SampleRecord) -> None:
        line = _canonical_json(asdict(record))
        if self._secret and self._secret in line:
            raise CheckpointError("refusing to persist a record containing a secret")
        with self.samples_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _repair_final_line(self, valid_lines: List[str], bad_line: str) -> None:
        with self.quarantine_path.open("a", encoding="utf-8") as handle:
            handle.write(bad_line)
            handle.flush()
            os.fsync(handle.fileno())
        content = "".join(line.rstrip("\r\n") + "\n" for line in valid_lines)
        _atomic_write(self.samples_path, content)

    def records(self) -> List[SampleRecord]:
        try:
            lines = self.samples_path.read_text(encoding="utf-8").splitlines(
                keepends=True
            )
        except OSError as exc:
            raise CheckpointError("cannot read samples checkpoint") from exc

        nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
        last_nonempty = nonempty_indexes[-1] if nonempty_indexes else None
        records = []
        valid_lines = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(SampleRecord(**payload))
                valid_lines.append(line)
            except (json.JSONDecodeError, TypeError) as exc:
                if index != last_nonempty:
                    raise CheckpointError(
                        "checkpoint corruption at line {}".format(index + 1)
                    ) from exc
                self._repair_final_line(valid_lines, line.rstrip("\r\n"))
        return records
