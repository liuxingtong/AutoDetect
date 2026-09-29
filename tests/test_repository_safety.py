import hashlib
import re
import subprocess
from pathlib import Path


PROVIDED_KEY_SHA256 = "ac6ff1ea7fa4989976e57aeef24ddb41d3fbfdb78430facada252169732c913d"
HIGH_ENTROPY_KEY = re.compile(r"sk-[A-Za-z0-9]{20,}")
KEY_TOKEN = re.compile(r"sk-[A-Za-z0-9_-]+")


def tracked_text_files():
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        path = Path(raw_path.decode())
        try:
            yield path, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def test_git_tracked_text_contains_no_api_credentials():
    findings = []
    for path, text in tracked_text_files():
        if HIGH_ENTROPY_KEY.search(text):
            findings.append(str(path))
        for token in KEY_TOKEN.findall(text):
            if hashlib.sha256(token.encode()).hexdigest() == PROVIDED_KEY_SHA256:
                findings.append(str(path))

    assert findings == []


def test_required_reproduction_documents_and_pitfall_schema_exist():
    pitfalls = Path("docs/reproduction-pitfalls.md")
    runbook = Path("docs/siliconflow-reproduction-runbook.md")

    assert pitfalls.is_file()
    assert runbook.is_file()
    header = next(line for line in pitfalls.read_text().splitlines() if line.startswith("|"))
    for column in ("Symptom", "Cause", "Resolution", "Impact", "Prevention"):
        assert column in header


def test_runbook_keeps_environment_cache_and_outputs_below_data():
    text = Path("docs/siliconflow-reproduction-runbook.md").read_text()

    assert "/data/liuxingtong/envs/autodetect-repro" in text
    assert "/data/liuxingtong/cache/pip" in text
    assert "/data/liuxingtong/cache/huggingface" in text
    assert "/data/liuxingtong/outputs/autodetect-smallscale" in text
    assert "/home/liuxingtong" not in text
    assert "~/" not in text
