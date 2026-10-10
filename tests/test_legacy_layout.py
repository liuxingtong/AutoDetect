"""Pin the layout of the legacy/ directory.

The three original per-domain AutoDetect entry points
(autodetect_if, autododetect_math, autodetect_code) are not used
by the current reproduction package, but they are kept under
``legacy/`` for reference and for direct numerical reproduction
of the original paper.

If a future refactor moves or deletes these directories, this
test fails so the operator notices before the paper's exact
numbers stop being reproducible from a clean checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY = REPO_ROOT / "legacy"

DOMAINS = ("autodetect_if", "autodetect_math", "autodetect_code")


@pytest.mark.parametrize("domain", DOMAINS)
def test_legacy_domain_directory_exists(domain):
    assert (LEGACY / domain).is_dir(), (
        "legacy directory for {} is missing; the original per-domain "
        "AutoDetect scripts must be preserved here for the paper's "
        "exact numbers to remain reproducible.".format(domain)
    )


@pytest.mark.parametrize("domain", DOMAINS)
def test_legacy_domain_has_entry_point(domain):
    entry = LEGACY / domain / "scripts" / (domain + ".py")
    assert entry.is_file(), (
        "expected the original {} entry point at {}".format(
            domain, entry
        )
    )


@pytest.mark.parametrize("domain", DOMAINS)
def test_legacy_domain_has_shell_driver(domain):
    sh = LEGACY / domain / "scripts" / (domain + ".sh")
    assert sh.is_file(), (
        "expected the original {} shell driver at {}".format(
            domain, sh
        )
    )


@pytest.mark.parametrize("domain", DOMAINS)
def test_legacy_domain_has_count_res(domain):
    count = LEGACY / domain / "scripts" / "count_res.py"
    assert count.is_file(), (
        "expected the original count_res.py at {}".format(count)
    )


def test_legacy_has_readme():
    assert (LEGACY / "README.md").is_file(), (
        "legacy/README.md is missing; add it so the next reader "
        "knows why these scripts are preserved and how to run them."
    )


def test_legacy_does_not_pollute_top_level():
    """The three original domains must NOT be at the top level any
    more; if they are, the move was not committed properly."""
    for domain in DOMAINS:
        assert not (REPO_ROOT / domain).exists(), (
            "{} reappeared at the repo root; the move to legacy/ "
            "was not complete.".format(domain)
        )


def test_legacy_does_not_shadow_autodetect_repro():
    """The legacy/ directory must not be importable as a top-level
    Python module name (e.g. ``import legacy``), so it cannot
    shadow autodetect_repro/ or anything else in the import path.

    Python 3.9+ treats a directory without __init__.py as an
    *implicit namespace package*; finding one with the same name
    is fine as long as it is not also a regular package.  We assert
    that no top-level module named ``legacy`` has been registered
    in sys.modules and that no submodule of it is shadowing
    anything in the package."""
    import sys
    import importlib

    # Make sure no eager import has populated it.
    assert "legacy" not in sys.modules
    # Importing a real submodule path that *is* in legacy/ should
    # not work because the inner .py files (e.g.
    # legacy/autodetect_if/scripts/autodetect_if.py) are not on
    # any sys.path; their parent legacy/ is not a regular package
    # and they have no setup that exposes them as importable.
    with pytest.raises((ImportError, ModuleNotFoundError)):
        importlib.import_module("legacy.autodetect_if.scripts.autodetect_if")
