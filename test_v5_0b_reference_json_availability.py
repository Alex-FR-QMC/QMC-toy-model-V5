# -*- coding: utf-8 -*-
"""
V5-0b §3.C.1 — Reference 6d JSON availability harness.

Read-only audit. Detects which 6d reference artifacts are present on
disk and characterizes their schema relative to keys expected by the
V5-0b protocol. The test is INTENTIONALLY non-blocking on absence.

Rules:
    MISSING                    != FAIL
    AVAILABLE_SCHEMA_VARIANT   != FAIL
    AVAILABLE_ALIAS            != FAIL
    INVALID_JSON               == FAIL (a present file must be readable)

This file:
    - imports NO mcq_v5.* module
    - performs NO numerical computation
    - does NOT recompute P5/P5bis/P4
    - does NOT call ExperimentProtocol
    - reads JSON only and reports schema descriptions

Verdict for this file (when all tests pass):
    REFERENCE_6D_AVAILABLE: per-file status dict (AVAILABLE | MISSING |
        AVAILABLE_ALIAS | AVAILABLE_SCHEMA_VARIANT)
    FULL_REPRODUCTION: OUT_OF_SCOPE (already methodologically fixed)

Side effect (audit ajout 2):
    The aggregated verdict is exported as
    tests/v5_0b_reference_audit_result.json so the closure report can
    quote actual disk state, not assumptions.
"""

import json
import sys
from pathlib import Path

import pytest


# =========================================================================
# Search roots — including project-relative for refactoring robustness
# =========================================================================

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent  # mcq_v5/

SEARCH_ROOTS = [
    Path("/mnt/data"),
    Path("/mnt/user-data/uploads"),
    Path("/mnt/project"),
    _PROJECT_ROOT / "data",
    _PROJECT_ROOT / "references",
    Path.cwd(),
]


def _existing_roots() -> list:
    return [p for p in SEARCH_ROOTS if p.exists() and p.is_dir()]


# =========================================================================
# Expected references with alias support
# =========================================================================

EXPECTED_REFERENCES = {
    "lambda_A_v2": {
        "filenames": ["6d_lambda_A_v2.json"],
        "expected_top_level_keys": ["D_proj", "R_sym"],
    },
    "p5bis_A": {
        "filenames": ["6d_p5bis_A.json"],
        "expected_top_level_keys": [
            "DISSOC", "B2", "Dh",
            "frac_h_active", "frac_grad_active",
            "jaccard_h_grad", "grad_status",
        ],
    },
    "p4_strict_A": {
        "filenames": ["6d_p4_strict_A.json"],
        "expected_top_level_keys": [
            "P4-NO-STRONG-SIGNAL", "variance_audit",
        ],
    },
    "p4_var": {
        "filenames": ["6d_p4_var.json", "6d_p4_var_audit.json"],
        "expected_top_level_keys": [
            "RR2", "RR3", "STR", "RSR",
        ],
    },
}

ALLOWED_STATUSES = frozenset({
    "AVAILABLE",
    "AVAILABLE_ALIAS",
    "AVAILABLE_SCHEMA_VARIANT",
    "MISSING",
})


def find_reference_file(candidates, search_roots):
    """Search candidates (in order) across roots (in order).

    Returns (path, matched_filename). The first hit wins.
    """
    for root in search_roots:
        for fname in candidates:
            p = root / fname
            if p.exists() and p.is_file():
                return (p, fname)
    return (None, None)


def audit_reference(name, spec, roots):
    """Audit one reference. Returns a JSON-safe dict."""
    canonical_filename = spec["filenames"][0]
    expected_keys = spec["expected_top_level_keys"]

    path, matched = find_reference_file(spec["filenames"], roots)

    if path is None:
        return {
            "name": name,
            "status": "MISSING",
            "canonical_filename": canonical_filename,
            "expected_top_level_keys": list(expected_keys),
            "searched_in": [str(r) for r in roots],
        }

    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as e:
        return {
            "name": name,
            "status": "INVALID_JSON",
            "canonical_filename": canonical_filename,
            "matched_filename": matched,
            "path": str(path),
            "error": str(e),
        }
    except OSError as e:
        return {
            "name": name,
            "status": "INVALID_JSON",
            "canonical_filename": canonical_filename,
            "matched_filename": matched,
            "path": str(path),
            "error": f"OSError: {e}",
        }

    if not isinstance(data, dict):
        return {
            "name": name,
            "status": "INVALID_JSON",
            "canonical_filename": canonical_filename,
            "matched_filename": matched,
            "path": str(path),
            "error": f"top-level is {type(data).__name__}, not dict",
        }

    present_top_level_keys = sorted(data.keys())
    missing_expected = sorted(
        k for k in expected_keys if k not in data
    )

    is_alias = (matched != canonical_filename)
    has_all_expected_keys = (len(missing_expected) == 0)

    if has_all_expected_keys and not is_alias:
        status = "AVAILABLE"
    elif is_alias:
        status = "AVAILABLE_ALIAS"
    else:
        status = "AVAILABLE_SCHEMA_VARIANT"

    return {
        "name": name,
        "status": status,
        "canonical_filename": canonical_filename,
        "matched_filename": matched,
        "path": str(path),
        "expected_top_level_keys": list(expected_keys),
        "present_top_level_keys": present_top_level_keys,
        "missing_expected_keys": missing_expected,
        "n_top_level_keys": len(present_top_level_keys),
    }


def build_full_report():
    """Compose the full V5-0b §3.C.1 + §3.C.2 report."""
    roots = _existing_roots()
    per_reference = {}
    for name, spec in EXPECTED_REFERENCES.items():
        per_reference[name] = audit_reference(name, spec, roots)

    return {
        "REFERENCE_6D_AVAILABLE": per_reference,
        "FULL_REPRODUCTION": {
            "status": "OUT_OF_SCOPE",
            "reason": "h-dynamics not implemented in V5-0a",
        },
        "search_roots_existing": [str(r) for r in roots],
        "search_roots_configured": [str(r) for r in SEARCH_ROOTS],
    }


# Cached report (built once per test session)
_CACHED_REPORT = None


def _get_report():
    global _CACHED_REPORT
    if _CACHED_REPORT is None:
        _CACHED_REPORT = build_full_report()
    return _CACHED_REPORT


# =========================================================================
# Tests
# =========================================================================

def test_reference_availability_statuses_in_allowed_set():
    """No reference should have an unexpected status. INVALID_JSON
    would fail this assertion intentionally."""
    report = _get_report()
    for name, entry in report["REFERENCE_6D_AVAILABLE"].items():
        assert entry["status"] in ALLOWED_STATUSES, (
            f"{name}: unexpected status {entry['status']!r}"
        )


def test_reference_availability_missing_is_not_fail():
    """A reference being MISSING must not cause the test to fail."""
    report = _get_report()
    for name, entry in report["REFERENCE_6D_AVAILABLE"].items():
        if entry["status"] == "MISSING":
            assert "searched_in" in entry
            assert isinstance(entry["searched_in"], list)


def test_present_json_files_are_readable_dicts():
    """For each reference NOT marked MISSING, the file must be a valid
    JSON dict (top-level)."""
    report = _get_report()
    for name, entry in report["REFERENCE_6D_AVAILABLE"].items():
        if entry["status"] == "MISSING":
            continue
        assert entry["status"] != "INVALID_JSON", (
            f"{name}: present file is not a valid JSON dict: {entry}"
        )
        assert "present_top_level_keys" in entry


def test_p4_var_alias_supported():
    """If 6d_p4_var.json is absent but 6d_p4_var_audit.json is present,
    the entry must be classified AVAILABLE_ALIAS, not MISSING."""
    report = _get_report()
    entry = report["REFERENCE_6D_AVAILABLE"]["p4_var"]
    if entry["status"] == "MISSING":
        pytest.skip("No p4_var file or alias present in any search root")
    if entry.get("matched_filename") == "6d_p4_var_audit.json":
        assert entry["status"] == "AVAILABLE_ALIAS"


def test_full_reproduction_is_out_of_scope():
    """Strict: FULL_REPRODUCTION must remain OUT_OF_SCOPE since
    h-dynamics is not in V5-0a."""
    report = _get_report()
    assert report["FULL_REPRODUCTION"]["status"] == "OUT_OF_SCOPE"
    assert "h-dynamics" in report["FULL_REPRODUCTION"]["reason"]


def test_reference_availability_report_json_strict():
    """The full report must be JSON-strict-serializable."""
    report = _get_report()
    s = json.dumps(report, allow_nan=False)
    reloaded = json.loads(s)
    assert "REFERENCE_6D_AVAILABLE" in reloaded
    assert "FULL_REPRODUCTION" in reloaded


def test_schema_variants_carry_diff_information():
    """AVAILABLE_SCHEMA_VARIANT and AVAILABLE_ALIAS entries must carry
    both `missing_expected_keys` and `present_top_level_keys`."""
    report = _get_report()
    for name, entry in report["REFERENCE_6D_AVAILABLE"].items():
        if entry["status"] in ("AVAILABLE_SCHEMA_VARIANT",
                               "AVAILABLE_ALIAS",
                               "AVAILABLE"):
            assert "missing_expected_keys" in entry
            assert "present_top_level_keys" in entry
            assert isinstance(entry["missing_expected_keys"], list)
            assert isinstance(entry["present_top_level_keys"], list)


def test_export_verdict_to_tests_directory():
    """Write the audit verdict to a JSON file so the closure report
    can quote actual disk state."""
    report = _get_report()
    out_path = _THIS_FILE.parent / "v5_0b_reference_audit_result.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
    assert out_path.exists()
    with open(out_path) as fh:
        reloaded = json.load(fh)
    assert reloaded["FULL_REPRODUCTION"]["status"] == "OUT_OF_SCOPE"
    assert "REFERENCE_6D_AVAILABLE" in reloaded


def test_no_mcq_v5_import():
    """V5-0b §3.C.1 forbids importing mcq_v5.* in this audit file."""
    src = Path(__file__).read_text()
    forbidden_imports = [
        "from mcq" + "_v5",
        "import mcq" + "_v5",
    ]
    for pat in forbidden_imports:
        assert pat not in src, f"forbidden mcq_v5 import: {pat}"


def test_no_numerical_computation_on_reference_values():
    """The audit must not recompute or compare numerical values."""
    src = Path(__file__).read_text()
    forbidden = [
        "np." + "testing." + "assert_allclose",
        "np." + "allclose",
        "pytest." + "approx",
    ]
    for pat in forbidden:
        assert pat not in src, (
            f"forbidden numerical pattern in availability audit: {pat}"
        )


if __name__ == "__main__":
    report = build_full_report()
    print(json.dumps(report, indent=2, allow_nan=False))
