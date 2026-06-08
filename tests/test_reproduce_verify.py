"""Tests for the benchmark reference-verification gate (`make verify`)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REF = "benchmarks/reference_summary.json"


def _run(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run(
        [sys.executable, "-m", "benchmarks.reproduce", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def test_verify_passes_against_reference(tmp_path):
    out = tmp_path / "summary.json"
    # 5 seeds is fast; a generous tolerance absorbs the small-sample wobble.
    result = _run(
        ["--seeds", "5", "--out", str(out), "--verify", _REF],
        env_extra={"GREEN_SARC_VERIFY_TOL": "0.10"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "verify: OK" in result.stdout


def test_verify_fails_on_drift(tmp_path):
    # Build a deliberately wrong reference (token counts doubled) and verify against it.
    ref = json.loads((_REPO_ROOT / _REF).read_text(encoding="utf-8"))
    for cond in ref["conditions"].values():
        if "tokens" in cond:
            cond["tokens"] = float(cond["tokens"]) * 2.0
    bad_ref = tmp_path / "bad_ref.json"
    bad_ref.write_text(json.dumps(ref), encoding="utf-8")

    out = tmp_path / "summary.json"
    result = _run(["--seeds", "5", "--out", str(out), "--verify", str(bad_ref)])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "verify: FAILED" in result.stdout
    assert "tokens" in result.stdout  # the drift table names the offending metric
