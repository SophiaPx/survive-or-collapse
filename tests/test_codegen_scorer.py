"""Smoke test for the codegen in-loop val scorer.

Verifies that run_codegen_tests + the parquet built by build_codegen_val_parquet.py
correctly identify canonical solutions as passing and obvious-wrong code as failing.
No GPU / no Ray — runs in CI on a CPU box.

    pytest tests/test_codegen_scorer.py -s
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from selfplay_grpo.rewards.grpo_reward_manager import run_codegen_tests  # noqa: E402
from selfplay_grpo.utils.code_utils.python_executor import PythonExecutor  # noqa: E402

PARQUET = REPO / "data" / "codegen_val_humaneval_plus_smoke.parquet"


@pytest.fixture(scope="module")
def executor():
    ex = PythonExecutor(get_answer_from_stdout=False, max_workers=8, timeout_length=5)
    yield ex
    ex.cleanup()


@pytest.fixture(scope="module")
def sample_rows():
    assert PARQUET.exists(), f"Run scripts/build_codegen_val_parquet.py first ({PARQUET} missing)"
    df = pd.read_parquet(PARQUET)
    # Pick three problems with non-empty plus tests so the assertion is meaningful.
    rows = []
    for _, row in df.iterrows():
        ei = row["extra_info"]
        if len(ei["base_inputs"]) > 0 and len(ei["plus_inputs"]) > 0:
            rows.append(row)
        if len(rows) == 3:
            break
    assert len(rows) == 3, "Need at least 3 humaneval rows with both base+plus tests"
    return rows


def test_canonical_solution_passes(executor, sample_rows):
    """The canonical_solution stored in reward_model.ground_truth must pass its own tests."""
    for row in sample_rows:
        ei = row["extra_info"]
        canonical = row["reward_model"]["ground_truth"]
        base_pass = run_codegen_tests(
            executor, canonical,
            entry_point=ei["entry_point"],
            inputs=list(ei["base_inputs"]),
            expected=list(ei["base_outputs"]),
            atol=float(ei["atol"]),
        )
        plus_pass = run_codegen_tests(
            executor, canonical,
            entry_point=ei["entry_point"],
            inputs=list(ei["plus_inputs"]),
            expected=list(ei["plus_outputs"]),
            atol=float(ei["atol"]),
        )
        assert base_pass, f"{ei['task_id']}: canonical failed its own base tests"
        assert plus_pass, f"{ei['task_id']}: canonical failed its own plus tests"


def test_wrong_code_fails(executor, sample_rows):
    """An obviously-wrong stub must NOT pass."""
    for row in sample_rows:
        ei = row["extra_info"]
        wrong = f"def {ei['entry_point']}(*args, **kwargs):\n    return 0"
        base_pass = run_codegen_tests(
            executor, wrong,
            entry_point=ei["entry_point"],
            inputs=list(ei["base_inputs"]),
            expected=list(ei["base_outputs"]),
            atol=float(ei["atol"]),
        )
        assert not base_pass, f"{ei['task_id']}: wrong stub unexpectedly passed base tests"
