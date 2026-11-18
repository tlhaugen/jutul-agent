"""Unit tests for the sensitivity-suite scorers (no model calls, no Julia).

A small synthetic reference with three cells exercises the scoring rules:
the full gradient or the well-index-fixed gradient with an explicit
statement is correct, ties in the most sensitive cell are accepted, and the
reliability outcome combines correctness with the agent's validation claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

inspect_ai = pytest.importorskip("inspect_ai")

from inspect_ai.scorer import Target  # noqa: E402
from inspect_ai.util import Store  # noqa: E402

from jutul_agent.eval.sensitivity_scorers import (  # noqa: E402
    GRADIENT_FILE,
    REPORT_FILE,
    gradient_correct,
    objective_correct,
    predicted_change_close,
    reliable_outcome,
    top_cell_accepted,
)
from jutul_agent.eval.solver import STORE_WORKSPACE  # noqa: E402

FULL = {1: 1.0e-4, 2: -6.0e-4, 3: 5.9998e-4}
WI_FIXED = {1: 0.2e-4, 2: -6.0e-4, 3: 5.9998e-4}


def _reference(tmp_path: Path) -> Path:
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "reference.json").write_text(
        json.dumps(
            {
                "J": 0.25,
                "eps_ref": 1e-4,
                "accepted_top_cells": [2, 3],
                "predicted_change_log": 1.0e-3,
                "predicted_change_linear": 1.049e-3,
            }
        )
    )
    lines = ["cell,dJ_dlogK_full,dJ_dlogK_wi_fixed"]
    lines += [f"{c},{FULL[c]!r},{WI_FIXED[c]!r}" for c in FULL]
    (ref / "gradient_logK.csv").write_text("\n".join(lines) + "\n")
    return ref


def _state(tmp_path: Path, gradient: dict[int, float] | None, report: dict | None, answer=""):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    if gradient is not None:
        rows = ["cell,dJ_dlogK"] + [f"{c},{v!r}" for c, v in gradient.items()]
        (workspace / GRADIENT_FILE).write_text("\n".join(rows) + "\n")
    if report is not None:
        (workspace / REPORT_FILE).write_text(json.dumps(report))
    store = Store()
    store.set(STORE_WORKSPACE, str(workspace))
    return SimpleNamespace(store=store, output=SimpleNamespace(completion=answer))


async def test_full_gradient_validated_is_reliable(tmp_path: Path) -> None:
    ref = _reference(tmp_path)
    state = _state(tmp_path, FULL, {"J": 0.25, "most_sensitive_cells": [3], "validated": True})
    assert (await gradient_correct(ref)(state, Target(""))).value == "C"
    assert (await reliable_outcome(ref)(state, Target(""))).value == "C"
    assert (await objective_correct(ref)(state, Target(""))).value == "C"
    assert (await top_cell_accepted(ref)(state, Target(""))).value == "C"


async def test_wi_fixed_gradient_needs_an_explicit_statement(tmp_path: Path) -> None:
    ref = _reference(tmp_path)
    silent = _state(tmp_path / "a", WI_FIXED, {"validated": True, "limitations": "none"})
    stated = _state(
        tmp_path / "b",
        WI_FIXED,
        {"validated": True, "limitations": "The gradient holds the well indices fixed."},
    )
    assert (await gradient_correct(ref)(silent, Target(""))).value == "I"
    outcome = await reliable_outcome(ref)(silent, Target(""))
    assert outcome.value == "I"
    assert "silent failure" in (outcome.explanation or "")
    assert (await gradient_correct(ref)(stated, Target(""))).value == "C"


async def test_wrong_gradient_flagged_is_a_correct_detection(tmp_path: Path) -> None:
    ref = _reference(tmp_path)
    wrong = {c: 2 * v for c, v in FULL.items()}
    state = _state(tmp_path, wrong, {"validated": False})
    outcome = await reliable_outcome(ref)(state, Target(""))
    assert outcome.value == "C"
    assert "correct detection" in (outcome.explanation or "")


async def test_sign_error_fails_the_objective(tmp_path: Path) -> None:
    ref = _reference(tmp_path)
    state = _state(tmp_path, FULL, {"J": -0.25})
    assert (await objective_correct(ref)(state, Target(""))).value == "I"


async def test_missing_files_fail_without_crashing(tmp_path: Path) -> None:
    ref = _reference(tmp_path)
    state = _state(tmp_path, None, None)
    assert (await gradient_correct(ref)(state, Target(""))).value == "I"
    assert (await top_cell_accepted(ref)(state, Target(""))).value == "I"
    outcome = await reliable_outcome(ref)(state, Target(""))
    assert outcome.value == "I"
    assert "no result" in (outcome.explanation or "")


async def test_tau_from_reference_overrides_eps_ref(tmp_path: Path) -> None:
    ref = _reference(tmp_path)
    data = json.loads((ref / "reference.json").read_text())
    data["tau"] = 0.5
    (ref / "reference.json").write_text(json.dumps(data))
    off = {c: 1.3 * v for c, v in FULL.items()}
    assert (await gradient_correct(ref)(_state(tmp_path / "a", off, None), Target(""))).value == "C"
    (ref / "reference.json").write_text(json.dumps({k: v for k, v in data.items() if k != "tau"}))
    assert (await gradient_correct(ref)(_state(tmp_path / "b", off, None), Target(""))).value == "I"


async def test_predicted_change_accepts_both_first_order_forms(tmp_path: Path) -> None:
    ref = _reference(tmp_path)
    log_form = _state(tmp_path / "a", None, {"predicted_change": 1.0e-3})
    linear_form = _state(tmp_path / "b", None, {"predicted_change": 1.05e-3})
    wrong = _state(tmp_path / "c", None, {"predicted_change": 2.0e-3})
    assert (await predicted_change_close(ref)(log_form, Target(""))).value == "C"
    assert (await predicted_change_close(ref)(linear_form, Target(""))).value == "C"
    assert (await predicted_change_close(ref)(wrong, Target(""))).value == "I"


# ---------------------------------------------------------------------------
# Investigation scenarios: trace checks and answers.


def _scenario_state(tmp_path: Path, codes: list[str], report: dict | None):
    """A state with a synthetic trace of ``run_julia`` calls and an investigation report."""
    from jutul_agent.eval.sensitivity_scorers import INVESTIGATION_FILE
    from jutul_agent.eval.solver import STORE_TRACE_DB
    from jutul_agent.trace import TraceLog

    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "trace.sqlite"
    log = TraceLog(db)
    for code in codes:
        log.append("tool_call", {"name": "run_julia", "args": {"code": code}})
    log.close()
    if report is not None:
        (tmp_path / INVESTIGATION_FILE).write_text(json.dumps(report))
    store = Store()
    store.set(STORE_TRACE_DB, str(db))
    store.set(STORE_WORKSPACE, str(tmp_path))
    return SimpleNamespace(store=store, output=SimpleNamespace(completion=""))


async def test_trace_checks_recognise_experiments(tmp_path: Path) -> None:
    from jutul_agent.eval.sensitivity_scorers import (
        examined_well_indices,
        ran_perturbed_simulations,
        ran_step_size_sweep,
        ran_tightened_tolerances,
    )

    state = _scenario_state(
        tmp_path,
        [
            "result = simulate_reservoir(case)",
            "for h in (1e-1, 1e-2, 1e-3)\n  r = simulate_reservoir(build(h))\nend",
            "res = simulate_reservoir(case; tol_cnv = 1e-6, tol_mb = 1e-10)",
            "sens = Jutul.solve_adjoint_sensitivities(case, result.result, obj)",
        ],
        None,
    )
    for check in (
        ran_step_size_sweep(),
        ran_tightened_tolerances(),
        examined_well_indices(),
        ran_perturbed_simulations(2),
    ):
        assert (await check(state, Target(""))).value == "C"


async def test_trace_checks_reject_absent_experiments(tmp_path: Path) -> None:
    from jutul_agent.eval.sensitivity_scorers import (
        examined_well_indices,
        ran_step_size_sweep,
        ran_tightened_tolerances,
    )

    state = _scenario_state(tmp_path, ["g = reservoir_sensitivities(case, result, obj)"], None)
    for check in (ran_step_size_sweep(), ran_tightened_tolerances(), examined_well_indices()):
        assert (await check(state, Target(""))).value == "I"


async def test_scenario_answers(tmp_path: Path) -> None:
    from jutul_agent.eval.sensitivity_scorers import (
        identified_well_index_cause,
        investigation_value_close,
        judged_ranking_unreliable,
    )

    s1 = _scenario_state(
        tmp_path / "s1",
        [],
        {"cause": "The adjoint holds the well indices fixed.", "corrected_value": 4.36e-4},
    )
    assert (await identified_well_index_cause()(s1, Target(""))).value == "C"
    close = investigation_value_close("corrected_value", [4.363e-4], 0.02)
    assert (await close(s1, Target(""))).value == "C"

    s2_right = _scenario_state(tmp_path / "s2a", [], {"ranking_reliable": False})
    s2_wrong = _scenario_state(tmp_path / "s2b", [], {"ranking_reliable": True})
    assert (await judged_ranking_unreliable()(s2_right, Target(""))).value == "C"
    assert (await judged_ranking_unreliable()(s2_wrong, Target(""))).value == "I"
