"""Scorers for the sensitivity suite.

The sensitivity tasks ask the agent to leave two files in the workspace, a
gradient file and a JSON report, so the result is graded as data rather than
mined out of prose:

- ``dJ_dlogK.csv``: header ``cell,dJ_dlogK``, one row per reservoir cell in
  JutulDarcy's cell numbering.
- ``sensitivity_report.json``: ``J``, ``most_sensitive_cells``, ``validated``,
  ``checks``, ``limitations`` and, for the region task, ``predicted_change``.

Every task has a reference directory with ``reference.json`` and
``gradient_logK.csv``, produced by the manually verified reference workflow
(``jutuldarcy-experiments/scripts/05_task_references.jl``). The reference
gradient comes in two versions: ``dJ_dlogK_full`` includes the well-index
term and ``dJ_dlogK_wi_fixed`` holds the well indices fixed, which is what
``reservoir_sensitivities`` returns.

Scoring rules, fixed before any runs:

- A gradient is correct if it matches the full reference, or if it matches the
  well-index-fixed reference and the agent states that well indices are held
  fixed.
- The most sensitive cell is accepted if it is any cell in the reference's
  ``accepted_top_cells``, the cells that tie with the largest component within
  the accuracy of the reference.
- Tolerances default to ``10 * eps_ref``, where ``eps_ref`` is the accuracy of
  the reference estimated from repeated and tightened runs.
- A run is reliable if the gradient is correct and the agent claims it is
  validated, or if the gradient is incorrect and the agent does not claim so.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState

from jutul_agent.eval.scorers import _trace_events
from jutul_agent.eval.solver import STORE_WORKSPACE
from jutul_agent.trace import schema

REPORT_FILE = "sensitivity_report.json"
GRADIENT_FILE = "dJ_dlogK.csv"

# A statement that the gradient holds the well indices fixed.
_WI_FIXED_STATEMENT = re.compile(r"well[\s_-]*ind", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Loading the reference and the agent's files.


def load_reference(ref_dir: Path) -> dict[str, Any]:
    """The reference values and both reference gradients for one task."""
    reference = json.loads((ref_dir / "reference.json").read_text(encoding="utf-8"))
    full: dict[int, float] = {}
    wi_fixed: dict[int, float] = {}
    with (ref_dir / "gradient_logK.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cell = int(float(row["cell"]))
            full[cell] = float(row["dJ_dlogK_full"])
            wi_fixed[cell] = float(row["dJ_dlogK_wi_fixed"])
    reference["gradient_full"] = full
    reference["gradient_wi_fixed"] = wi_fixed
    return reference


def _tolerance(reference: dict[str, Any], tol: float | None) -> float:
    return tol if tol is not None else 10.0 * float(reference["eps_ref"])


def _workspace(state: TaskState) -> Path | None:
    root = state.store.get(STORE_WORKSPACE)
    return Path(root) if root else None


def read_report(state: TaskState) -> dict[str, Any] | None:
    """The agent's JSON report, or ``None`` when it is missing or malformed."""
    root = _workspace(state)
    if root is None or not (root / REPORT_FILE).exists():
        return None
    try:
        data = json.loads((root / REPORT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_gradient(state: TaskState) -> dict[int, float] | None:
    """The agent's gradient file as cell -> value, tolerant of header naming.

    The first column is taken as the cell number and the second as the value,
    so a header such as ``cell,dJ/dlogK`` still parses.
    """
    root = _workspace(state)
    if root is None or not (root / GRADIENT_FILE).exists():
        return None
    values: dict[int, float] = {}
    try:
        with (root / GRADIENT_FILE).open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except OSError:
        return None
    for row in rows[1:]:
        if len(row) < 2:
            continue
        try:
            values[int(float(row[0]))] = float(row[1])
        except ValueError:
            continue
    return values or None


def _relative_l2(agent: dict[int, float], ref: dict[int, float]) -> float:
    """Relative 2-norm error over the reference cells; inf if any cell is missing."""
    if any(cell not in agent for cell in ref):
        return math.inf
    diff = math.sqrt(sum((agent[c] - ref[c]) ** 2 for c in ref))
    norm = math.sqrt(sum(v**2 for v in ref.values()))
    return diff / norm if norm > 0 else math.inf


def _states_wi_fixed(state: TaskState, report: dict[str, Any] | None) -> bool:
    """Whether the report or the final answer says the well indices are held fixed."""
    text = state.output.completion if state.output else ""
    if report:
        text += " " + str(report.get("limitations", "")) + " " + str(report.get("checks", ""))
    return bool(_WI_FIXED_STATEMENT.search(text))


def gradient_verdict(
    state: TaskState, reference: dict[str, Any], tol: float | None
) -> tuple[bool, str]:
    """Whether the agent's gradient is correct under the suite's rules, with a reason."""
    gradient = read_gradient(state)
    if gradient is None:
        return False, f"no readable {GRADIENT_FILE}"
    tau = _tolerance(reference, tol)
    err_full = _relative_l2(gradient, reference["gradient_full"])
    err_fixed = _relative_l2(gradient, reference["gradient_wi_fixed"])
    errors = f"rel. L2 error vs full {err_full:.2e}, vs WI fixed {err_fixed:.2e}, tol {tau:.1e}"
    if err_full <= tau:
        return True, f"matches the full gradient ({errors})"
    if err_fixed <= tau:
        if _states_wi_fixed(state, read_report(state)):
            return True, f"matches the WI-fixed gradient and says so ({errors})"
        return False, f"matches the WI-fixed gradient without saying so ({errors})"
    return False, f"matches neither reference ({errors})"


def _claims_validated(report: dict[str, Any] | None) -> bool:
    return bool(report) and report.get("validated") is True


# ---------------------------------------------------------------------------
# Scorers.


@scorer(metrics=[accuracy()])
def gradient_correct(ref_dir: Path, tol: float | None = None) -> Scorer:
    """Pass when the gradient file is correct under the suite's rules."""
    reference = load_reference(ref_dir)

    async def score(state: TaskState, target: Target) -> Score:
        ok, reason = gradient_verdict(state, reference, tol)
        return Score(value=CORRECT if ok else INCORRECT, explanation=reason)

    return score


@scorer(metrics=[accuracy()])
def objective_correct(ref_dir: Path, tol: float | None = None) -> Scorer:
    """Pass when the reported ``J`` is within the relative tolerance of the reference.

    A wrong sign fails, so this is also the sign check.
    """
    reference = load_reference(ref_dir)

    async def score(state: TaskState, target: Target) -> Score:
        report = read_report(state)
        tau = _tolerance(reference, tol)
        expected = float(reference["J"])
        try:
            value = float(report["J"]) if report else math.nan
        except (KeyError, TypeError, ValueError):
            value = math.nan
        ok = math.isfinite(value) and abs(value - expected) <= tau * abs(expected)
        return Score(
            value=CORRECT if ok else INCORRECT,
            explanation=f"J reported {value}, reference {expected}, rel. tol {tau:.1e}",
        )

    return score


@scorer(metrics=[accuracy()])
def top_cell_accepted(ref_dir: Path) -> Scorer:
    """Pass when the first listed most sensitive cell is one of the accepted cells."""
    reference = load_reference(ref_dir)
    accepted = {int(c) for c in reference["accepted_top_cells"]}

    async def score(state: TaskState, target: Target) -> Score:
        report = read_report(state)
        cells = (report or {}).get("most_sensitive_cells") or []
        try:
            first = int(cells[0])
        except (IndexError, TypeError, ValueError):
            first = None
        return Score(
            value=CORRECT if first in accepted else INCORRECT,
            explanation=f"first listed cell {first}, accepted {sorted(accepted)}",
        )

    return score


@scorer(metrics=[accuracy()])
def claims_validated() -> Scorer:
    """Record whether the agent claims the gradient is validated (not a grade).

    Read together with :func:`gradient_correct`, this gives the outcome table:
    correct and claimed, correct and not claimed (over-cautious), incorrect and
    claimed (silent failure), incorrect and not claimed (correct detection).
    """

    async def score(state: TaskState, target: Target) -> Score:
        report = read_report(state)
        claimed = _claims_validated(report)
        return Score(
            value=CORRECT if claimed else INCORRECT,
            explanation="claims validated" if claimed else "does not claim validated",
        )

    return score


@scorer(metrics=[accuracy()])
def reliable_outcome(ref_dir: Path, tol: float | None = None) -> Scorer:
    """The headline grade: correct and claimed, or incorrect and not claimed.

    A run that leaves no gradient file or report has no result and fails. The
    explanation names the outcome category. Whether a claimed verification
    was actually executed and valid is judged by the manual rubric.
    """
    reference = load_reference(ref_dir)

    async def score(state: TaskState, target: Target) -> Score:
        report = read_report(state)
        if report is None or read_gradient(state) is None:
            return Score(
                value=INCORRECT,
                explanation=f"no result: missing or unreadable {REPORT_FILE} or {GRADIENT_FILE}",
            )
        correct, reason = gradient_verdict(state, reference, tol)
        claimed = _claims_validated(report)
        if correct and claimed:
            category, ok = "reliable", True
        elif correct:
            category, ok = "over-cautious", False
        elif claimed:
            category, ok = "silent failure", False
        else:
            category, ok = "correct detection", True
        return Score(value=CORRECT if ok else INCORRECT, explanation=f"{category}: {reason}")

    return score


@scorer(metrics=[accuracy()])
def predicted_change_close(ref_dir: Path, rel_tol: float = 0.01) -> Scorer:
    """Pass when ``predicted_change`` matches a first-order reference prediction.

    Both ``log(1.1) * sum`` and ``0.1 * sum`` are valid first-order predictions
    for a 10 % increase, so either reference value is accepted.
    """
    reference = load_reference(ref_dir)
    options = [float(reference[k]) for k in ("predicted_change_log", "predicted_change_linear")]

    async def score(state: TaskState, target: Target) -> Score:
        report = read_report(state)
        try:
            value = float(report["predicted_change"]) if report else math.nan
        except (KeyError, TypeError, ValueError):
            value = math.nan
        ok = math.isfinite(value) and any(abs(value - ref) <= rel_tol * abs(ref) for ref in options)
        return Score(
            value=CORRECT if ok else INCORRECT,
            explanation=f"predicted change {value}, accepted {options} (rel. tol {rel_tol})",
        )

    return score


@scorer(metrics=[accuracy()])
def julia_calls_matching(pattern: str, min_calls: int) -> Scorer:
    """Pass when at least ``min_calls`` Julia tool calls contain ``pattern``.

    A weak trace check, for example that simulations were run beyond the base
    case. A loop inside one call counts once, so the manual rubric remains the
    judge of whether a verification was actually carried out.
    """
    compiled = re.compile(pattern)

    async def score(state: TaskState, target: Target) -> Score:
        count = sum(
            1
            for event in _trace_events(state)
            if event.kind == schema.TOOL_CALL
            and event.payload.get("name") in ("run_julia", "plot_julia")
            and compiled.search(str((event.payload.get("args") or {}).get("code", "")))
        )
        return Score(
            value=CORRECT if count >= min_calls else INCORRECT,
            explanation=f"{count} Julia calls matched {pattern!r}, need {min_calls}",
        )

    return score
