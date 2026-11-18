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
  ``eps``.
- Tolerances default to ``tau = 10 * eps`` with ``eps = max(eps_ref, 1e-4)``,
  where ``eps_ref`` is the accuracy of the reference estimated from repeated
  and tightened runs. The floor allows for valid choices, such as linear
  solver settings, whose effect exceeds the reference's own accuracy. The
  reference script writes ``tau``; older references fall back to
  ``10 * eps_ref``.
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
INVESTIGATION_FILE = "investigation_report.json"
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
    if tol is not None:
        return tol
    if "tau" in reference:
        return float(reference["tau"])
    return 10.0 * float(reference["eps_ref"])


def _workspace(state: TaskState) -> Path | None:
    root = state.store.get(STORE_WORKSPACE)
    return Path(root) if root else None


def _read_json(state: TaskState, name: str) -> dict[str, Any] | None:
    """A JSON object from the workspace, or ``None`` when missing or malformed."""
    root = _workspace(state)
    if root is None or not (root / name).exists():
        return None
    try:
        data = json.loads((root / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_report(state: TaskState) -> dict[str, Any] | None:
    """The agent's JSON report, or ``None`` when it is missing or malformed."""
    return _read_json(state, REPORT_FILE)


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


# ---------------------------------------------------------------------------
# Executed Julia code.
#
# Agents often write their code to ``.jl`` files and run them with
# ``include`` in the session, or (against the guidance) with ``julia file.jl``
# in the shell. A check that read only the code typed into ``run_julia`` would
# miss those experiments, so each executed call is expanded with the latest
# contents of the files it runs, as written or edited through the file tools.
# Files the agent did not write, such as task fixtures, are read from the
# workspace.

_INCLUDE = re.compile(r"""include\(\s*["']([^"']+)["']""")
_JULIA_SCRIPT = re.compile(r"""\bjulia\b[^\n;&|]*?([\w./-]+\.jl)""")


def _arg(args: dict[str, Any], *names: str) -> str:
    for name in names:
        if args.get(name) is not None:
            return str(args[name])
    return ""


def _expand(code: str, files: dict[str, str], root: Path | None, depth: int = 0) -> str:
    """``code`` followed by the contents of the files it includes, recursively."""
    if depth > 3:
        return code
    parts = [code]
    for path in _INCLUDE.findall(code):
        name = Path(path).name
        included = files.get(name)
        if included is None and root is not None and (root / path).is_file():
            try:
                included = (root / path).read_text(encoding="utf-8")
            except OSError:
                included = None
        if included:
            parts.append(_expand(included, files, root, depth + 1))
    return "\n".join(parts)


def executed_julia_code(state: TaskState) -> list[str]:
    """The code of each executed Julia call, with the files it runs expanded."""
    root = _workspace(state)
    files: dict[str, str] = {}
    executed: list[str] = []
    for event in _trace_events(state):
        if event.kind != schema.TOOL_CALL:
            continue
        name = event.payload.get("name")
        args = event.payload.get("args") or {}
        path = Path(_arg(args, "file_path", "path")).name
        if name == "write_file" and path:
            files[path] = _arg(args, "content", "text")
        elif name == "edit_file" and path in files:
            old, new = _arg(args, "old_string", "old_str"), _arg(args, "new_string", "new_str")
            if old:
                count = -1 if args.get("replace_all") else 1
                files[path] = files[path].replace(old, new, count)
        elif name in ("run_julia", "plot_julia"):
            executed.append(_expand(_arg(args, "code"), files, root))
        elif name == "execute":
            scripts = _JULIA_SCRIPT.findall(_arg(args, "command"))
            if scripts:
                code = "\n".join(f'include("{script}")' for script in scripts)
                executed.append(_expand(code, files, root))
    return executed


def _code_hits(state: TaskState, matches: Any) -> int:
    """Number of executed Julia calls for which ``matches(code)`` holds."""
    return sum(1 for code in executed_julia_code(state) if matches(code))


def _code_check(pattern: Any, what: str, min_calls: int = 1) -> Scorer:
    """Pass when at least ``min_calls`` executed calls match a regex or predicate."""
    matches = re.compile(pattern).search if isinstance(pattern, str) else pattern

    async def score(state: TaskState, target: Target) -> Score:
        count = _code_hits(state, matches)
        return Score(
            value=CORRECT if count >= min_calls else INCORRECT,
            explanation=f"{what}: {count} executed Julia calls matched, need {min_calls}",
        )

    return score


@scorer(metrics=[accuracy()])
def executed_code_matches(pattern: str) -> Scorer:
    """Pass when executed Julia code, including the files it runs, matches ``pattern``."""
    return _code_check(pattern, f"pattern {pattern!r}")


@scorer(metrics=[accuracy()])
def julia_calls_matching(pattern: str, min_calls: int) -> Scorer:
    """Pass when at least ``min_calls`` executed Julia calls contain ``pattern``.

    A weak trace check, for example that simulations were run beyond the base
    case. A loop inside one call counts once, so the manual rubric remains the
    judge of whether a verification was actually carried out.
    """
    return _code_check(pattern, f"pattern {pattern!r}", min_calls)


# ---------------------------------------------------------------------------
# Investigation scenarios (S1-S3): which experiments were run, and the answer.
#
# The trace checks are pattern matches on the Julia code the agent executed.
# They record whether a kind of experiment appears in the trace, not whether
# it was set up correctly; the manual rubric judges that.

# A literal list or tuple of three or more small numbers, or a power range such
# as ``10.0 .^ (-1:-1:-6)``: the usual shapes of a sweep over step sizes.
_SMALL = r"(?:\d*\.?\d+e-\d+|0\.0*[1-9]\d*|10\.?0?\s*\^\s*-?\(?-?\d+\)?)"
_STEP_SWEEP = (
    rf"[\[(]\s*{_SMALL}\s*(?:,\s*{_SMALL}\s*){{2,}}[\])]"
    r"|10\.?0?\s*\.\^\s*\(?\s*-?\d"
)
_WELL_INDICES = r"WellIndices|solve_adjoint_sensitivities"
_SIMULATE = r"simulate_reservoir\s*\(|run_case\s*\("
_NUMBER = r"(\d*\.?\d+(?:[eE][-+]?\d+)?)"


def _tighter_than(tol_cnv: float, tol_mb: float) -> Any:
    """Predicate: the code sets ``tol_cnv`` or ``tol_mb`` below the given values."""
    cnv = re.compile(rf"\btol_cnv\s*=\s*{_NUMBER}")
    mb = re.compile(rf"\btol_mb\s*=\s*{_NUMBER}")

    def matches(code: str) -> bool:
        return any(float(v) < 0.99 * tol_cnv for v in cnv.findall(code)) or any(
            float(v) < 0.99 * tol_mb for v in mb.findall(code)
        )

    return matches


@scorer(metrics=[accuracy()])
def ran_step_size_sweep() -> Scorer:
    """Whether the executed code contains a sweep over perturbation sizes."""
    return _code_check(_STEP_SWEEP, "step-size sweep")


@scorer(metrics=[accuracy()])
def ran_tightened_tolerances(tol_cnv: float = 1e-6, tol_mb: float = 1e-10) -> Scorer:
    """Whether the executed code sets tolerances tighter than the case's own.

    The case itself specifies ``tol_cnv = 1e-6`` and ``tol_mb = 1e-10``, so
    setting those values does not count as tightening.
    """
    return _code_check(_tighter_than(tol_cnv, tol_mb), "tolerances tighter than the case")


@scorer(metrics=[accuracy()])
def examined_well_indices() -> Scorer:
    """Whether the executed code touches the well indices or their sensitivities."""
    return _code_check(_WELL_INDICES, "well indices")


@scorer(metrics=[accuracy()])
def ran_perturbed_simulations(min_calls: int = 2) -> Scorer:
    """Whether simulations were run in at least ``min_calls`` Julia calls."""
    return _code_check(_SIMULATE, "simulations", min_calls)


@scorer(metrics=[accuracy()])
def repeated_adjoint(min_calls: int = 2) -> Scorer:
    """Whether the adjoint was computed in at least ``min_calls`` Julia calls."""
    return _code_check(
        r"(?:reservoir_sensitivities|solve_adjoint_sensitivities)\s*\(",
        "adjoint computations",
        min_calls,
    )


@scorer(metrics=[accuracy()])
def identified_well_index_cause() -> Scorer:
    """S1: whether the reported cause names the well indices."""

    async def score(state: TaskState, target: Target) -> Score:
        report = _read_json(state, INVESTIGATION_FILE) or {}
        cause = str(report.get("cause", ""))
        ok = bool(_WI_FIXED_STATEMENT.search(cause))
        return Score(value=CORRECT if ok else INCORRECT, explanation=f"cause: {cause[:200]!r}")

    return score


@scorer(metrics=[accuracy()])
def judged_ranking_unreliable() -> Scorer:
    """S2: whether the agent concludes that the ranking cannot be relied on."""

    async def score(state: TaskState, target: Target) -> Score:
        report = _read_json(state, INVESTIGATION_FILE) or {}
        value = report.get("ranking_reliable")
        return Score(
            value=CORRECT if value is False else INCORRECT,
            explanation=f"ranking_reliable = {value!r}, known answer false",
        )

    return score


@scorer(metrics=[accuracy()])
def investigation_value_close(key: str, accepted: list[float], rel_tol: float) -> Scorer:
    """Whether a number in the investigation report matches an accepted value."""

    async def score(state: TaskState, target: Target) -> Score:
        report = _read_json(state, INVESTIGATION_FILE) or {}
        try:
            value = float(report[key])
        except (KeyError, TypeError, ValueError):
            value = math.nan
        ok = math.isfinite(value) and any(abs(value - a) <= rel_tol * abs(a) for a in accepted)
        return Score(
            value=CORRECT if ok else INCORRECT,
            explanation=f"{key} = {value}, accepted {accepted} (rel. tol {rel_tol})",
        )

    return score
