"""Sensitivity scenarios: investigation tasks S1-S3 on the reference case (RQ3).

Each scenario starts from an observation whose cause is known from the
manually verified reference analysis, and asks the agent to resolve it. The
prompts present the observation but not the competing explanations, which are
fixed in the report and used only for grading.

- S1: adjoint and finite differences disagree in the producer cell only.
  Known cause: the adjoint gradient holds the well indices fixed.
- S2: is cell 7 really more sensitive than cell 121? Known answer: no, they
  differ by less than the accuracy of the computation.
- S3: first-order prediction for doubling the permeability in a region.
  Known answer: right sign and magnitude, about 10 % too low.

The automatic scorers record the answer and which kinds of experiment appear
in the executed code. Whether the experiments were informative is judged with
the manual rubric. Reference values come from ``_sensitivity_reference/base``,
produced by ``jutuldarcy-experiments/scripts/05_task_references.jl``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from jutul_agent.eval.scorers import (
    julia_probe_count,
    no_interpreters_via_execute,
    tool_call_count,
    used_tools,
)
from jutul_agent.eval.sensitivity_scorers import (
    examined_well_indices,
    identified_well_index_cause,
    investigation_value_close,
    judged_ranking_unreliable,
    ran_perturbed_simulations,
    ran_step_size_sweep,
    ran_tightened_tolerances,
    repeated_adjoint,
)
from jutul_agent.eval.solver import jutul_agent_solver, load_eval_credentials

load_eval_credentials()

BASE = Path(__file__).parent / "_sensitivity_reference" / "base"
_SCENARIOS = json.loads((BASE / "scenarios.json").read_text(encoding="utf-8"))

_LIMITS = {"time_limit": 3600, "token_limit": 3_000_000, "message_limit": 160}

# Central difference in the producer cell at h = 1e-3 from the reference
# analysis (report Section 4.2.3); not recomputed by 05_task_references.jl.
_S1_FINITE_DIFFERENCE = 4.365e-4

_CASE = (
    "Using JutulDarcy, set up and run this exact two-phase reservoir case.\n"
    "- Grid: 20 x 20 x 1 cells over a 1000 x 1000 x 100 m domain.\n"
    "- Rock: uniform porosity 0.2 and uniform permeability 0.1 darcy.\n"
    "- Wells: a vertical injector at (1, 1) and a vertical producer at (20, 20).\n"
    "- Fluids: immiscible aqueous and vapor phases with reference densities "
    "1000 and 700 kg/m^3.\n"
    "- Initial state: 150 bar and fully aqueous-saturated.\n"
    "- Schedule: 24 steps of 30 days.\n"
    "- Injector: vapor injection on total-rate control corresponding to one pore "
    "volume injected over the full schedule.\n"
    "- Producer: 50 bar bottom-hole pressure.\n\n"
    "Define J as cumulative producer surface gas production normalized by the "
    "total injected volume, with positive J corresponding to positive physical "
    "production. Cells are numbered as in JutulDarcy, so the injector is in "
    "cell 1 and the producer in cell 400.\n\n"
)


def _output_spec(fields: str) -> str:
    return f"\nWrite `investigation_report.json` to the workspace with the fields {fields}.\n"


def _common_scorers() -> list:
    return [
        used_tools(["run_julia"]),
        no_interpreters_via_execute(),
        tool_call_count(),
        julia_probe_count(),
    ]


@task
def s1_well_cell_disagreement() -> Task:
    """S1: find why adjoint and finite differences disagree in the producer cell."""
    adjoint = _SCENARIOS["s1_adjoint_wi_fixed"]
    sample = Sample(
        id="s1-well-cell-disagreement",
        input=(
            _CASE + "A colleague computed the log-permeability sensitivity dJ/dlogK for "
            "this case with JutulDarcy's `reservoir_sensitivities` and checked it "
            "with central finite differences in log-permeability, rebuilding the "
            "case for each perturbation. The checks agree in cells 7, 121 and 394 "
            "to within 5e-5 relative to the largest component. In the producer "
            f"cell, cell 400, the adjoint gives {adjoint:.3e}, while the central "
            f"difference gives {_S1_FINITE_DIFFERENCE:.3e}.\n\n"
            "Find the cause of this disagreement, and explain how you established it.\n"
            + _output_spec(
                "`cause` (text), `evidence` (text describing the experiments you ran "
                "and what they showed) and `corrected_value` (the value of dJ/dlogK "
                "in cell 400 that you believe is correct, or null)"
            )
        ),
        metadata={
            "needs_env": True,
            "expected": (
                "the adjoint gradient holds the well indices fixed; the full "
                f"sensitivity in cell 400 is about {_SCENARIOS['s1_adjoint_full']:.4e}"
            ),
        },
    )
    return Task(
        dataset=[sample],
        solver=jutul_agent_solver("jutuldarcy"),
        scorer=[
            identified_well_index_cause(),
            investigation_value_close(
                "corrected_value", [_SCENARIOS["s1_adjoint_full"], _S1_FINITE_DIFFERENCE], 0.02
            ),
            ran_step_size_sweep(),
            ran_tightened_tolerances(),
            examined_well_indices(),
            ran_perturbed_simulations(2),
            *_common_scorers(),
        ],
        **_LIMITS,
    )


@task
def s2_ambiguous_ranking() -> Task:
    """S2: judge whether cell 7 can be ranked above cell 121."""
    first, second = _SCENARIOS["s2_cells"]
    sample = Sample(
        id="s2-ambiguous-ranking",
        input=(
            _CASE + "A colleague computed the log-permeability sensitivity dJ/dlogK for "
            f"this case and reports that cell {first} is the most sensitive cell, "
            f"ahead of cell {second}.\n\n"
            "Can this ranking be relied on? Investigate, and explain how you reached "
            "your conclusion.\n"
            + _output_spec(
                "`ranking_reliable` (true or false) and `evidence` (text describing "
                "the experiments you ran and what they showed)"
            )
        ),
        metadata={
            "needs_env": True,
            "expected": (
                f"no: cells {first} and {second} differ by "
                f"{_SCENARIOS['s2_scaled_difference']:.2e} of the largest component, "
                "below the accuracy of the computation; the case is symmetric about "
                "the diagonal through the wells"
            ),
        },
    )
    return Task(
        dataset=[sample],
        solver=jutul_agent_solver("jutuldarcy"),
        scorer=[
            judged_ranking_unreliable(),
            ran_tightened_tolerances(),
            repeated_adjoint(2),
            *_common_scorers(),
        ],
        **_LIMITS,
    )


@task
def s3_large_regional_change() -> Task:
    """S3: predict the effect of doubling the permeability in a region and judge it."""
    factors = _SCENARIOS["s3_factors"]
    i = factors.index(2.0)
    predicted_log = _SCENARIOS["s3_predicted"][i]
    # The same first-order prediction with delta theta = f - 1 instead of log f.
    predicted_linear = predicted_log / math.log(2.0) * (2.0 - 1.0)
    simulated = _SCENARIOS["s3_simulated"][i]
    sample = Sample(
        id="s3-large-regional-change",
        input=(
            _CASE + "Use JutulDarcy's adjoint functionality to predict the change in J if "
            "the permeability is doubled in every cell whose x-index and y-index are "
            "both at most 6, and judge how far the prediction can be trusted.\n"
            + _output_spec(
                "`predicted_change` (number), `simulated_change` (number, or null if "
                "you did not simulate the change), `assessment` (text: how far the "
                "prediction can be trusted) and `evidence` (text describing the "
                "experiments you ran and what they showed)"
            )
        ),
        metadata={
            "needs_env": True,
            "expected": (
                f"first-order prediction about {predicted_log:.3e} (log form); simulated "
                f"change about {simulated:.3e}; the prediction is about 10 % too low"
            ),
        },
    )
    return Task(
        dataset=[sample],
        solver=jutul_agent_solver("jutuldarcy"),
        scorer=[
            investigation_value_close("predicted_change", [predicted_log, predicted_linear], 0.01),
            investigation_value_close("simulated_change", [simulated], 0.01),
            ran_perturbed_simulations(2),
            *_common_scorers(),
        ],
        **_LIMITS,
    )


TASKS = [s1_well_cell_disagreement, s2_ambiguous_ranking, s3_large_regional_change]
