"""Sensitivity suite: adjoint permeability sensitivities graded against verified references.

Six tasks (T1-T6) test whether the agent computes, verifies and interprets an
adjoint gradient reliably. Each task's correct answer comes from the manually
verified reference workflow, stored under ``_sensitivity_reference/`` and
produced by ``jutuldarcy-experiments/scripts/05_task_references.jl``. Never
edit a reference by hand or re-capture it until a run passes.

- T1 reference: the fully specified case of the report, Section 4.2.
- T2 held-out: a heterogeneous permeability field and different well positions.
- T3 sign trap: as T1, but the prompt leaves the sign convention implicit.
- T4 stale-model trap: a provided helper reuses the base case's static
  parameters, so perturbed runs do not see the perturbation.
- T5 planted inconsistency: a provided run function uses loose solver
  tolerances, so the gradient and finite differences disagree.
- T6 interpretation: a first-order prediction for a regional permeability change.

The prompts specify the case, the objective and the output files, but not the
verification procedure, so that the comparison with and without the
sensitivity skill measures what the agent does on its own initiative.
"""

from __future__ import annotations

import json
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
    claims_validated,
    executed_code_matches,
    gradient_correct,
    gradient_written_by_code,
    julia_calls_matching,
    objective_correct,
    predicted_change_close,
    reliable_outcome,
    top_cell_accepted,
)
from jutul_agent.eval.solver import jutul_agent_solver, load_eval_credentials
from jutul_agent.eval.tasks._sensitivity_case import (
    FLUIDS,
    SIMULATION,
    SIMULATION_KWARGS,
    SIMULATION_STEPPING,
)

load_eval_credentials()

REFERENCE_ROOT = Path(__file__).parent / "_sensitivity_reference"
BASE = REFERENCE_ROOT / "base"
HELDOUT = REFERENCE_ROOT / "heldout"
LOOSE = REFERENCE_ROOT / "loose"

# The T6 region (x and y index at most this value), chosen in the reference script.
_REGION_MAX = int(json.loads((BASE / "reference.json").read_text(encoding="utf-8"))["region_x_max"])

_LIMITS = {"time_limit": 3600, "token_limit": 3_000_000, "message_limit": 160}

# ---------------------------------------------------------------------------
# Prompt building blocks.

_BASE_CASE = (
    "- Grid: 20 x 20 x 1 cells over a 1000 x 1000 x 100 m domain.\n"
    "- Rock: uniform porosity 0.2 and uniform permeability 0.1 darcy.\n"
    "- Wells: a vertical injector at (1, 1) and a vertical producer at (20, 20).\n"
    + FLUIDS
    + "- Initial state: 150 bar and fully aqueous-saturated.\n"
    "- Schedule: 24 steps of 30 days.\n"
    "- Injector: vapor injection on total-rate control corresponding to one pore "
    "volume injected over the full schedule.\n"
    "- Producer: 50 bar bottom-hole pressure.\n" + SIMULATION
)

_J_WITH_SIGN = (
    "Define J as cumulative producer surface gas production normalized by the "
    "total injected volume, with positive J corresponding to positive physical "
    "production.\n"
)

_J_WITHOUT_SIGN = (
    "Define J as cumulative producer surface gas production normalized by the "
    "total injected volume.\n"
)

_QUESTION = (
    "Which cells' permeability has the largest influence on J? Use JutulDarcy's "
    "adjoint functionality to compute the sensitivity dJ/dlogK for every cell, "
    "and tell me how far the result can be trusted.\n"
)


def _output_spec(extra: str = "") -> str:
    return (
        "\nWrite two files to the workspace:\n"
        "- `dJ_dlogK.csv` with the header `cell,dJ_dlogK` and one row per cell, "
        "using JutulDarcy's cell numbering.\n"
        "- `sensitivity_report.json` with the fields `J` (number), "
        "`most_sensitive_cells` (list of cell numbers, most sensitive first), "
        "`validated` (true if you consider the reported gradient verified, "
        "false otherwise), `checks` (list describing any checks you performed) "
        f"and `limitations` (text){extra}.\n"
    )


def _scorers(ref_dir: Path) -> list:
    """The scorers every gradient task shares."""
    return [
        reliable_outcome(ref_dir),
        gradient_correct(ref_dir),
        claims_validated(),
        gradient_written_by_code(),
        objective_correct(ref_dir),
        top_cell_accepted(ref_dir),
        executed_code_matches(r"(reservoir_sensitivities|solve_adjoint_sensitivities)\s*\("),
        julia_calls_matching(r"simulate_reservoir\s*\(|run_case\s*\(", 2),
        used_tools(["run_julia"]),
        no_interpreters_via_execute(),
        tool_call_count(),
        julia_probe_count(),
    ]


def _expected(ref_dir: Path) -> str:
    """A short ground-truth note recorded in the trace for later review."""
    reference = json.loads((ref_dir / "reference.json").read_text(encoding="utf-8"))
    return (
        f"J about {reference['J']:.6g}; most sensitive cells {reference['accepted_top_cells']}; "
        "gradient should include the well-index term or state that it is held fixed"
    )


# ---------------------------------------------------------------------------
# Julia fixtures for T4 and T5: the base case as a colleague's setup script.

_CASE_SETUP_COMMON = """\
using Jutul
using JutulDarcy

darcy, kg, meter, day, bar = si_units(:darcy, :kilogram, :meter, :day, :bar)

nx, ny, nz = 20, 20, 1
grid = CartesianMesh((nx, ny, nz), (1000.0, 1000.0, 100.0) .* meter)
nc = number_of_cells(grid)

K_base = fill(0.1 * darcy, nc)
dt = fill(30.0 * day, 24)
total_time = sum(dt)

rho_water = 1000.0 * kg / meter^3
rho_gas = 700.0 * kg / meter^3
system = ImmiscibleSystem(
    (AqueousPhase(), VaporPhase()), reference_densities = (rho_water, rho_gas)
)
densities = ConstantCompressibilityDensities(
    p_ref = 1.0 * bar, density_ref = [rho_water, rho_gas], compressibility = [1e-6, 1e-4] ./ bar
)

function build_case(K)
    reservoir = reservoir_domain(grid, permeability = K, porosity = 0.2)
    injector = setup_vertical_well(reservoir, 1, 1, name = :Injector)
    producer = setup_vertical_well(reservoir, nx, ny, name = :Producer)
    model, parameters = setup_reservoir_model(
        reservoir, system, wells = [injector, producer], extra_out = true
    )
    replace_variables!(model, PhaseMassDensities = densities)
    state0 = setup_reservoir_state(model, Pressure = 150.0 * bar, Saturations = [1.0, 0.0])
    inj_rate = sum(pore_volume(model, parameters)) / total_time
    controls = Dict(
        :Injector => InjectorControl(TotalRateTarget(inj_rate), [0.0, 1.0], density = rho_gas),
        :Producer => ProducerControl(BottomHolePressureTarget(50.0 * bar)),
    )
    forces = setup_reservoir_forces(model, control = controls)
    case = JutulCase(model, dt, forces, parameters = parameters, state0 = state0)
    return case, inj_rate
end

case_base, inj_rate = build_case(K_base)

# J: cumulative producer surface gas production normalized by the injected
# volume, positive for positive physical production.
function objective_function(model, state, dt_step, step_i, forces)
    gas_rate = JutulDarcy.compute_well_qoi(model, state, forces, :Producer, SurfaceGasRateTarget)
    return -dt_step * gas_rate / (inj_rate * total_time)
end

objective_value(case, result) = Jutul.evaluate_objective(objective_function, case, result.result)
"""

_STALE_HELPER = """
# Helper for perturbation studies: J with the permeability of cell i scaled by
# `factor`. Reuses the precomputed static parameters of the base case to save
# setup time.
function J_with_scaled_permeability(i, factor)
    K = copy(K_base)
    K[i] *= factor
    case, _ = build_case(K)
    case = JutulCase(case.model, case.dt, case.forces,
                     parameters = case_base.parameters, state0 = case.state0)
    return objective_value(case, run_case(case))
end
"""


def _run_case(kwargs: str) -> str:
    """``run_case``, the setup script's standard way of running a simulation."""
    return (
        "\n# The project's standard way of running a simulation.\n"
        "run_case(case) = simulate_reservoir(\n"
        f"    case; output_substates = true, info_level = -1, {kwargs}\n"
        ")\n"
    )


def _loose_run_case() -> str:
    """``run_case`` with the loose tolerances chosen by the reference script.

    The time stepping is as specified for the case, so the tolerances are the
    only error.
    """
    tolerances = json.loads((LOOSE / "tolerances.json").read_text(encoding="utf-8"))
    loose = ", ".join(f"{name} = {value!r}" for name, value in tolerances.items())
    return _run_case(f"{SIMULATION_STEPPING}, {loose}")


# ---------------------------------------------------------------------------
# Tasks.


@task
def t1_reference() -> Task:
    """T1: the fully specified reference case."""
    sample = Sample(
        id="t1-reference",
        input=(
            "Using JutulDarcy, set up and run this exact two-phase reservoir case.\n"
            + _BASE_CASE
            + "\n"
            + _J_WITH_SIGN
            + "\n"
            + _QUESTION
            + _output_spec()
        ),
        metadata={"needs_env": True, "expected": _expected(BASE)},
    )
    return Task(
        dataset=[sample], solver=jutul_agent_solver("jutuldarcy"), scorer=_scorers(BASE), **_LIMITS
    )


@task
def t2_heldout() -> Task:
    """T2: heterogeneous permeability and different wells, never seen while writing the skill."""
    permeability = (HELDOUT / "permeability_darcy.csv").read_text(encoding="utf-8")
    sample = Sample(
        id="t2-heldout",
        input=(
            "Using JutulDarcy, set up and run this exact two-phase reservoir case.\n"
            "- Grid: 20 x 20 x 1 cells over a 1000 x 1000 x 100 m domain.\n"
            "- Rock: uniform porosity 0.2. The permeability of each cell is given "
            "in darcy in `permeability_darcy.csv`, with one row per cell in "
            "JutulDarcy's cell numbering.\n"
            "- Wells: a vertical injector at (1, 20) and a vertical producer at (20, 1).\n"
            + FLUIDS
            + "- Initial state: 150 bar and fully aqueous-saturated.\n"
            "- Schedule: 24 steps of 30 days.\n"
            "- Injector: vapor injection on total-rate control corresponding to one "
            "pore volume injected over the full schedule.\n"
            "- Producer: 50 bar bottom-hole pressure.\n"
            + SIMULATION
            + "\n"
            + _J_WITH_SIGN
            + "\n"
            + _QUESTION
            + _output_spec()
        ),
        metadata={
            "needs_env": True,
            "fixtures": {"permeability_darcy.csv": permeability},
            "expected": _expected(HELDOUT),
        },
    )
    return Task(
        dataset=[sample],
        solver=jutul_agent_solver("jutuldarcy"),
        scorer=_scorers(HELDOUT),
        **_LIMITS,
    )


@task
def t3_sign_trap() -> Task:
    """T3: the reference case with the sign convention left implicit."""
    sample = Sample(
        id="t3-sign-trap",
        input=(
            "Using JutulDarcy, set up and run this exact two-phase reservoir case.\n"
            + _BASE_CASE
            + "\n"
            + _J_WITHOUT_SIGN
            + "\n"
            + _QUESTION
            + _output_spec()
        ),
        metadata={"needs_env": True, "expected": _expected(BASE)},
    )
    return Task(
        dataset=[sample], solver=jutul_agent_solver("jutuldarcy"), scorer=_scorers(BASE), **_LIMITS
    )


@task
def t4_stale_model() -> Task:
    """T4: a provided helper silently reuses the base case's static parameters."""
    sample = Sample(
        id="t4-stale-model",
        input=(
            "A colleague has set up a JutulDarcy case in `case_setup.jl`, including "
            "the objective J, `run_case`, the project's standard way of running a "
            "simulation, and a helper for perturbation studies. Include the script, "
            "work from it and use `run_case` for the simulations.\n\n" + _QUESTION + _output_spec()
        ),
        metadata={
            "needs_env": True,
            "fixtures": {
                "case_setup.jl": _CASE_SETUP_COMMON + _run_case(SIMULATION_KWARGS) + _STALE_HELPER
            },
            "expected": _expected(BASE),
        },
    )
    return Task(
        dataset=[sample], solver=jutul_agent_solver("jutuldarcy"), scorer=_scorers(BASE), **_LIMITS
    )


@task
def t5_loose_tolerance() -> Task:
    """T5: the provided run function uses loose solver tolerances."""
    sample = Sample(
        id="t5-loose-tolerance",
        input=(
            "A colleague has set up a JutulDarcy case in `case_setup.jl`, including "
            "the objective J and `run_case`, the project's standard way of running "
            "a simulation. Include the script and use `run_case` for the "
            "simulations.\n\n" + _QUESTION + _output_spec()
        ),
        metadata={
            "needs_env": True,
            "fixtures": {"case_setup.jl": _CASE_SETUP_COMMON + _loose_run_case()},
            "expected": _expected(BASE),
        },
    )
    return Task(
        dataset=[sample], solver=jutul_agent_solver("jutuldarcy"), scorer=_scorers(BASE), **_LIMITS
    )


@task
def t6_interpretation() -> Task:
    """T6: a first-order prediction for a regional permeability change."""
    sample = Sample(
        id="t6-interpretation",
        input=(
            "Using JutulDarcy, set up and run this exact two-phase reservoir case.\n"
            + _BASE_CASE
            + "\n"
            + _J_WITH_SIGN
            + "\nUse JutulDarcy's adjoint functionality to predict, to first order, "
            "the change in J if the permeability is increased by 10 % in every cell "
            f"whose x-index and y-index are both at most {_REGION_MAX}. Explain the "
            "prediction and how far it can be trusted.\n"
            + _output_spec(
                ", and additionally `predicted_change` (the first-order prediction of "
                "the change in J)"
            )
        ),
        metadata={"needs_env": True, "expected": _expected(BASE)},
    )
    return Task(
        dataset=[sample],
        solver=jutul_agent_solver("jutuldarcy"),
        scorer=[predicted_change_close(BASE), *_scorers(BASE)],
        **_LIMITS,
    )


TASKS = [
    t1_reference,
    t2_heldout,
    t3_sign_trap,
    t4_stale_model,
    t5_loose_tolerance,
    t6_interpretation,
]
