"""Case description shared by the sensitivity suite and the scenarios.

Both lines match the reference (``05_task_references.jl``) and are stated in
full, since JutulDarcy's defaults would otherwise change the discrete J:

- ``ImmiscibleSystem`` uses 1000 kg/m^3 for both phases in the reservoir unless
  a density model is given, and treats the given densities as surface
  densities only.
- The time steps define the discrete problem. With JutulDarcy's adaptive
  time stepping the case takes 42 ministeps and J differs by about 6e-3
  relative from the case with the report steps as time steps. With the
  default limit of 15 Newton iterations, steps are cut (32 instead of 26
  ministeps), J differs by about 2e-3, and finite differences jump. With
  ``timesteps = :none`` and 50 iterations, every run in the reference work
  takes the same 26 ministeps.

The default nonlinear tolerances are accurate enough for this case (J within
about 7e-7 of a tightly converged run), so the prompts leave them at their
defaults.
"""

FLUIDS = (
    "- Fluids: immiscible aqueous and vapor phases with densities 1000 and "
    "700 kg/m^3 at 1 bar and constant compressibilities 1e-6 and 1e-4 1/bar; "
    "the same densities are the surface densities. Use JutulDarcy's default "
    "relative permeabilities and viscosities.\n"
)

SIMULATION = (
    "- Simulation: use the 24 report steps as time steps without adaptive "
    "substepping (`timesteps = :none` in `simulate_reservoir`, so a step is only "
    "cut if Newton fails to converge), JutulDarcy's default nonlinear tolerances, "
    "and at most 50 Newton iterations per step (`max_nonlinear_iterations = 50`).\n"
)

# The same settings as Julia keyword arguments, for the provided setup scripts.
SIMULATION_STEPPING = "timesteps = :none, max_nonlinear_iterations = 50"
SIMULATION_KWARGS = SIMULATION_STEPPING
