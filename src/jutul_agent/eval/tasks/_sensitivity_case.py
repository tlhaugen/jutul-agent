"""Case description shared by the sensitivity suite and the scenarios.

Both lines match the reference (``05_task_references.jl``) and are stated in
full, since JutulDarcy's defaults would otherwise change the discrete J:

- ``ImmiscibleSystem`` uses 1000 kg/m^3 for both phases in the reservoir unless
  a density model is given, and treats the given densities as surface
  densities only.
- With the default nonlinear tolerances, J carries an error of about 3e-3
  relative in this case, and it jumps when a perturbation changes the number
  of Newton iterations. Adaptive substepping (``timesteps = :auto``) adds
  further jumps, and so do step cuts when Newton fails to reach tight
  tolerances within the default 15 iterations. With 50 iterations, every
  run in the reference work is cut in the same way.
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
    "cut if Newton fails to converge), the nonlinear tolerances `tol_cnv = 1e-6` "
    "and `tol_mb = 1e-10`, and at most 50 Newton iterations per step "
    "(`max_nonlinear_iterations = 50`).\n"
)

# The same settings as Julia keyword arguments, for the provided setup scripts.
SIMULATION_STEPPING = "timesteps = :none, max_nonlinear_iterations = 50"
SIMULATION_KWARGS = f"{SIMULATION_STEPPING}, tol_cnv = 1e-6, tol_mb = 1e-10"
