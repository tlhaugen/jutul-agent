---
name: jutuldarcy-sensitivities
description: Compute, validate, and interpret parameter sensitivities in JutulDarcy using adjoint and numerical gradients
---

# JutulDarcy sensitivity analysis

## When to use

Use this skill when the user asks about:

- parameter sensitivities or gradients,
- adjoint differentiation,
- which reservoir cells or parameters most influence a result,
- sensitivities with respect to permeability, porosity, controls, or other model parameters,
- gradients for optimization,
- numerical verification of a JutulDarcy gradient.

A reliable sensitivity analysis has three parts:

1. define what is differentiated,
2. compute the gradient,
3. verify and interpret it.

Do not report a gradient as reliable merely because an adjoint call completed.

## 1. Identify the quantity of interest and parameter

Before differentiating, make the mathematical problem explicit.

Identify:

- a scalar quantity of interest `J`,
- the parameter or parameter field `p`,
- the parameterization in which the gradient will be reported.

The adjoint objective must be scalar.

Typical objectives include:

- cumulative production,
- cumulative injection,
- pressure at a selected location or time,
- total recovery,
- a mismatch or loss function.

Typical parameters include:

- cell-wise permeability,
- porosity,
- well controls,
- fluid or rock parameters exposed by the model.

If the user's request does not uniquely determine `J` or `p`, inspect the
existing model and results before choosing them. State any interpretation you
make.

## 2. Check the objective carefully

Prefer smooth, differentiable objectives.

Thresholds, clipping, `max`, discrete switching events, or other non-smooth
operations can make an adjoint gradient misleading or undefined. If the
requested objective contains such operations, explain the issue and use a
smooth alternative when appropriate.

Check sign conventions before interpreting the gradient.

In particular, production rates may use a negative sign convention. A negative
objective representing positive physical production can therefore reverse the
meaning of every gradient sign.

Do not silently change the objective sign. Define the objective explicitly and
state what increasing `J` means physically.

## 3. Reuse and validate the forward problem

Prefer reusing the model, state, parameters, timestep schedule, forces, and
simulation results already present in the persistent Julia REPL.

If a new case is required, follow the `jutuldarcy-overview` and
`jutuldarcy-wells` skills.

The forward problem must be valid before sensitivities are computed:

- validate the assembled case,
- run the forward simulation,
- check for solver failure or obvious non-convergence,
- inspect the quantity of interest.

Do not continue to gradient interpretation when the forward solve is invalid.

## 4. Inspect the installed sensitivity API

Jutul and JutulDarcy APIs can change between versions. Use the installed source,
examples, and Julia introspection instead of guessing function signatures.

Useful search terms include:

- `reservoir_sensitivities`
- `adjoint`
- `sensitivity`
- `setup_reservoir_dict_optimization`
- `parameters_gradient_reservoir`
- `finite_difference_gradient_entry`

Use the source tools and REPL tools that are already available:

- `grep` and `read_file` for installed source and examples,
- `@doc`,
- `methods`,
- `names`,
- `fieldnames`.

Use the API available in the active environment rather than assuming one
particular JutulDarcy release.

## 5. Compute the adjoint gradient

Use Jutul/JutulDarcy's installed adjoint functionality through `run_julia`.

Keep these three quantities distinct:

- the physical parameter,
- the parameter representation used internally,
- the gradient representation reported to the user.

After computing a gradient, check at least:

- the number of gradient components,
- whether all values are finite,
- minimum and maximum values,
- largest absolute entries,
- an overall scale such as the gradient norm.

For spatial parameter fields, preserve the mapping from gradient entries to
reservoir cells.

## 6. Use an interpretable parameterization

Raw gradients can have unintuitive scales because reservoir parameters are
expressed in SI units.

For positive permeability `K`, a log-permeability gradient is often easier to
interpret:

    dJ/dlogK = K * dJ/dK

This represents sensitivity to a relative change in permeability rather than an
absolute change measured in m².

Never compare or validate `dJ/dK` against `dJ/dlogK` without performing the
conversion.

State clearly which gradient is being reported.

## 7. Verify representative entries numerically

Numerical verification is part of the sensitivity workflow.

Prefer Jutul's installed finite-difference or optimization utilities when they
apply. Inspect their current signatures before calling them.

Otherwise use a central finite difference.

For log-permeability, perturb one component with:

    K_i_plus  = K_i * exp(epsilon)
    K_i_minus = K_i * exp(-epsilon)

and estimate:

    dJ/dlogK_i ≈
        (J(K_i_plus) - J(K_i_minus)) / (2 * epsilon)

When permeability changes, ensure that model quantities derived from
permeability, including transmissibilities, are recomputed. Do not perturb a
permeability array while accidentally reusing stale derived quantities.

For each checked component, report:

- cell or parameter index,
- adjoint gradient,
- finite-difference gradient,
- relative error,
- perturbation size.

When practical, test more than one `epsilon`. Decreasing error followed by a
plateau is usually more informative than a single finite-difference result.

For a large parameter field, validate a representative subset rather than every
entry. Include important high-sensitivity entries and, when useful, lower-
sensitivity entries from other regions.

## 8. Interpret the gradient physically

Do not stop at printing a vector.

For important sensitivities, explain:

- where the parameter is located,
- the sign of the sensitivity,
- its magnitude,
- the parameterization,
- what an increase in the parameter is predicted to do to the objective.

For spatial permeability sensitivities, relate important cells to wells,
connectivity, and expected flow paths when the model supports that
interpretation.

Be careful with magnitude comparisons. A very large raw `dJ/dK` can simply
reflect that permeability is numerically very small in SI units. Prefer
log-parameterized or otherwise normalized sensitivities when comparing
importance across cells.

## 9. Report reliability and limitations

A sensitivity result should include enough information for the user to judge
whether it is trustworthy.

Report relevant items such as:

- objective definition,
- objective sign convention,
- differentiated parameter,
- parameterization,
- forward-solve status,
- numerical gradient-check error,
- finite-difference step size,
- assumptions made during interpretation.

Treat a large disagreement between adjoint and finite differences as a problem
to investigate, not something to hide.

Possible causes include:

- comparing different parameterizations,
- incorrect objective definition,
- stale model quantities after a perturbation,
- finite-difference step size that is too large or too small,
- nonlinear solver tolerances,
- convergence differences between perturbed runs,
- an API mismatch or incorrect extraction of the gradient.

Do not claim a gradient is validated until these issues have been checked.

## Recommended workflow

For a typical sensitivity request:

1. Inspect or construct the forward case.
2. Validate and run it.
3. Define the scalar objective `J`.
4. Identify parameter `p` and parameterization.
5. Locate the installed adjoint API.
6. Compute the gradient.
7. Check its dimensions and numerical sanity.
8. Validate selected entries numerically.
9. Convert or normalize the gradient when useful.
10. Interpret the important entries physically.
11. Report assumptions, validation error, and limitations.

Prefer evidence from the running model, installed source, and numerical checks
over assumptions about what the API or physics should do.