---
name: jutuldarcy-sensitivities
description: Compute, verify, and interpret parameter sensitivities in JutulDarcy using adjoint gradients and finite-difference checks, including directional checks that cover well cells
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
A gradient can be finite, plausible, and still wrong or incomplete.

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

Production rates use a negative sign convention in JutulDarcy. A negative
objective representing positive physical production reverses the meaning of
every gradient sign.

Check the sign of `J` against its physical meaning, not only against finite
differences. If the adjoint objective and the finite-difference calculation
use the same wrong sign, they agree and the error is not detected.

Do not silently change the objective sign. Define the objective explicitly and
state what increasing `J` means physically.

## 3. Validate the forward problem

The base simulation, its model, state, parameters, schedule and forces may be
reused from the persistent Julia session for the adjoint computation.

If a new case is required, follow the `jutuldarcy-overview` and
`jutuldarcy-wells` skills.

The forward problem must be valid before sensitivities are computed:

- validate the assembled case,
- run the forward simulation,
- check for solver failure or obvious non-convergence,
- inspect the quantity of interest.

Do not continue to gradient interpretation when the forward solve is invalid.

Reuse applies to the base run only. Every perturbed simulation used for
verification must be built from a new case (see section 7).

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
reservoir cells, and report cell positions both as linear index and as
Cartesian index when the grid is structured.

### Derived quantities may be missing from the gradient

An input parameter such as permeability often enters the discrete equations
only through derived quantities, for example face transmissibilities and well
indices. The adjoint function may not propagate every derived quantity back to
the input parameter. In particular, a permeability gradient may hold the well
indices fixed, so that the sensitivity in perforated cells is incomplete, while
a perturbed simulation that rebuilds the case changes them.

In JutulDarcy, `JutulDarcy.reservoir_sensitivities(case, result, objective;
include_parameters = true)` returns the permeability gradient under the key
`:permeability`, with the well indices held fixed. The sensitivities with
respect to the well indices are available from the lower-level adjoint
solve, which returns one entry per model, keyed by the well names:

    sens  = Jutul.solve_adjoint_sensitivities(case, result.result, objective)
    dJ_dWI = sens[:WellName][:WellIndices]      # one value per perforation
    WI     = case.parameters[:WellName][:WellIndices]

Here `result` is the output of `simulate_reservoir`, and `:WellName` is the
name given to the well, for example `:Injector` or `:Producer`. Inspect
`keys(sens)` to see the available entries.

The full log-permeability sensitivity in a perforated cell `c` is

    dJ/dlogK_c = (dJ/dlogK_c with well indices fixed) + (dWI_c/dlogK_c) * dJ/dWI_c

For an isotropic Peaceman well index, `WI_c` is proportional to `K_c`, so
`dWI_c/dlogK_c = WI_c`, and the correction is `WI .* dJ_dWI` added to the
entries of the perforated cells. Map each perforation to its reservoir cell
from the well definition, and check that the number of well indices equals
the number of perforated cells before adding the correction.

Either apply this correction or report clearly that the gradient holds well
indices fixed. The directional check in section 7 detects the difference.

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

## 7. Verify the gradient numerically

Numerical verification is part of the sensitivity workflow. It has two parts,
component checks and a directional check, and both are needed. Component checks
can locate an error, but only test the entries that are checked. A directional
check tests all entries at once.

Prefer Jutul's installed finite-difference or optimization utilities when they
apply. Inspect their current signatures before calling them. Otherwise use
central finite differences.

### Rebuild every perturbed case

When a parameter changes, every quantity derived from it must be recomputed,
including transmissibilities and well indices. Build each perturbed simulation
from a new case with the perturbed parameter. Do not perturb a parameter array
inside a model object from an earlier tool call, since that object can still
hold quantities derived from the unperturbed values.

### Component checks

For log-permeability, perturb one component with:

    K_i_plus  = K_i * exp(epsilon)
    K_i_minus = K_i * exp(-epsilon)

and estimate:

    dJ/dlogK_i ≈ (J(K_i_plus) - J(K_i_minus)) / (2 * epsilon)

Check a representative subset that includes:

- the largest entries,
- at least one entry close to zero,
- every cell perforated by a well.

### Directional check

Choose a random unit vector `v` over all parameter components, including well
cells. Perturb all components at once, `logK + epsilon * v` and
`logK - epsilon * v`, and compare

    (J(logK + epsilon * v) - J(logK - epsilon * v)) / (2 * epsilon)

with the adjoint prediction `dot(grad, v)`. If the component checks agree but
the directional check does not, the error lies in components that were not
checked, often in perforated cells.

### Measuring the error

Measure errors relative to the largest gradient component,

    e = |g_FD - g_adj| / max_i |g_adj_i|

since the relative error of a single component is not meaningful when the
component is close to zero. For the directional check, the relative error
`|D_FD - D_adj| / |D_adj|` is appropriate.

Test several perturbation sizes, for example `epsilon = 1e-1, 1e-2, ..., 1e-6`.
The error should decrease and then either level off or grow again. A level
plateau usually reflects solver tolerances. To test this, repeat the comparison
with tightened nonlinear and linear tolerances. The plateau level is an estimate
of how accurately the gradient can be verified.

If two components differ by less than this accuracy, do not claim that one of
them is larger. Report both.

For each check, report:

- cell index or direction,
- adjoint value,
- finite-difference value,
- error `e`,
- perturbation sizes used.

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
interpretation. Look at the whole field, not only the largest entries. Regions
of opposite sign can cancel when a whole region is perturbed, so the effect of
a regional change is the sum of its cell sensitivities, not the largest one.

Be careful with magnitude comparisons. A very large raw `dJ/dK` can simply
reflect that permeability is numerically very small in SI units. Prefer
log-parameterized or otherwise normalized sensitivities when comparing
importance across cells.

The gradient is local. It predicts the effect of small changes and does not
necessarily predict the effect of large ones.

## 9. Report reliability and limitations

A sensitivity result should include enough information for the user to judge
whether it is trustworthy.

Report relevant items such as:

- objective definition,
- objective sign convention,
- differentiated parameter,
- parameterization,
- whether well indices and other derived quantities are included in the gradient,
- forward-solve status,
- component-check and directional-check errors,
- finite-difference step sizes,
- the estimated verification accuracy,
- assumptions made during interpretation.

Treat a disagreement between adjoint and finite differences as a problem to
investigate, not something to hide.

Possible causes include:

- comparing different parameterizations,
- incorrect objective definition or sign,
- stale model quantities after a perturbation,
- derived quantities, such as well indices, held fixed in the adjoint gradient,
- finite-difference step size that is too large or too small,
- nonlinear or linear solver tolerances,
- convergence or time-step differences between perturbed runs,
- an API mismatch or incorrect extraction of the gradient.

Do not claim a gradient is validated until both component checks and a
directional check agree within the estimated accuracy. If a discrepancy cannot
be resolved, report the result as not validated and state what was checked.

## Recommended workflow

For a typical sensitivity request:

1. Inspect or construct the forward case.
2. Validate and run it.
3. Define the scalar objective `J` and check its sign against its physical meaning.
4. Identify parameter `p` and parameterization.
5. Locate the installed adjoint API.
6. Compute the gradient and check its dimensions and numerical sanity.
7. Check whether derived quantities such as well indices are included, and correct or report this.
8. Convert or normalize the gradient when useful.
9. Verify the largest entries, an entry close to zero and the well cells, each from a rebuilt case.
10. Verify the whole gradient with a directional check.
11. Interpret the field physically.
12. Report assumptions, verification errors and limitations.

Prefer evidence from the running model, installed source, and numerical checks
over assumptions about what the API or physics should do.