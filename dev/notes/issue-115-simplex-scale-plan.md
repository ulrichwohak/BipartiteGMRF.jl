# Issue #115 — expose the Nelder-Mead initial simplex scale

**Status:** agreed with user 2026-08-19; implementing on `feat/issue-114-init-warm-start`
(same branch as #114, per user request — #115 is what makes #114's warm starts
actually pay off, so they ship together)

## Goal

`optimize_problem` calls a bare `NelderMead()` (`src/solvers/common.jl:352`),
so Optim's default `AffineSimplexer(a=0.025, b=0.5)` builds the initial simplex:
vertex `j+1` is `(1 + b)·x_j + a` in coordinate `j`. Two consequences, both of
which bite exactly when `init` is supplied:

1. A **50% relative** perturbation is enormous in unconstrained space
   (`log σ = −0.9 → −1.325`, i.e. `σ` down 35%), so a warm-started simplex still
   spans a wide region and the wall-clock saving is far smaller than #114 wants.
2. A coordinate at **exactly zero** gets only the absolute `a = 0.025`, and one
   merely near zero gets `0.5|x|`, which is smaller still. Warm starts produce
   exactly those: `log ω = 0` (the ladder starts pinned), `atanh(η) ≈ 0`,
   `log σ ≈ 0` at `σ ≈ 1`, `atanh(ρ/L) ≈ 0` at `ρ ≈ 0`. A simplex that is
   near-degenerate in one coordinate can drive the f-value spread under
   `g_abstol` immediately and return `converged = true` having barely moved.
   `ExactCholesky` is rescued by its L-BFGS polish (`exact.jl:134-156`);
   **`HutchSLQ` has no polish** (`hutch.jl:185`), so it can silently return the
   init.

## Decisions

### D1. Two fields, `simplex_scale` and `simplex_shift` (user decision)

On both `ExactCholesky` and `HutchSLQ`, defaulting to Optim's own values:

| field | Optim | default | meaning |
|---|---|---|---|
| `simplex_scale` | `b` | 0.5 | relative perturbation, `(1+b)·x_j` |
| `simplex_shift` | `a` | 0.025 | absolute perturbation, added on top |

Both are exposed rather than just `b`, because `a` is what keeps a coordinate
sitting at *exactly* zero from degenerating — hiding it would leave the sharpest
edge of the bug unreachable. It also makes the natural warm-start configuration
expressible: `simplex_scale = 0.0, simplex_shift = 0.05` is a uniform absolute
box around `init`, independent of coordinate magnitude.

Validation: both `>= 0`, and `simplex_scale + simplex_shift > 0` — at `0, 0`
every vertex equals `x`, the simplex is fully degenerate, and Nelder-Mead
"converges" instantly on a zero f-spread.

`EMIWBlocks` is untouched: it is an EM loop, and its two `optimize` calls
(`emiwblocks.jl:161`, `:190`) are 1-D bracketed Brent searches with no simplex.

### D2. `g_abstol` stays start-dependent (user decision)

`f0 = obj(p0)` → `fscale = max(1.0, abs(f0))` → `g_abstol` (`common.jl:347-350`)
is left exactly as it is. Documented, not changed: no existing fit's stopping
behavior moves. The consequence worth stating in the docs is that under
`HutchSLQ` (which passes `g_reltol·fscale` through unmodified, `hutch.jl:181`)
a better starting point buys a *tighter* tolerance and can cost extra
iterations; `ExactCholesky` floors it at `1e-3` (`exact.jl:132`), so it usually
does not move. Note also that `max(1.0, |f0|)` means the direction of the effect
reverses for a negative NLL.

### D3. Dispatch, not field access

`optimize_problem` is generic over `AbstractGMRFSolver`, so the simplexer comes
from a new solver-dispatched method `nelder_simplexer(solver)`, alongside the
four that already exist (`make_nll_cache`, `nll_value`, `nelder_g_abstol`,
`polish`). The comment block at `common.jl:150-157` that enumerates them is
updated to five.

## Worksteps

- [x] **1. `src/types.jl`** *(done)* — `simplex_scale` / `simplex_shift` on both
      solvers, appended after the existing fields, with a shared
      `validate_simplex`. Both constructors are keyword-only (the inner `new` is
      the sole positional use), so appending is safe; nothing in `src/` or
      `test/` builds a solver positionally or compares solvers for equality.
- [x] **2. `src/solvers/{common,exact,hutch}.jl`** *(done)* — `nelder_simplexer`
      methods, `NelderMead(initial_simplex = nelder_simplexer(solver))`,
      `AffineSimplexer` imported, five-methods comment updated.
- [x] **3. Tests** *(done)* — 101 assertions in `test/test_init_warm_start.jl`.
- [x] **4. Docs** *(done)* — the shared explanation lives on the `HutchSLQ`
      docstring with `ExactCholesky` pointing at it; `docs/src/performance.md`
      "Warm starts" rewritten (it previously ended by pointing at #115);
      CHANGELOG under v0.5.0. Documenter build verified.

## Verification (advisor run stalled; checked directly instead)

- **Backward compatibility.** Optim's `NelderMead(; kwargs...)` keeps
  `parameters = AdaptiveParameters()` when only `initial_simplex` is passed
  (`nelder_mead.jl:75-85`), so `NelderMead(initial_simplex=AffineSimplexer(0.025,
  0.5))` is equivalent to a bare `NelderMead()`. Confirmed at runtime against
  the version this project resolves to, not just by reading the source. A test
  asserts a fit with explicit defaults is `==` on `nll` and
  `theta_unconstrained` to one without.
- **Argument order.** `AffineSimplexer(a, b)` is positional with
  `(1 + b)·x_j + a`, while the solver kwargs are named for meaning
  (`simplex_scale` = `b`, `simplex_shift` = `a`). A swap would be silent and
  would inverting the feature, so a test pins the mapping in both directions.
- **Setting reaches the optimizer.** Added after the fact: a deliberately tiny
  simplex must change `theta_unconstrained`. Without it, a silent revert to a
  bare `NelderMead()` would leave every other assertion in the testset passing.
- **Field append.** New fields are last on both structs; `fieldnames` confirms
  the pre-existing order is untouched.

## Out of scope

- Changing `g_abstol` (D2).
- Any change to `EMIWBlocks`.

## Out of scope

- Changing `g_abstol` (D2).
- Any change to `EMIWBlocks`.
