# Issue #114 — `init=` warm-start parameter for `fit_mle` / `optimize_problem`

**Status:** plan agreed with user 2026-08-19; advisor-reviewed; awaiting go-ahead to create the branch
**Base branch:** `release/v0.4.0` (current HEAD `e41145d`, v0.4.1) — *not* `main`, which is ~4 releases behind
**Working branch (create only on "go"):** `feat/issue-114-init-warm-start`
**Follow-up filed:** #115 (NelderMead initial simplex scale) — out of scope here

## Goal

Let a caller supply a starting point for the optimizer in structural units,
instead of always starting from the hardcoded heuristic in `initial_params`
(`src/util.jl:111`). Partial `init` allowed: any missing field falls back,
field-by-field, to the existing default. `init=nothing` leaves today's default
path untouched.

## Decisions

### D1. `init` is a `NamedTuple` only (user decision)

No raw-unconstrained-vector escape hatch. Consequence, accepted knowingly: a
degenerate optimum with `sigma_z == 0.0` exactly — the `error_groups` fit that
motivated the issue — **cannot be used as an init**, because `log(0) = -Inf`
and validation must reject it. That does not block the intended diagnostic,
which is to re-fit *from the MD point* and see whether it lands in the same
place.

### D2. Units: original outcome scale (user decision)

`suffstats(...; standardize=true)` (the default) divides `y` by `y_std`
(`src/stats.jl:265`); every scale is un-scaled on the way out
(`src/solvers/common.jl:329-331`, `beta_std .* y_std` at `:322`, EMIW at
`src/solvers/emiwblocks.jl:236,247,250`). So `init` sigmas are read in
**original outcome units** and divided by `stats.y_std` before the codec.

- `init=(rho=r.rho, sigma_a=r.sigma_a, ...)` from a prior `GMRFResult` restarts
  where that fit finished, **to floating-point round-off** (`σ → log(σ/y_std) →
  exp(·)·y_std` is a 1–2 ulp round-trip, not an identity).
- A raw-data MD estimate passes straight through, no manual rescaling.
- `standardize=false` ⇒ `y_std == 1` and the conventions coincide.
- Advisor confirmed this survives `X`/`beta` (never standardized), `error_cov`
  (renormalized to `tr(R)=K` at `src/prepare.jl:482`, so `σ_ε` keeps its
  meaning), and match-weighted standardization (`src/stats.jl:250-252`, changes
  only the *value* of `y_std`).
- `rho`, `eta`, `omega`, `r` are scale-free and unconverted.

This deviates from the issue body's literal `log(init.sigma_a)` and from its
acceptance criterion as phrased; under D2 that reads
`unpack_params(p0).sigma_a == init.sigma_a / y_std`.

### D3. Key validation: strict on typos, permissive on `nothing`

Accepted keys: `rho`, `sigma_a`, `sigma_z`, `sigma_epsilon`, `rho_eps`, `eta`,
`omega`, `phi`, `r`, `delta`, plus `beta` **accepted and ignored**. Anything
else throws an `ArgumentError` naming the offender — a silent `sigma_eps=` typo
would look like a working warm start and waste an hour of wall clock.

A field whose value is `nothing` counts as *not supplied*. Together with the
ignored `beta`, this makes `init = params(result)` work directly:
`params` returns `(rho, sigma_a, sigma_z, sigma_epsilon, rho_eps, eta, beta)`
(`src/api.jl:108-116`) with `nothing` in the inactive slots.

### D4. One rule for pinned vs absent parameters

The draft had three inconsistent behaviors; unify:

- **Block absent from the fit** → **error**. (`init.eta` with no `error_eta`,
  `init.omega` with no `error_groups`, `init.phi`/`r`/`delta` with no
  `error_blocks`.)
- **Block present but the parameter is pinned** → **warn and ignore**.
  (`fix_rho`; `error_ar1.eta_fixed !== nothing`; `solver.r isa Float64`;
  `solver.delta isa Float64`; `weighting.estimate_rho_eps == false`.)

Activity predicates must match the code exactly (advisor 2.7):

| parameter | active iff | source |
|---|---|---|
| `rho_eps` | `weighting.observations == :effective && weighting.estimate_rho_eps` | `common.jl:266-267` |
| `omega` | `length(error_classes.counts) - 1 > 0` (not merely `error_classes !== nothing`) | `common.jl:270` |
| `eta` | `error_ar1 !== nothing && error_ar1.eta_fixed === nothing` | `common.jl:268` |

### D5. Domains

- `rho`: `abs(rho) < rho_limit(model)`. Under `rho_limit=:auto` the limit is
  resolved from the NB spectrum inside `_build_model` (`src/fit.jl:33`) and can
  sit well below 0.99 — the error message must name the *resolved* limit and
  say it came from `:auto`.
- `sigma_a`, `sigma_z`, `sigma_epsilon`, `phi`: finite and strictly positive.
- `rho_eps`: `0 ≤ rho_eps < 0.999`, **not** `< 1`. `rhoeps_to_unconstrained`
  clamps silently (`src/util.jl:11-15`), so `0.9995` would be accepted and then
  quietly moved. Document the ceiling.
- `eta`: no separate check — `eta_to_unconstrained` already throws a good
  message (`src/util.jl:24-25`).
- `omega`: all entries finite and positive. Accept **either** length `C−1`
  (classes 2..C, the internal layout) **or** length `C` with `omega[1] ≈ 1`,
  since the result reports `error_class_variances = vcat(1.0, omega)` of length
  `C` (`common.jl:191`). Classes are sorted ascending by group size
  (`src/prepare.jl:640`); class 1 is the smallest and is pinned at 1.

### D6. EMIW mapping (corrected — the draft was wrong here)

`r` and `δ` are **bracketed Brent minimizations over their full domain**
(`emiwblocks.jl:190` and `:161`), not searches from a starting point:

- `init_delta` is essentially a **no-op** when `solver.delta == :estimate`:
  δ is globally re-minimized on the first pass of the inner loop (`:155-164`),
  its only residual effect being one `Ωeff` build. **Dropped from scope**; if
  passed with `solver.delta == :estimate`, warn that it seeds only the first
  E-step.
- `init_r` does seed the first E-step for real (`r` enters `Ωeff` at `:128` and
  `svals` at `:147`, before its own update at `:184-192`), but it is an initial
  value, not a search start — document it that way. **Reject `init.r` when
  `mmax == 1`**: `r` is not estimated at all in that case (`:184`), so the
  value would silently become permanent.
- `φ ↔ σ_ε`: `omega_bar = φδ/(δ−2) = σ_ε²` (`:231,236`) is correct. `init.phi`
  is used directly (divided by `y_std²`); failing that, `init.sigma_epsilon`
  converts as `φ₀ = (σ_ε/y_std)² · (δ₀−2)/δ₀`.
- **All defaulting and that conversion happen inside `optimize_emiw`**, not in
  `solve`. The fallbacks live there — `(default_rho_start(limit), 0.7, 0.04)`
  at `:96`, `φ₀ = 0.5` at `:98`, `δ₀ = 10.0` at `:108` — and hard-coding `10.0`
  in a second file would drift. So replace the all-or-nothing
  `init_theta::NTuple{3,Float64}` (`:78`) with field-wise
  `Union{Nothing,Float64}` kwargs, or pass the rescaled `init` NamedTuple down
  whole. This changes that private kwarg's unit convention; the only existing
  caller (`test/test_error_blocks_iw.jl:151-154`) runs `standardize=false`, so
  it is unaffected.
- `_em_blocks_ψ_from_θ` does a bare `atanh(rho/limit)` with **no domain guard**
  (`src/solvers/emblocks.jl:173-175`) and `optimize_emiw` validates only
  `init_phi > 0` (`:99`). Promoting these to public surface requires adding
  `|rho| < limit` and `sa, sz > 0` checks.

### D7. Documented caveats (no code change here — see #115)

The docstring must say that a warm start is **not guaranteed to be faster**:

- `g_abstol` is derived from `f0 = obj(p0)` (`common.jl:283-286`), so a better
  start yields a tighter tolerance. `ExactCholesky` masks this behind a
  `max(1e-3, ·)` floor (`exact.jl:132`); `HutchSLQ` passes it straight through
  (`hutch.jl:181`) and can therefore take *more* iterations from a better start.
- NelderMead's default `AffineSimplexer(a=0.025, b=0.5)` perturbs each
  coordinate by 50%, so the initial simplex still spans a wide region; and
  coordinates near zero — `log ω = 0`, `atanh(η) ≈ 0`, `log σ ≈ 0` at `σ ≈ 1` —
  collapse to a step of `0.025` and can trip convergence immediately.
  `ExactCholesky` is rescued by its LBFGS polish; **`HutchSLQ` has no polish**
  (`hutch.jl:185`), so a warm-started Hutch fit can silently return the init.

## Worksteps

One reviewable increment each; stop and report between them.

- [x] **1. `src/util.jl` — codec + validation.** *(done)*
      `validate_init(init, status; rho_limit)` implementing D3/D4/D5, plus
      `init_field`, `validate_positive_init`, `init_omega_codes`, `init_eta_code`,
      and `initial_params(rho_fixed, estimate_rho_eps; rho_limit, init=nothing)`
      with field-by-field fallback for `rho`, `sigma_a`, `sigma_z`,
      `sigma_epsilon`, `rho_eps`. Sigmas arrive already divided by `y_std`.
      `status` is a NamedTuple over `(rho, rho_eps, eta, omega, phi, r, delta)`
      with values `:free`, `:absent`, or the pinned value itself; step 2 builds it.
      **Correction to the golden below:** `default_rho_start(L) = min(0.5, 0.5L)`,
      so for `L ≤ 1` the first entry is `atanh(0.5)`, *not* `atanh(0.5/L)`.
      `init.omega` length is always unambiguous (`n_omega` vs `n_omega+1`).
      Advisor confirmed the `init=nothing` path is bit-identical to the old
      literal-vector construction (`reinterpret(UInt64, ·)` equal for
      `rho_limit ∈ {0.3, 0.5, 0.99, 1.0}`) and that Aqua/JET/docs are unaffected.
      Four advisor findings fixed in the same increment:
      (a) validation now iterates `INIT_OPTIONAL_KEYS` and defaults a missing
      `status` key to `:absent`, so a `status` that forgets a key fails loud
      instead of letting that init field pass unvalidated *and* unused — the
      exact silent-warm-start failure D3 exists to prevent;
      (b) one consistent `:absent` default across all keys, via a single
      `validate_init_domain`;
      (c) `initial_params` guards `init.rho` itself, so an out-of-range value
      cannot reach `atanh` and surface as a contextless `DomainError`;
      (d) `init.omega` must be an `AbstractVector` — `Float64.(collect(2.0))`
      silently returned a scalar, making the return type
      `Union{Vector{Float64},Float64}`.
      Also renamed `check_positive_init` → `validate_positive_init` to parallel
      `validate_init`, and added `validate_real_init` so a non-numeric field
      raises `ArgumentError` rather than `MethodError`.

      **Naming:** step 2's helper is `initial_point`, no underscore — `util.jl`
      and `solvers/common.jl` use bare snake_case for internals
      (`initial_params`, `full_params`, `objective_stats`, `build_gmrf_result`);
      the `_`-prefixed convention lives in `emiwblocks.jl`/`emblocks.jl`.
- [x] **2. `src/solvers/common.jl` — `initial_point` helper + threading.** *(done)*
      `initial_point(stats, fix_rho, estimate_rho_eps, n_omega, estimate_eta,
      init; rho_limit)` assembles the whole `p0` — the sigma rescaling, the
      `initial_params` call, the ω ladder and the η append — so the layout is
      unit-testable with exact `==`. `init_status` builds the `:free` /
      `:absent` / pinned-value map from the same predicates `optimize_problem`
      uses. `optimize_problem` gained `init::Union{Nothing,NamedTuple}=nothing`.
      Validation runs on the caller's *original-unit* init so error messages
      quote their numbers; the rescaling that follows is positivity-preserving,
      so nothing escapes.
- [x] **3. Public surface.** *(done)* `init` on both `solve` methods and all
      three `fit_mle` methods, pass-through only.
- [x] **4. EMIW.** *(done)* `optimize_emiw`'s `init_theta`/`init_phi` replaced by
      one `init` NamedTuple, accepting partial specifications; validation and
      rescaling moved inside, so `_em_blocks_ψ_from_θ`'s bare `atanh`/`log` are
      now guarded (they were not before). `emiw_init_status` reports pinned
      `r`/`delta` by value. `test/test_error_blocks_iw.jl:151` migrated —
      `standardize=false` there, so the numbers are unchanged.
- [x] **5. Tests.** *(done)* `test/test_init_warm_start.jl`, 74 assertions,
      wired into `runtests.jl`. Full suite green including Aqua.
      `Base.CoreLogging.Warn` rather than `using Logging`, which is not a test
      dependency and would have meant touching `Project.toml`.
- [x] **6. Docs + CHANGELOG.** *(done)* `init` section on the suffstats-based
      `fit_mle` docstring (the other two and `docs/src/api.md` pick it up by
      reference), a docstring for the previously undocumented `EMIWBlocks`
      `solve` method, a "Warm starts" section in `docs/src/performance.md`, a
      README example, and an `Unreleased → Added` CHANGELOG entry. Version bump
      to v0.5.0 is left for the release commit.

## Advisor findings on increments 2–6, and what was done

Mechanism confirmed correct: parameter layout and every consumer index, the
`:free`/`:pinned`/`:absent` predicates, unit handling and the completeness of
the rescale set, `nothing` semantics, EMIW's δ-before-φ ordering, the test
migration, and end-to-end threading with no silently-ignored `init` on any
entry point.

Acted on:

- **`init.sigma_epsilon` was silently dropped when `init.phi` was also given**
  (EMIW) — exactly the failure `validate_init` exists to prevent. Now warns.
- **Wrong hint when `init.r` is rejected at `mmax == 1`**: the generic message
  said "pass `error_blocks=:iw`", which the caller had already done. Now a
  dedicated message naming all-singleton blocks as the reason, plus a test.
- **Spurious warning on `init = params(result)`**: a fit with a pinned
  parameter (numeric `rho_eps`, numeric `error_eta`, solver-fixed `r`/`delta`)
  reports that value on the result, so re-fitting the same configuration warned
  about a value it had itself produced. `init_status` now reports the pinned
  *value* rather than the symbol `:pinned`, and the warning is suppressed when
  the supplied value agrees with it.
- Four docstring inaccuracies: the incomplete list of pinned cases; the claim
  that `params(result)` "works directly" without the pinned-block caveat;
  "0.025 step" (Optim uses `0.5|x|` unless the coordinate is *exactly* zero);
  and "a better start buys a tighter tolerance" (`fscale = max(1, |f0|)`
  reverses this for negative NLL, and `ExactCholesky` floors it — the sentence
  describes `HutchSLQ`). Also completed the suffstats kwarg list.
- Removed the redundant re-validation in `optimize_emiw` and the now-unreachable
  `φ > 0` throw whose message named `init.phi` even when the value came from
  `init.sigma_epsilon`.

Not acted on:

- The advisor flagged a possible layout collision if `estimate_rho_eps` and the
  ω ladder were ever both active (both would want slot 5). **`suffstats`
  forbids it**: `error_groups`, `error_eta` and `error_blocks` each require
  `Weighting(observations=:raw)` (`stats.jl:125`, `:140`, `:171`), while
  `estimate_rho_eps` requires `:effective`. At most one of the three can occupy
  slot 5. Documented on `init_status` rather than guarded.
- JET: `test_jet.jl` is excluded from the default suite and its two targets
  (`jet_exact_solve_flow`, `jet_covariance_flow`) do not reach `solve`, so the
  type-unstable NamedTuple operations in `validate_init` / `rescale_init_sigmas`
  are not covered either way. Pre-existing exclusion, once-per-fit cost.

## Out of scope

- NelderMead simplex scale and the `g_abstol` start-dependence → **#115**.
- Raw unconstrained-vector `init` (D1) — so the `sigma_z == 0.0` boundary fit
  that motivated the issue still cannot be used *as* an init.
