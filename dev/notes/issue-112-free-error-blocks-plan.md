# Issue #112 — Free per-firm error covariance (arbitrary Ωᵢ): implementation plan

- **Status:** incomplete scaffold — core dense-verified, EM convergence **not** established (see "Status" note below)
- **Issue:** `ulrichwohak/BipartiteGMRF.jl#112`
- **Baseline:** PR #111 (`error_cov = R`, `error_groups`) on `origin/feat/eta-error-bands` (`ec85427`)
- **Branch:** `feat/free-error-blocks` (off `ec85427`)
- **Audience:** implementing agent

## Status (implementation, 2026-08-18, uncommitted on `feat/free-error-blocks`)

**Done + dense-verified (machine precision):** `FreeBlockStats`, `EMFreeBlocks`;
`suffstats(…; error_blocks=:free, firm_group=…)` with all rejections (mutex,
`:raw`, length, unknown mode, `match_id`, duplicate-manager, and multi-firm
`firm_group`); `assemble_freeblock_precision` (`A'Ω⁻¹A` + `A'Ω⁻¹y`);
`make_emfree_workspace` (R1 full-pattern seeding); `emfree_e_step`;
`emfree_nll` (Woodbury form); `solve`/`validate_capability`/`build_emfree_result`;
`dof()`; ρ≈0 warning + metadata caveats. 24 tests green.

**Blocked (the issue's real gate):** the EM loop (E-step → Fisher-identity score
→ safeguarded score step → closed-form M-step) drives to the boundary or crashes
on small/weakly-identified graphs. The closed-form M-step `Ωᵢ ← S_{ε,i}` yields
**near-singular blocks** on bridge-heavy/cyclic test graphs, so the next
`M = K + A'Ω⁻¹A` is ill-conditioned and the 4-term `AᵢM⁻¹Aᵢ'` Gram loses
positive-definiteness to roundoff. This is the incidental-parameters/weak-
identification difficulty of §4 manifesting as numerical breakdown — the issue's
§5.1 claim "free blocks stay interior automatically" does **not** hold empirically.
Criterion 1 (EM fixed point ↔ dense joint MLE) and criterion 4 (MC centering) are
therefore not established. Next step requires a decision: constrained/regularized
fallback (§5.4), a solve-based (not 4-term) `AᵢM⁻¹Aᵢ'`, and/or MC verification
on a graph shown to identify `σ_a`.
---

## 1. Scope

Third rung of the error-model ladder: **within-firm error covariance Ωᵢ arbitrary
positive-definite, estimated jointly**, no parametrization, EM with closed-form
M-step. Subsumes the two landed rungs:

| rung | Ωᵢ | estimation | landed |
|---|---|---|---|
| `error_cov` | fixed PD, read from caller | one-time reweighting | PR #111 |
| `error_groups` | collapsed to firm-mean scalar ω_c | NelderMead on (θ, ω) | PR #111 |
| **free Ωᵢ** | **arbitrary PD, estimated** | EM alternating exact M-step | this |

`error_cov` and `error_groups` stay intact as roots.

The model (issue §1): `y = Aα + ε`, `α ~ N(0, K⁻¹)`, `ε ~ N(0, Ω)`,
`Ω = blkdiagᵢ(Ωᵢ)`, blocks contiguous by firm, each Ωᵢ free PD. Reported scalar is
the cross-firm average `bar σ² = (1/m) Σᵢ tr(Ωᵢ)`.

---

## 2. Branch & base

- **Base:** `origin/feat/eta-error-bands` = `ec85427`. This is the only ref carrying
  the PR-#111 rungs; verified `error_cov`/`error_groups`/`ErrorClassStats` are absent
  from `release/v0.3.0` and local `feat/gmrf-integration`.
- **New branch:** `feat/free-error-blocks` (honors `error_blocks = :free`;
  `feat/clustered-errors` is the issue-title alternative).

---

## 3. Verified facts grounding the plan

1. **E-step primitives already exist.** `GaussianMarkovRandomFields.GMRFWorkspace`
   exposes `workspace_solve`, `selinv`, `selinv_diag`, `selinv_dot`,
   `selinv_extract_at`; the dependency (`v0.12.4`) already requires `SelectedInversion`.
   **No new Takahashi code.** `selinv_extract_at(ws_M, B)` is documented for
   "diag(A Σ Aᵀ)"/observation-local pattern reads = `Aᵢ M⁻¹ Aᵢ'`;
   `selinv_dot(ws_M, ∂K/∂θ)` = `tr(∂K/∂θ M⁻¹)`.
2. **`M = Q + λV'V` is already the posterior precision under iid.** Free-Ω only changes
   the data term `λV'V → A'Ω⁻¹A`. Same `M` role, same solver shape.
3. **Parameter codec is clean to extend** (`util.jl`): `pfull = [atanh(ρ/ρlim),
   log σ_a, log σ_z, log σ_ε]`; `full_params`/`unpack_params`/`initial_params`.
   The `error_groups` path already appends `log ω` classes in `optimize_problem`.
4. **Dense-reference test convention exists** (`test_error_cov.jl`,
   `test_error_groups.jl`): compare `nll_exact_value` to a hand-built
   `Σ = V Q⁻¹ V' + σ²·(…)`. Criteria 1/2/3 copy this shape.

---

## 4. Architecture — free-Ω vs the landed rungs

| | `error_cov` | `error_groups` | **free Ωᵢ** |
|---|---|---|---|
| Ωᵢ | fixed, read from caller | collapsed to mean-scalar ω_c | **estimated, PD, free** |
| design rows | edge-level, R-weighted | **collapsed** to firm mean | **edge-level, never collapsed** |
| data term | `V'R⁻¹V` (one-time) | reassembled `V'ΛV` per eval | `A'Ω⁻¹A` per EM iter |
| estimation | one-shot | NelderMead | **EM** |

Single structural difference (§2): `A'Ω⁻¹A` gains manager→manager fill-in
(`F'Ω⁻¹F`), a clique per firm on that firm's managers — the same pattern the
firm-elimination already creates. **This is the load-bearing assumption to de-risk
first** (risk R1).

**Critical invariant (issue §3): no within-firm averaging.** Data stay at edge level
everywhere; Ωᵢ⁻¹ enters only as weighting in the normal equations. The firm mean is
**not** a sufficient statistic unless all of firm i's edges share the same manager
composition. Collapsing is valid only in the match-grouped case; outside it averaging
destroys the manager linkage that identifies σ_a² under free Ωᵢ.

---

## 5. API contract

Add two keywords (alongside `error_cov`, `error_groups`, `error_group_cap`):

```
suffstats(::Type{M}, f_idx, w_idx, y; …, error_blocks = nothing, firm_group = nothing, …)
fit_mle(::Type{M}, f_idx, w_idx, y;      …, error_blocks = nothing, firm_group = nothing, …)
```

- `error_blocks::Union{Nothing,Symbol}` — only `:free` in scope.
- `firm_group::Union{Nothing,AbstractVector{<:Integer}}` — firm id per input row;
  blocks discovered from it, never parametrized.
- Rules: `error_blocks` mutex with `error_cov` and `error_groups`; requires
  `Weighting(observations=:raw)`; EM solver only; composes with `match_id` (matched
  block → single-residual degenerate case); **not with `X` in v1** (profiled mean under
  block-Ω deferred to M-E).
- Reported `bar σ²` lands in `result.sigma_epsilon` (aliased `sigma_epsilon_mean` in
  metadata); per-block matrices **not retained** — only traces.
- Final naming is M-E (maintainer reconciliation); implement with `error_blocks` +
  `firm_group` now.

---

## 6. Data model

```julia
struct FreeBlockStats
    block_of::Vector{Int}     # edge-level row s → block index
    sizes::Vector{Int}        # m_i
    distinct::Vector{Int}     # d_i (distinct managers) — drives rank-def rejection
    V::SparseMatrixCSC       # edge-level design rows (K × n), NOT collapsed
    y::Vector{Float64}       # edge-level standardized outcomes
    dof_blocks::Int          # Σ_i m_i(m_i+1)/2
end
```

Field on `BipartiteGMRFStats`: `free_blocks::Union{Nothing,FreeBlockStats}`
(parallel to `error_classes`). Design rows from the existing `observation_rows`
(already gives edge-level `V`/`y`).

**Duplicate-row rejection (ship-first, §5.4.1).** At `suffstats`, when
`error_blocks=:free`, compute `d_i` per block; if `m_i > d_i` (a firm–manager pair
repeats within a block), **throw** with firm index, `m_i`, `d_i`. No constrained family
in v1; no ridge as default. Matched rows under `match_id` collapse to distinct managers
by construction, so this composes there.

---

## 7. Estimation algorithm (EM)

New file `src/solvers/emfreeblocks.jl`. One iteration:

1. **Per-firm block Cholesky.** `chol(Symmetric(Ωᵢ))` → `1'Ωᵢ⁻¹1`, `1'Ωᵢ⁻¹Fᵢ`,
   `Fᵢ'Ωᵢ⁻¹Fᵢ`, `Ωᵢ⁻¹yᵢ`. `O(Σ mᵢ³)`, parallel across firms. Never materialize Ω globally.
2. **Assemble** `A'Ω⁻¹A` + `A'Ω⁻¹y`; `M = K + A'Ω⁻¹A`.
3. **E-step** (one `GMRFWorkspace` for `M`, `update_precision!` per iter):
   `hat α = solve_M(A'Ω⁻¹y)`; `r = y − A hat α`; `S_{ε,i} = r_i r_i' + A_i M⁻¹ A_i'`
   via `selinv_extract_at(ws_M, A_i A_i')`; `S_α` moments via `selinv_dot`.
4. **M-step Ω:** `Ωᵢ ← S_{ε,i}`; `bar σ² = Σ tr(S_{ε,i})/m`.
   **Guardrail:** `cholesky(Symmetric(S_{ε,i}))`, throw with firm idx / mᵢ / dᵢ on
   failure (never silent, §7.5). Rank-def blocks excluded by §6 rejection.
5. **GMRF step:** one damped Newton step on `(ρ, σ_a, σ_z)` solving
   `tr(∂K/∂θ_k (K⁻¹ − S_α)) = 0`. Gradient exact via
   `selinv_dot(ws_Q, ∂K/∂θ_k) − [hat α' ∂K/∂θ_k hat α + selinv_dot(ws_M, ∂K/∂θ_k)]`.
   Needs analytic `∂K/∂θ` for `BipartiteVarianceStableModel` (three sparse matrices,
   derivable from the diagonal/cross formulas in `types.jl`). Hessian: finite-diff of
   the 3-vector gradient for v1; analytic Fisher as stretch.

Convergence: relative change in joint ℓ (marginal NLL at current (θ, Ω)) below `ftol`;
cap `max_iter`. Marginal NLL from §2 via the existing `nll_exact_value`-style
construction — also what criterion 1's dense reference compares against.

**Solver type:** `struct EMFreeBlocks <: AbstractGMRFSolver` (`max_iter`, `ftol`,
`newton_damp`, `abstol`). `validate_capability` gates: `error_blocks=:free` requires the
EM solver (throw otherwise, like `error_groups`→ExactCholesky). `solve` dispatches via a
new `optimize_problem_emfree` branch so the `fit_mle`/`solve`/`build_gmrf_result` flow
is unchanged.

---

## 8. Acceptance criteria → tests (`test/test_error_blocks.jl`)

1. **Dense-Σ reference (criterion 1).** Plant known free blocks; hand-build
   `Σ = A K⁻¹ A' + Ω`; assert EM fixed point = exact `ℓ` and `(ρ, σ_a, σ_z)` to `1e-8`.
2. **Reduction to iid (criterion 2).** All blocks size 1 (or `Ωᵢ = σ²I` planted) →
   bit-reproduce plain `suffstats` (`design.VtV/projected_y/ydot` ≈ `1e-12`) and the
   plain fit. Milestone M-A's `matches_Ri_identity`.
3. **Reduction to `error_groups` (criterion 3) — re-scoped.** See §8.1.
4. **Identification MC gate (criterion 4).** Fixed graph, redraw `a,z,ε` many times →
   `(ρ̂, σ̂a, σ̂z)` centered at truth; `σ̂a` dispersion grows on bridge-heavy graphs.
   Script/experiment, **not** a unit test.
5. **PD + rank-def guardrails (criteria 5, 6).** Cholesky failure throws with firm
   idx/mᵢ/dᵢ; duplicated-manager block rejected at `suffstats`/`fit_mle`.
6. **ρ=0 behavior (criterion 7).** Warn + set `metadata.unidentified_at_rho_zero` /
   `sigma_a_network_identified`; never silently return a point estimate.

### 8.1 Criterion 3 — re-scoped (reviewer correction)

The free-block MLE is unconstrained per firm (Σᵢ mᵢ(mᵢ+1)/2 free entries) while the
grouped MLE has C free ω's. On a single finite sample the two do **not** coincide
pointwise — the unconstrained fit absorbs per-firm noise (rank-one `rᵢrᵢ'` + posterior
term) that the grouped estimator pools away. Issue §5.1(ii) states this outright:
each Ω̂ᵢ is individually a terrible estimate, only pooled traces/means are meaningful.
Requiring pointwise agreement on one planted sample would be a flaky/invalid criterion.

Replace the pointwise claim with two valid checks:

1. **Algebraic constrained-mode equivalence (deterministic unit test).** Constrain Ωᵢ
   to equicorrelation with one synthetic ω per size class; assert the free-block
   marginal NLL equals the grouped path's NLL at the same ω to machine precision. They
   are the *same statistical model* under that constraint; this proves the free-block
   path is a correct superset of the grouped path.
2. **Population consistency (Monte-Carlo, with criterion 4).** Plant equicorrelated
   blocks, redraw `a,z,ε` many times; assert the **average** per-size-class free-block ω
   ladder is centered on the grouped `error_class_variances` (no first-order bias).

---

## 9. Files touched

- `src/types.jl` — `FreeBlockStats`, `EMFreeBlocks`, `BipartiteGMRFStats.free_blocks`.
- `src/prepare.jl` — `build_freeblock_V_stats` (block discovery via union-find on
  `firm_group`, rank-def rejection), reuse `observation_rows`.
- `src/solvers/emfreeblocks.jl` (new) — EM loop, `∂K/∂θ`, Newton step, marginal NLL.
- `src/solvers/common.jl` — `validate_capability`, `solve`/`optimize_problem` dispatch,
  `build_gmrf_result` metadata.
- `src/stats.jl` — `suffstats` keywords + validation; `dof` (reconcile exact counting
  with maintainer in M-C).
- `src/fit.jl` — keyword pass-through.
- `src/api.jl` — `coef`/`params`/`show` surface `sigma_epsilon` as `bar σ²`; caveats.
- `src/BipartiteGMRF.jl` — export + include order.
- `test/test_error_blocks.jl` (new), `test/runtests.jl`, `CHANGELOG.md`, `docs/src`.

---

## 10. Milestones (issue §9, concretized)

- **M-A** — EM scaffold + symbolic-factorization de-risk probe (R1); `matches_Ri_identity`.
- **M-B** — dense-Σ reference (criterion 1), exact-`nll` match to `1e-8`.
- **M-C** — `bar σ²` pinning, `dof`, metadata; algebraic constrained-mode equivalence +
  population reduction to `error_groups` (§8.1).
- **M-D** — MC gate + giant-firm guardrail (`max mᵢ` warning; Woodbury fallback is the
  §5.2 stretch, not v1) + ρ=0 warning.
- **M-E** — API naming with maintainer, docs, Aqua/JET, full suite, existing
  `error_cov`/`error_groups` tests untouched.

---

## 11. Risks & de-risk order

- **R1 (RESOLVED).** `update_precision!` enforces an exact fixed pattern; an
  iid-seeded workspace rejects the free-Ω `M` (probing confirmed, `dev/probe_r1.jl`),
  so the EM seeds `ws_M` from `K + A'Ω₀⁻¹A` with Ω₀ carrying full within-firm
  off-diagonal support. That fixed superset pattern holds across EM iterations, and
  `logdet`/`selinv_diag`/`selinv_extract_at` match a dense reference to ~1e-16.
- **R2:** incidental-parameters bias — gated by criterion 4, not by proof.
- **R3:** `σ_a` network-only identification — metadata must say so; do not read noisy
  `σ̂a` as "no firm effect".
- **R4:** Newton-step Hessian choice — v1 finite-diff; analytic if MC gate slow.
- **R5:** `dof` counting convention for free blocks — resolve with maintainer in M-C;
  does not block M-A/M-B.

---

## 12. Decisions (defaults; reverse on call)

- Solver is a **new `EMFreeBlocks` type**, not a branch inside `ExactCholesky`.
- v1 ships **rejection** for duplicate firm–manager rows (§5.4.1); no constrained
  family, no ridge.
- v1 supports **ungrouped + `match_id`-composed** blocks; `X` deferred to M-E.
- `bar σ²` reuses `result.sigma_epsilon` (aliased `sigma_epsilon_mean`); per-block
  matrices dropped, only traces kept.