# Changelog

All notable changes to BipartiteGMRF.jl are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v0.5.1] &mdash; 2026-08-20

### Fixed

- **`EMIWBlocks` E-step no longer scales with the node count** (issue #116).
  The per-firm posterior correction `Aᵢ M⁻¹ Aᵢ'` did one full triangular solve
  against the whole factorization per block row — `K_tot` global solves per EM
  iteration, each allocating a dense length-`n` vector, which made fits on
  graphs with ~10⁶ firms effectively unrunnable. It now reads one selected
  inverse of `M` per E-step at the block pattern and forms the correction
  locally. The PSD guarantee the previous Gram form was written for is kept:
  the block's `Σ_SS` is a principal submatrix of `M⁻¹`, hence PD, and the
  result is returned as `(Aᵢ L)(Aᵢ L)'` from its Cholesky — which matters,
  because `S_{ε,i}` feeds a Gamma rate that is then passed to `log`. Measured
  on a synthetic graph with 8,000 firm blocks: 61 s and 25 GiB of allocation
  per E-step pass before, 32 ms and 23 MiB after, with results agreeing to
  1.8e-15.

## [v0.5.0] &mdash; 2026-08-19

### Added

- **Warm starts** (`init`, issue #114). `fit_mle`, `solve` and
  `optimize_problem` take an optional `init` NamedTuple of starting values,
  replacing the fixed heuristic in `initial_params` field by field: `rho`,
  `sigma_a`, `sigma_z`, `sigma_epsilon`, plus `rho_eps`, `eta` (`error_eta`),
  `omega` (`error_groups`) and `phi`/`r`/`delta` (`error_blocks = :iw`).
  Values are read in **original outcome units**, so `init = params(result)`
  restarts a previous fit and a minimum-distance pilot estimate needs no
  rescaling. A field for a parameter the fit does not estimate is an error; one
  for a parameter that is pinned (`fix_rho`, a numeric `error_eta`, a fixed
  solver `r`/`delta`) warns and is ignored. `init = nothing` reproduces the
  previous starting point exactly. See `docs/src/performance.md` for when a
  warm start does and does not save time.
- **Nelder-Mead initial simplex controls** (`simplex_scale`, `simplex_shift` on
  `ExactCholesky` and `HutchSLQ`, issue #115). Vertex `j+1` of the initial
  simplex is `(1 + simplex_scale)·x_j + simplex_shift`. The defaults (0.5 and
  0.025) are Optim's own, so existing fits are unchanged; lowering
  `simplex_scale` is what makes a warm start actually search locally rather
  than re-spanning the region around it. `simplex_shift` is the absolute term
  that keeps a coordinate sitting at exactly zero — which warm starts routinely
  produce (`log ω = 0`, `atanh(η) ≈ 0`) — from degenerating; setting both to
  zero is rejected.

### Changed

- `optimize_emiw`'s private `init_theta` / `init_phi` keywords are replaced by
  the single `init` NamedTuple, which accepts partial specifications and is
  validated (the old `init_theta` reached `atanh` unguarded).

## [v0.4.1] &mdash; 2026-08-19

### Added

- **Inverse-Wishart per-firm error blocks** (`error_blocks = :iw`,
  `firm_group`, `EMIWBlocks`, issue #112). The per-firm error covariances
  `Ω_i` are treated as nuisance realizations drawn iid from an inverse-Wishart
  population law and integrated out — never estimated — via variational EM
  over `q(α)·∏q(u_i)`. Estimated hyperparameters `(ρ, σ_a, σ_z, φ, r, δ)`
  (mean error scale, within-firm correlation, and Student-t tail dof) are
  reported in the result metadata (`error_scale_phi`, `error_corr_r`,
  `t_dof_delta`, `omega_bar`); the objective is the ELBO, a lower bound on the
  integrated log-likelihood. Requires `BipartiteVarianceStableModel`.

## [v0.4.0] &mdash; 2026-08-19

### Added

- **Correlated errors** (`error_cov = R`). The error covariance becomes
  `σ_ε² R`, a sparse symmetric matrix over input rows whose connected blocks
  are arbitrary positive-definite matrices (any correlation pattern, any
  within-block heteroskedasticity). The scale is pinned by `tr(R) = K` so
  `σ_ε²` keeps its mean-error-variance meaning.
- **Group-robust errors** (`error_groups = g`). Observations sharing a group id
  are collapsed to their group mean, so an arbitrary unknown PD within-group
  error covariance enters the likelihood only through the group-mean variance.
  One free `ω` per group-size class is estimated inside the MLE and reported in
  the result metadata (`error_class_sizes`, `error_class_variances`).
- **AR(1) within-firm error correlation** (`error_eta`, `edge_index`, issue
  #113). The error covariance is `σ_ε² R(η)` with `R(η)` block-diagonal by firm
  and `Corr(ε_k, ε_l) = η^|k-l|` over the per-row within-firm `edge_index`
  (a unique permutation of `1..m_i` per firm). `η` is fixed in `(-1, 1)` or
  estimated jointly via `error_eta = :estimate`, and reported as `result.eta`;
  `dof` counts it. Requires `Weighting(observations=:raw)` and the
  `ExactCholesky` solver.

## [v0.3.0] &mdash; 2026-08-17

### Added

- **GaussianMarkovRandomFields.jl integration.** All four model types
  (`BipartiteNormalizedModel`, `BipartiteUnnormalizedModel`,
  `BipartiteSpectralModel`, `BipartiteVarianceStableModel`) implement
  `GaussianMarkovRandomFields.LatentModel`, giving access to `rand`,
  `var`, `logpdf`, `GMRFWorkspace` reuse, and selected inversion.
- **`suffstats` / `fit_mle` interface** following the Distributions.jl
  convention. Data enters as three parallel vectors
  `(f::Vector{Int}, w::Vector{Int}, y::Vector{Float64})` &mdash; one
  entry per observed spell.
- **`StatsAPI.StatisticalModel` interface** on `GMRFResult`: `coef`,
  `coefnames`, `params`, `loglikelihood`, `nobs`, `dof`, `aic`, `bic`,
  `converged`, `isfitted`, `islinear`.
- **`simulate(model, f, w; ...)`** for Monte Carlo data generation from
  index vectors (plus an adjacency-matrix overload).
- **`decompose(result; kind, probes, target)`** for model and fitted
  variance decompositions, decoupled from estimation.
- **`covariance(result)` / `cov_block(op; firms, workers)`** for
  factorized covariance extraction by node index.
- **`ExactCholesky` workspace reuse** via `GMRFWorkspace` &mdash;
  symbolic factorization computed once and reused across likelihood
  evaluations.
- **`Weighting` type** bundling observation-model choices (`:raw`,
  `:edge`, `:effective`) with optional joint `rho_eps` estimation.
- **`rho_limit = :auto`** for `BipartiteVarianceStableModel`, resolved
  from the non-backtracking spectrum at model construction time.
- **Graph-only edges** (issue #106, capability 1). Non-finite values in
  `y` (NaN) mark edges that enter the prior adjacency `A_prior` and the
  model's precision matrix but contribute nothing to the likelihood
  (`V'V`, `V'y`, `y'y`, `K`).
- **Match-grouped observations** (issue #106, capability 2). New
  `match_id` kwarg on `suffstats` and `fit_mle`: edges sharing a match
  id form one observation whose design row averages `1/F_s` over firms
  and `1/M_s` over workers. V'V acquires off-diagonal firm-firm and
  worker-worker blocks. Outcomes within a match must agree; mixing
  finite and NaN outcomes in a match is rejected.

- **Profiled-out mean structure** (issue #109). New `X` kwarg on
  `suffstats` and `fit_mle` accepts a $K \times p$ design matrix. The
  coefficients $\hat\beta(\theta)$ are computed in closed form at each
  likelihood evaluation via $p$ extra solves against the same
  factorization. `result.beta` reports the profiled coefficients in
  original outcome units; `coef`, `coefnames`, `params`, and `dof`
  include them automatically. `simulate` accepts optional `X` and `β`
  kwargs. Works with all model types, both solvers, and all weighting
  and match-grouping modes.

### Changed

- **`BipartiteVarianceStableModel` marginal-SD parametrization** (issue
  #108, **breaking**). The precision matrix is now multiplied by
  $1/(1-\rho^2)$, so $\sigma_a$ and $\sigma_z$ are marginal standard
  deviations on a forest: $\text{Cov}(a_i, a_j) = \sigma_a^2\,\rho^{d(i,j)}$
  with no correction factor. Previously, the marginal variance on a forest
  was $\sigma_a^2/(1-\rho^2)$. To convert old parameters to the new
  convention: $\sigma_a^{\text{new}} = \sigma_a^{\text{old}} / \sqrt{1-\rho^2}$.
  The change touches `precision_matrix`, `QOpVS`, `set_q_params!`, and
  `q_diag`; all downstream code (decomposition, covariance, simulation) is
  unaffected because it reads from $\mathbf{Q}^{-1}$.

- **Data-agnostic API.** `suffstats` and `fit_mle` accept integer index
  vectors instead of DataFrames. Mapping entity identifiers to dense
  1-based indices and filtering unusable rows are the caller's
  responsibility.
- **DataFrames dependency removed** entirely. Edge collapse is now a
  small dict-based routine in `prepare.jl`.
- **`cov_block` addresses nodes by index**, not by entity-ID
  dictionaries. The `firms` / `workers` kwargs take integer vectors.
- **`GMRFResult` no longer stores decompositions.** Use
  `decompose(result)` after fitting instead of the removed `decompose`
  kwarg on `solve` / `fit_mle`.
- **`BipartiteGMRFStats` restructured** into `DesignStats` (V'V, V'y,
  y'y), `EdgeData` (parallel index/outcome arrays), and `WeightStats`
  sub-structs. The two hand-written 36-field copy constructors are
  replaced by a generic `replace_stats` helper.
- **`GMRFProblem` removed.** The model + stats separation replaces the
  monolithic problem struct.
- **`ModelSpec` / `Prior` layer removed.** Model types are constructed
  directly from adjacency matrices; `fit_mle` is the entry point.
- **NB spectrum and feasibility live on the model**, not on stats.
  `feasibility(model)` no longer needs the stats object.
- **Solver dispatch refactored.** Cache construction, NLL evaluation,
  Nelder-Mead tolerance, and the polish stage dispatch on the solver
  type; duplicated NLL code removed.
- **`Weighting` stores `estimate_rho_eps::Bool`** instead of a Symbol
  sentinel.
- **Shared Hutchinson probe loop** for model/fitted decompositions;
  decomposition now uses sparse `FF`/`WW` block multiplies instead of
  scalar `cnt_f`/`cnt_w` loops, supporting off-diagonal blocks from
  match-grouped observations.
- **`CovarianceOperator` fully parametrized** &mdash; no abstractly-typed
  fields.
- **`DesignStats` stores `FF` and `WW`** (firm-firm and worker-worker
  blocks of V'V) as sparse matrices instead of `cnt_f`/`cnt_w` vectors.

### Removed

- `max_degree`, `on_missing`, `outcome`, `firm_id`, `worker_id` kwargs
  (DataFrame-era column-name plumbing).
- `GMRFProblem` struct and its constructors.
- `ModelSpec` / `AbstractGMRFPrior` types.
- `decompose` kwarg on `solve` / `fit_mle`.
- Legacy wrapper functions and deprecated aliases.
- DataFrames.jl as a dependency (including test dependency).

### Fixed

- **Runtime dispatch in `cov_block` eliminated** (JET `@test_opt` clean).
- **Failed PCG probes in decompositions** are now skipped regardless of
  the `verbose` setting, preventing silent NaN propagation.
- **Likelihood constants** corrected in the
  Distributions/StatsAPI/CommonSolve generics refactor.

## [0.1.0] &mdash; 2026-03-16

Initial release accompanying the working paper. DataFrame-based API with
`GMRFProblem` as the main entry point.
