# Changelog

All notable changes to BipartiteGMRF.jl are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] &mdash; feat/gmrf-integration

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

### Changed

- **Data-agnostic API.** `suffstats` and `fit_mle` accept integer index
  vectors instead of DataFrames. Mapping entity identifiers to dense
  1-based indices and filtering unusable rows are the caller's
  responsibility. Non-finite outcomes are rejected, not silently dropped.
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
- **Shared Hutchinson probe loop** for model/fitted decompositions.
- **`CovarianceOperator` fully parametrized** &mdash; no abstractly-typed
  fields.

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
