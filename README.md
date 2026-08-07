# BipartiteGMRF.jl

[![CI](https://github.com/ulrichwohak/BipartiteGMRF.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/ulrichwohak/BipartiteGMRF.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ulrichwohak/BipartiteGMRF.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ulrichwohak/BipartiteGMRF.jl)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://ulrichwohak.github.io/BipartiteGMRF.jl/stable)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19048278.svg)](https://doi.org/10.5281/zenodo.19048278)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Julia tools for fitting bipartite-graph Gaussian Markov random-field
random-effects models. The library places a joint Gaussian model on two sets
of latent node effects on a bipartite graph &mdash; firm and worker effects
in the canonical labor-economics application, but applicable to any matched
bipartite structure &mdash; and exposes model and fitted variance
decompositions, covariance-block extraction, and simulation over the latent
field.

Built on [GaussianMarkovRandomFields.jl](https://github.com/timweiland/GaussianMarkovRandomFields.jl):
the bipartite model types implement the `LatentModel` interface, giving you
`rand`, `var`, `logpdf`, workspace reuse, and selected inversion for free.

The parameter $\rho$ governs the *local conditional dependence* between
adjacent nodes in the graph and admits an interpretation as local,
edge-level assortative matching. It is **not** the AKM/KSS-style sorting
parameter (the population correlation between fitted firm and worker fixed
effects), with which it coincides only under restrictive network conditions
and from which it can diverge sharply in sparse mobility networks. See the
paper for the contrast.

The package was developed as the computational backbone of

> Koren, Mikl&oacute;s; Wohak, Ulrich; Orb&aacute;n, Krisztina; Telegdy, &Aacute;lmos
> (2026). *A Random-Effects Model Reveals Strong Positive Sorting in CEO Labor
> Markets.* Working paper, March 16, 2026.

and is released for reuse in any setting where bipartite-graph random effects
are of interest &mdash; worker&ndash;firm panels, student&ndash;school
achievement data, patient&ndash;provider health outcomes, and similar matched
structures.

## Background

Let $i$ index firms and $m$ index workers (or, generically, the two sides of
a bipartite pairing). For each observed spell $k$, the outcome $y_{im}$ is
modeled as

$$
y_{im} = \mathbf{x}_{im}'\boldsymbol{\beta} + a_i + z_m + \varepsilon_{im},
$$

where $\mathbf{x}_{im}$ are observation-level covariates whose coefficients
$\boldsymbol{\beta}$ are profiled out of the likelihood in closed form,
$a_i$ and $z_m$ are jointly Gaussian on the bipartite graph and
parametrized by $(\rho, \sigma_a, \sigma_z, \sigma_\varepsilon)$. Stacking
the latent effects as

$$
\mathbf{x} = (a_1, \ldots, a_{N_f}, z_1, \ldots, z_{N_m})^\top,
$$

the precision matrix is

$$
\mathbf{Q} = \mathbf{S}^{-1}\left(\mathbf{D} - \rho\mathbf{A}\right)\mathbf{S}^{-1},
$$

where $\mathbf{A}$ is the bipartite adjacency matrix, $\mathbf{D}$ is the
diagonal degree matrix, and
$ \mathbf{S} = \operatorname{diag}(\sigma_a \mathbf{1}_{N_f}, \sigma_z \mathbf{1}_{N_m}) $
is the variance-scaling matrix. The parameter $\rho \in (-1, 1)$ governs the
strength and sign of local dependence between linked nodes.

Replacing hundreds of thousands of latent effects with four distributional
parameters has three practical consequences:

1. **No limited-mobility bias.** Two-way fixed-effects estimators on sparse
   networks attenuate or even flip the sign of estimated sorting. The
   random-effects approach side-steps this by estimating variance components
   from the full covariance structure rather than from point estimates of
   individual effects.
2. **Likelihood-based estimation scales.** The precision matrix $\mathbf{Q}$
   is sparse, so its log-determinant and quadratic forms can be evaluated
   with sparse Cholesky, or, for very large networks, with the Hutchinson
   trace estimator and stochastic Lanczos quadrature on sparse
   matrix-vector products.
3. **Uses the full network.** Disconnected components, leaves, and high-degree
   hubs all contribute likelihood-information rather than being dropped.

Applied to the universe of Hungarian CEO&ndash;firm spells, 1990&ndash;2018
($N_f = 530{,}213$, $N_m = 617{,}613$, $K = 1{,}131{,}996$), the baseline
local-dependence estimate is $\hat{\rho} = 0.706$ &mdash; strong positive
local sorting in the GMRF, not an AKM/KSS-style sorting correlation.
For comparison, the corresponding two-way fixed-effects estimate on the same
sample is $-0.64$; the leave-one-out (Kline&ndash;Saggio&ndash;S&oslash;lvsten)
bias correction reduces the magnitude to $-0.39$ but does not resolve the
sign. See the paper for details and counterfactuals.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/ulrichwohak/BipartiteGMRF.jl")
```

Or, for local development:

```julia
using Pkg
Pkg.develop(path = ".")
Pkg.test("BipartiteGMRF")
```

The library requires Julia 1.10 or newer and depends on
`GaussianMarkovRandomFields`, `Distributions`, `Optim`, `LinearSolve`, and
`FiniteDiff`. It is an estimator, not a data manipulator: it does not read or
write files and does not know about tables or column names &mdash; pass it
integer-indexed observation vectors, receive typed results.

## Quick Start

### Estimation

The package extends the [Distributions.jl](https://juliastats.org/Distributions.jl/stable/fit/)
`suffstats` / `fit_mle` generic functions (so `using Distributions,
BipartiteGMRF` involves no name clashes). Data enters as three parallel
vectors: 1-based firm index, 1-based worker index, and outcome, one entry per
observed spell. Non-finite values in `y` (NaN) mark *graph-only* edges that
enter the prior graph but not the likelihood. Mapping entity identifiers to
dense indices is the caller's job:

```julia
using BipartiteGMRF

f = [1, 1, 2, 2, 3, 3]      # firm index per observation
w = [1, 2, 2, 3, 3, 4]      # worker index per observation
y = [1.2, 0.7, 0.9, 1.5, 1.1, 0.4]

# One-step
result = fit_mle(BipartiteNormalizedModel, f, w, y;
    solver = ExactCholesky(),
    seed   = 42,
)

result.rho           # local dependence parameter
result.sigma_a       # firm effect SD
result.sigma_z       # worker effect SD
result.sigma_epsilon # residual SD
```

#### Mean structure

Pass a design matrix `X` (one row per observation) to profile out a
linear mean $\mathbf{X}\boldsymbol{\beta}$. At each optimizer iteration
$\hat{\boldsymbol{\beta}}(\theta)$ is computed in closed form from $p$
extra solves against the same factorization &mdash; negligible cost:

```julia
# Degree-dependent mean: y = β₀ + β_f·d_f + β_m·d_m + a + z + ε
X = hcat(ones(length(f)), Float64.(f), Float64.(w))  # K × 3

result = fit_mle(BipartiteNormalizedModel, f, w, y;
    X      = X,
    solver = ExactCholesky(),
)

result.beta          # profiled-out [β₀, β_f, β_m]
```

Or separate sufficient-statistics computation from estimation:

```julia
# Precompute sufficient statistics (reusable across model types)
ss = suffstats(BipartiteNormalizedModel, f, w, y)

# Fit
result = fit_mle(BipartiteNormalizedModel, ss; solver = ExactCholesky())
```

`GMRFResult` implements `StatsAPI.StatisticalModel`:

```julia
using StatsAPI

coef(result)          # [rho, sigma_a, sigma_z, sigma_epsilon]  (+ rho_eps, beta... if used)
coefnames(result)     # ["rho", "sigma_a", "sigma_z", "sigma_epsilon", ...]
params(result)        # (rho=..., sigma_a=..., sigma_z=..., sigma_epsilon=..., rho_eps=..., beta=...)
loglikelihood(result) # original-units log-likelihood (constants included)
nobs(result)
dof(result)           # number of *estimated* parameters
aic(result)
bic(result)
```

### Variance Decomposition

```julia
model_vd  = decompose(result; kind = :model,  probes = 200)
fitted_vd = decompose(result; kind = :fitted, probes = 200)
```

`kind = :model` decomposes variance using the GMRF's precision structure
alone. `kind = :fitted` includes fitted effects (mode + trace correction).

### Covariance Extraction

```julia
op    = covariance(result; kind = :model, units = :original)
block = cov_block(op; firms = [1, 2], workers = [1, 2])  # node indices
```

### Simulation

```julia
using Random

# Simulate from index vectors (same format as estimation input)
model = BipartiteNormalizedModel(sparse([1,1,2,2,3], [1,2,2,3,3],
                                       ones(5), 3, 3))

sim = simulate(model, f, w; ρ = 0.5, σ_a = 1.0, σ_z = 0.8, σ_ε = 0.3,
               rng = Xoshiro(42))
sim.y               # outcome vector
sim.firm_effects    # sampled α
sim.worker_effects  # sampled z

# With a mean structure: y = Xβ + a + z + ε
X = hcat(ones(length(f)), Float64.(f))
sim = simulate(model, f, w; ρ = 0.5, σ_a = 1.0, σ_z = 0.8, σ_ε = 0.3,
               X = X, β = [1.0, 0.5], rng = Xoshiro(42))
```

An adjacency-matrix overload `simulate(model, A; ...)` is also available.

### Direct GMRF Access

The model types implement `GaussianMarkovRandomFields.LatentModel`, so you
get the full GMRF.jl interface:

```julia
using GaussianMarkovRandomFields: var, logpdf

gmrf = model(; ρ = 0.5, σ_a = 1.0, σ_z = 0.8)
x = rand(gmrf)        # sparse Cholesky sampling
v = var(gmrf)          # marginal variances via selected inversion
l = logpdf(gmrf, x)    # log-density
```

## Models and Solvers

The library separates the **model type** (the shape of $\mathbf{Q}$) from
the **numerical solver** (how the likelihood is evaluated and optimized).

Model types (`AbstractBipartiteModel <: LatentModel` subtypes):

- `BipartiteNormalizedModel` &mdash; degree-normalized Laplacian (default).
- `BipartiteUnnormalizedModel` &mdash; the $\mathbf{D} - \rho\mathbf{A}$
  precision used in the paper.
- `BipartiteSpectralModel` &mdash; spectral normalization by
  $\sigma_1(\mathbf{A})$.
- `BipartiteVarianceStableModel` &mdash; variance-stable $\mathbf{Q}$ with
  marginal-SD parametrization: $\sigma_a$ and $\sigma_z$ are the marginal
  standard deviations on a forest, so $\text{Cov}(a_i, a_j) = \sigma_a^2
  \rho^{d(i,j)}$ without a $1/(1-\rho^2)$ correction factor. The precision
  is $\mathbf{Q} = \frac{1}{1-\rho^2}\,\mathbf{S}^{-1}[(1-\rho^2)\mathbf{I}
  + \rho^2\mathbf{D} - \rho\mathbf{A}]\,\mathbf{S}^{-1}$.

Solvers (`AbstractGMRFSolver` subtypes):

- `ExactCholesky()` &mdash; deterministic sparse CHOLMOD factorization with
  symbolic reuse via `GMRFWorkspace`; gradient-free Nelder-Mead search
  followed by an optional finite-difference L-BFGS polish.
- `HutchSLQ()` &mdash; Hutchinson trace estimator plus stochastic Lanczos
  quadrature for the log-determinant, preconditioned conjugate gradient for
  the quadratic form, and Nelder-Mead for optimization.

Unsupported model/solver combinations throw `ArgumentError` before fitting.

## Observation Weighting

`Weighting` bundles the observation-model choices:

```julia
Weighting(observations = :raw)
Weighting(observations = :edge)
Weighting(observations = :effective, rho_eps = 0.5)
Weighting(observations = :effective, rho_eps = :estimate)
```

`:raw` treats each row as an independent draw; `:edge` collapses to
match-means with equal edge weights; `:effective` uses match-effective weights
with a within-match residual correlation `rho_eps` (either fixed or
estimated jointly).

Decomposition targets are `:estimation`, `:personyear`, and `:edge`.

## Graph-Only Edges and Match Grouping

### Graph-only edges

An edge whose outcome is unknown can stay in the graph (affecting the
precision matrix `Q` and node degrees) without contributing to the
likelihood. Pass `NaN` as the outcome:

```julia
f = [1, 1, 2, 2, 3]
w = [1, 2, 2, 3, 1]
y = [1.2, 0.7, 0.9, 1.5, NaN]   # edge (3,1) is graph-only

result = fit_mle(BipartiteNormalizedModel, f, w, y; solver = ExactCholesky())
```

### Match-grouped observations

When multiple edges share a single outcome (e.g. a firm managed by two
CEOs simultaneously), pass a `match_id` vector. Edges with the same
match id form one observation whose design row averages `1/F_s` over
firms and `1/M_s` over workers:

```julia
f   = [1, 2, 1]        # match 1 has firms {1,2}; match 2 has firm {1}
w   = [1, 1, 2]        # match 1 has worker {1};  match 2 has worker {2}
y   = [1.0, 1.0, 0.5]  # same outcome within a match (enforced)
mid = [1, 1, 2]

result = fit_mle(BipartiteNormalizedModel, f, w, y;
    match_id = mid, solver = ExactCholesky())
```

Outcomes must be identical within a match; disagreement raises
`ArgumentError`. Currently requires `Weighting(observations = :raw)`.

## Repository Layout

```
src/
├── BipartiteGMRF.jl       # module entry point
├── types.jl               # LatentModel subtypes, BipartiteGraph, stats structs, GMRFResult
├── stats.jl               # suffstats(): arrays → BipartiteGMRFStats
├── fit.jl                 # fit_mle(): type-based and model-based entry points
├── prepare.jl             # edge collapse, V'V construction, weighting helpers
├── util.jl                # replace_stats, scaled_params, shared helpers
├── operators/             # QOp/QOpVS per model type
├── linalg/                # PCG, SLQ
├── solvers/               # shared optimize loop + ExactCholesky / HutchSLQ methods
├── decomposition/         # model.jl, fitted.jl
├── nonbacktracking/       # NB spectrum, feasibility
├── covariance/            # operator, block extraction
├── simulate.jl            # Monte Carlo simulation
└── api.jl                 # decompose, covariance, StatsAPI methods
```

Project-specific data preparation, estimation, and post-estimation scripts
from the original research codebase are archived with the reproducibility
compendium rather than tracked in this reusable library repository.

## Performance

`HutchSLQ` is designed for networks where dense $\mathbf{Q}^{-1}$ is
infeasible. On the baseline CEO&ndash;firm sample
($N_f + N_m \approx 1.15$ million, $K \approx 1.13$ million), one
likelihood evaluation under `HutchSLQ` takes a few seconds on a single core;
full Nelder-Mead optimization converges in a few hundred evaluations.
`ExactCholesky` is preferred when the bipartite graph is small enough that
CHOLMOD fill-in is tolerable; switch to `HutchSLQ` when factorization memory
becomes the bottleneck.

For typical workloads, set `BLAS.set_num_threads(1)` at the start of your
session. The library does not mutate global BLAS state itself.

## Reproducibility

The original research compendium &mdash; data preparation pipeline,
estimation scripts, paper artifacts, and run logs &mdash; is archived on
Zenodo:

> [https://zenodo.org/records/19048278](https://zenodo.org/records/19048278)

This repository hosts the reusable library extracted from that compendium.
The project-specific pipeline scripts are available in the Zenodo archive for
direct reproducibility.

## Citation

If you use `BipartiteGMRF.jl` in academic work, please cite both the package
and the underlying paper.

**Paper:**

```bibtex
@article{koren_wohak_orban_telegdy_2026_ceo_sorting,
    author  = {Koren, Mikl{\'o}s and Wohak, Ulrich and Orb{\'a}n, Krisztina
               and Telegdy, {\'A}lmos},
    title   = {A Random-Effects Model Reveals Strong Positive Sorting in
               {CEO} Labor Markets},
    year    = {2026},
    note    = {Working paper, March 16, 2026}
}
```

**Software:**

```bibtex
@software{bipartite_gmrf_jl,
    author  = {Wohak, Ulrich and Koren, Mikl{\'o}s},
    title   = {{BipartiteGMRF.jl}: Bipartite-graph {GMRF} random-effects
               estimation in {J}ulia},
    year    = {2026},
    doi     = {10.5281/zenodo.19048278},
    url     = {https://zenodo.org/records/19048278},
    version = {0.1.0}
}
```

`CITATION.cff` in the repository root mirrors this information for GitHub's
"Cite this repository" UI.

## Related Packages

- [`GaussianMarkovRandomFields.jl`](https://github.com/timweiland/GaussianMarkovRandomFields.jl)
  &mdash; the GMRF framework this package builds on. Provides `LatentModel`,
  workspace reuse, selected inversion, and AD support.
- [`VarianceComponentsHDFE.jl`](https://github.com/HighDimensionalEconLab/VarianceComponentsHDFE.jl)
  &mdash; fixed-effects AKM with leave-one-out (Kline&ndash;Saggio&ndash;S&oslash;lvsten)
  bias correction. Complementary point of comparison; see Section 4 of the
  paper.
- [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) &mdash; underlying
  optimizer.

## Contributing

Issues and pull requests are welcome. For substantive changes, please open an
issue first to discuss scope. New model specifications, solver back-ends, and
observation-weighting schemes are particularly welcome contributions; please
include tests against a dense reference on small synthetic networks.

## License

`BipartiteGMRF.jl` is released under the MIT License. See [LICENSE](LICENSE)
for the full text.

## Acknowledgments

This research was funded by the European Research Council (ERC Advanced
Grant 101097789) and by the National Research, Development and Innovation
Office (OTKA contract numbers 143346 and 144193, and Frontline Research
Excellence Program contract number 144193). The views expressed in the paper
are those of the authors and do not necessarily reflect the official view of
the European Union, the European Research Council, or the National Research,
Development and Innovation Office.

We thank Gregory Clark for useful discussions.
