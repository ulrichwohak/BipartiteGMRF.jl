# BipartiteGMRF.jl

[![CI](https://github.com/ulrichwohak/BipartiteGMRF.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/ulrichwohak/BipartiteGMRF.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://ulrichwohak.github.io/BipartiteGMRF.jl/stable)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19048278.svg)](https://doi.org/10.5281/zenodo.19048278)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Julia tools for fitting bipartite-graph Gaussian Markov random-field
random-effects models. The library places a joint Gaussian prior on two sets
of latent node effects on a bipartite graph &mdash; firm and worker effects
in the canonical labor-economics application, but applicable to any matched
bipartite structure &mdash; and exposes prior and posterior variance
decompositions plus covariance-block extraction over the latent field.

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
y_{im} = a_i + z_m + \varepsilon_{im},
$$

with $a_i$ and $z_m$ jointly Gaussian on the bipartite graph and
parametrized by $(\rho, \sigma_a, \sigma_z, \sigma_\varepsilon)$. Stacking
the latent effects as

$$
\mathbf{x} = (a_1, \ldots, a_{N_f}, z_1, \ldots, z_{N_m})^\top,
$$

the prior precision matrix is

$$
\mathbf{Q} = \mathbf{S}^{-1}\\left(\mathbf{D} - \rho\mathbf{A}\right)\mathbf{S}^{-1},
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
local sorting in the GMRF prior, not an AKM/KSS-style sorting correlation.
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

The library requires Julia 1.10 or newer and depends on `DataFrames`, `Optim`,
and `FiniteDiff`. It deliberately does not read or write Parquet, CSV, JSON,
or `estimates.txt` files &mdash; pass it a `DataFrame`, receive typed results.

## Quick Start

```julia
using BipartiteGMRF, DataFrames

result = gmrf_mle(
    df;
    outcome    = :y,
    firm_id    = :firm_id,
    worker_id  = :worker_id,
    prior      = NormalizedPrior(),
    solver     = ExactCholesky(),
    decompose  = 200,
    seed       = 42,
)

result.rho           # local dependence parameter; local sorting, not AKM/KSS sorting
result.sigma_a       # firm effect SD
result.sigma_z       # worker effect SD
result.sigma_epsilon # match-specific noise SD

posterior = posterior_decomposition(result; probes = 200, seed = 42)
block     = cov_block(prior_covariance(result); firms = [1, 2], workers = [10, 11])
```

`gmrf_mle` returns a `GMRFResult` containing point estimates, the `GMRFProblem`
the result was fit against, optimization diagnostics, and (optionally) prior
and posterior decompositions. Accessors `coef`, `nll`, `converged`, and
`prior_decomposition` are also exported.

## Priors and Solvers

The library separates the **prior precision model** (the shape of
$\mathbf{Q}$) from the **numerical solver** (how the likelihood is evaluated
and optimized).

Prior models:

- `NormalizedPrior()` &mdash; degree-normalized Laplacian (default).
- `UnnormalizedPrior()` &mdash; the $\mathbf{D} - \rho\mathbf{A}$ precision
  used in the paper.
- `SpectralPrior()` &mdash; spectral normalization by $\sigma_1(\mathbf{A})$.
- `VarianceStablePrior()` &mdash; variance-stable $\mathbf{Q}$ with diagonal
  $[1 + \rho^2 (d_i - 1)] / \sigma_i^2$, intended for spanning-tree
  subgraphs where it gives degree-independent marginal variances.

Solvers:

- `ExactCholesky()` &mdash; deterministic sparse CHOLMOD factorization, with
  finite-difference gradients fed to L-BFGS / Nelder-Mead.
- `HutchSLQ()` &mdash; Hutchinson trace estimator plus stochastic Lanczos
  quadrature for the log-determinant, preconditioned conjugate gradient for
  the quadratic form, and Nelder-Mead for optimization.

Unsupported prior/solver combinations throw `ArgumentError` before fitting.

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

## Covariance Extraction

```julia
prior_op = prior_covariance(result; units = :original)
post_op  = posterior_covariance(result; units = :original)

firm_worker = cov_block(prior_op; row_firms = [1, 2], col_workers = [10, 11])
principal   = cov_block(post_op;  firms = [1, 2],     workers = [10, 11])
```

`CovarianceOperator` owns the relevant sparse factorization. `CovarianceBlock`
stores the numeric matrix together with the row and column entity IDs and a
record of whether values are in standardized or original units.

## Repository Layout

```
src/
├── BipartiteGMRF.jl       # module entry point
├── types.jl
├── prepare.jl, util.jl
├── operators/             # Q-operators per prior model
├── linalg/                # PCG, SLQ
├── solvers/               # exact and Hutch/SLQ likelihood
├── decomposition/         # prior, posterior
├── covariance/            # operator, block extraction
└── api.jl                 # gmrf_mle, solve, accessors
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
    author  = {Wohak, Ulrich and Koren, Mikl{\'o}s and Orb{\'a}n, Krisztina
               and Telegdy, {\'A}lmos},
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

- [`VarianceComponentsHDFE.jl`](https://github.com/HighDimensionalEconLab/VarianceComponentsHDFE.jl)
  &mdash; fixed-effects AKM with leave-one-out (Kline&ndash;Saggio&ndash;S&oslash;lvsten)
  bias correction. Complementary point of comparison; see Section 4 of the
  paper.
- [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) &mdash; underlying
  optimizer.
- [`Graphs.jl`](https://github.com/JuliaGraphs/Graphs.jl) &mdash; bipartite
  graph utilities, used by the legacy preparation pipeline.

## Contributing

Issues and pull requests are welcome. For substantive changes, please open an
issue first to discuss scope. New prior models, solver back-ends, and
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
