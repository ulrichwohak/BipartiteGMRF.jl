# Examples

## Two-Step API

```julia
using BipartiteGMRF, DataFrames

problem = GMRFProblem(
    df;
    outcome=:y,
    firm_id=:firm_id,
    worker_id=:worker_id,
    prior=NormalizedPrior(),
    weighting=Weighting(observations=:raw),
)

exact = solve(problem, ExactCholesky(); decompose=100, seed=42)
hutch = solve(problem, HutchSLQ(logdet_probes=20, lanczos_iters=20);
    decompose=false, seed=42)
```

## Effective Weighting

```julia
result = gmrf_mle(
    df;
    weighting=Weighting(observations=:effective, rho_eps=:estimate),
    solver=ExactCholesky(),
    decompose=100,
)
```
