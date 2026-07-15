# Solvers

- `ExactCholesky(; optim_iters=200, polish=true, autodiff=:finitediff, g_reltol=1e-7)`
- `HutchSLQ(; logdet_probes=30, lanczos_iters=30, cg_tol=1e-6,
  cg_maxiter=700, optim_iters=1000, g_reltol=1e-7)`

`ExactCholesky()` uses sparse Cholesky factorizations for deterministic
likelihood evaluation and finite-difference polishing. `HutchSLQ()` uses PCG
and stochastic Lanczos quadrature for larger graphs.

For `VarianceStablePrior`, `HutchSLQ()` evaluates the log-determinant ratio as
`logdet(B + lambda*S*VtV*S) - logdet(B)`, using identical random probes for both
terms. This avoids cancellation between sigma-dependent log determinants when
one latent standard deviation is small.

Both solvers accept `seed` through `gmrf_mle` or `solve` to make stochastic
paths reproducible.

`HutchSLQ`'s `g_reltol` sets the Nelder-Mead convergence tolerance *relative* to
the objective magnitude at the start point: the optimizer stops once the
simplex-objective spread falls below `g_reltol * max(1, |nll₀|)`. A relative
tolerance is scale-invariant, so the same setting converges on both small and
very large graphs. The stochastic SLQ/PCG objective is accurate only to a
relative level, so an absolute tolerance tight enough for a small graph is
unreachable on a large one and would force the optimizer to exhaust
`optim_iters` without converging. `ExactCholesky` uses the same `g_reltol` but
with a `1e-3` absolute floor — `g_tol = max(1e-3, g_reltol * max(1, |nll₀|))` —
so its small/medium-graph behaviour is unchanged while large exact problems
still converge promptly.
