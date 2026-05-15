# Solvers

- `ExactCholesky(; optim_iters=200, polish=true, autodiff=:finitediff)`
- `HutchSLQ(; logdet_probes=30, lanczos_iters=30, cg_tol=1e-6,
  cg_maxiter=700, optim_iters=1000)`

`ExactCholesky()` uses sparse Cholesky factorizations for deterministic
likelihood evaluation and finite-difference polishing. `HutchSLQ()` uses PCG
and stochastic Lanczos quadrature for larger graphs.

Both solvers accept `seed` through `gmrf_mle` or `solve` to make stochastic
paths reproducible.
