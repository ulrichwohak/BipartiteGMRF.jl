# Solvers

- `ExactCholesky(; optim_iters=200, polish=true, autodiff=:finitediff, g_reltol=1e-7)`
- `HutchSLQ(; logdet_probes=30, lanczos_iters=30, cg_tol=1e-6,
  cg_maxiter=700, optim_iters=1000, g_reltol=1e-7)`

`ExactCholesky()` uses sparse Cholesky factorizations for deterministic
likelihood evaluation: a gradient-free Nelder-Mead search, followed (with
`polish=true`) by a finite-difference L-BFGS polish. `HutchSLQ()` uses PCG
and stochastic Lanczos quadrature for larger graphs.

`ExactCholesky()` reuses the sparse matrix layout and symbolic Cholesky
analysis across parameter evaluations. The layout includes every position
that can be nonzero under the model, even when its value is zero at the
initial parameters or cancels at a later evaluation. This is important for
grouped observations with AR(1) errors: changing `eta` can make a previously
zero observation-information entry nonzero. Numerical updates preserve these
positions in both the prior and posterior precisions; an entry outside the
fixed layout raises an error instead of being discarded. Non-positive-definite
trial precisions are still treated as infeasible parameter values.

For `BipartiteVarianceStableModel`, `HutchSLQ()` evaluates the log-determinant
ratio as `logdet(B + lambda*S*VtV*S) - logdet(B)`, using identical random
probes for both terms. This avoids cancellation between sigma-dependent log
determinants when one latent standard deviation is small.

Both solvers accept `seed` through `fit_mle` or `solve` to make stochastic
paths reproducible.

## AR(1) residuals without network factorization

`HutchSLQ` supports fixed `error_eta=e` and joint `error_eta=:estimate`,
with or without `X`, and with grouped `match_id`, under raw observation
weighting. This applies to normalized, unnormalized, spectral, and
variance-stable models; the existing model restrictions still apply.
Multiple error classes still require `ExactCholesky`; inverse-Wishart error
blocks still require `EMIWBlocks`. AR(1) cannot be combined with effective
weighting, `error_cov`, `error_groups`, or `error_blocks`.

Changing eta recombines all weighted design, outcome and control products.
The sparse structural union retains adjacent-interval worker–worker entries,
even when their values pass through zero. Operators and Jacobi preconditioners
read current matrix values rather than using object identity as a change flag.
The exact residual log determinant is added to twice the negative log
likelihood: `(K - observed_firm_blocks) * log(1 - eta^2)`.

Preparation, initialization, optimization and final coefficient reconstruction
on this path use no network Cholesky factorization. Each network solve uses
PCG and verifies `norm(b - M*x)/norm(b) <= cg_tol` (zero RHS is exact).
Failed solves reject objective trials; an invalid final trial or failed final
coefficient solve raises an error. There is no direct-solver fallback.
The dense `p × p` coefficient system still uses Cholesky.

HutchSLQ also rejects numerically unresolved AR trials with
`1 - eta^2 <= sqrt(eps(Float64))`: the existing sufficient-statistic basis
subtracts terms of order `1/(1-eta^2)`, which can corrupt even a singleton's
eta-independent contribution this close to the boundary. This is an explicit
numerical failure guard, not clipping eta to a different value. It is not a
general accuracy guarantee near the boundary or at extreme variance ratios.

Nonpositive or nonfinite Lanczos Ritz values reject a trial; they are not
floored into positive values, and no implicit diagonal jitter is added.
Finite-probe SLQ is not a proof of positive definiteness over an arbitrary
explicit VS rho domain. Check `feasibility(model)` and use a valid domain;
`rho_limit=:auto` is an explicit caller choice. Fitting never prunes graph
links or silently changes an explicitly supplied rho limit.

Parameter, coefficient, `nll`, `loglikelihood`, AIC and BIC accessors do not
factor network matrices. `covariance(result; kind=:model/:fitted)` is optional
and **does** factor the network precision, even after HutchSLQ fitting.
`decompose` currently rejects AR(1) results because grouped target weights and
correlated residual contributions are not yet implemented correctly there.

Hold `seed` fixed across evaluations, starts and rho-profile comparisons to
reuse probes. Increase `logdet_probes`, `lanczos_iters` and solve accuracy,
then repeat with independent seeds to assess stochastic sensitivity. Report
likelihood differences and parameter/profile stability, not only `converged`.
A flat rho profile can move substantially under small approximation errors;
see the measured examples in [Performance](@ref).

`HutchSLQ`'s `g_reltol` sets the Nelder-Mead convergence tolerance *relative* to
the objective magnitude at the start point: the optimizer stops once the
simplex-objective spread falls below `g_reltol * max(1, |nll₀|)`. A relative
tolerance adjusts the stopping threshold to the initial objective scale; it
does not control SLQ error. Common probes make repeated evaluations
deterministic, so the optimizer can converge to a biased approximate optimum.
`ExactCholesky` uses the same `g_reltol` but
with a `1e-3` absolute floor — `g_tol = max(1e-3, g_reltol * max(1, |nll₀|))` —
so its small/medium-graph behaviour is unchanged while large exact problems
still converge promptly.
