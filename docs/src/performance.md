# Performance

The package does not set global BLAS thread counts at load time. For many sparse
Cholesky and iterative-solver workloads, using one BLAS thread can improve
repeatability and avoid oversubscription:

```julia
using LinearAlgebra
BLAS.set_num_threads(1)
```

Make that choice in applications or scripts, not in package code.

For larger graphs, prefer `HutchSLQ()` and tune `logdet_probes`,
`lanczos_iters`, `cg_tol`, and `cg_maxiter` against the desired runtime and
stochastic tolerance.

## Mean controls: memory and runtime

Supplying `X` estimates a linear mean jointly with the covariance parameters.
At each covariance-parameter trial, the package solves the covariance-weighted
coefficient problem exactly on the `ExactCholesky` route. This is profiled maximum
likelihood, not ordinary demeaning or REML: there is no extra coefficient-system
log-determinant term.

For sparse `X`, observation filtering and mean-statistic construction preserve
sparse storage for the large node-by-control products where mathematically
possible. The same treatment applies to the AR(1) and error-class combinations
formed during likelihood evaluation. A mathematically dense sparse product can
still fill in; sparse input is not a guarantee that every later matrix is sparse.
Dense input continues to use the dense preparation route.

Let `n` denote latent nodes and `p` mean coefficients. The profiling workspace
retains at most eight dense solved columns, rather than an `n × p` dense solution
matrix. Its storage is `O(n*b + p*b + p^2)`, with `b = min(8, p)`, in addition to
the retained mean products, network solver, and factorization. The coefficient
matrix remains dense: it alone requires approximately `8*p^2` bytes, and its
Cholesky solve costs `O(p^3)`. Factorization fill-in is a separate potential
memory limit.

The implementation uses the dependency's public **vector** solve interface.
A block of eight columns still requires eight vector solves, not one batched
factor solve. A mean-bearing objective evaluation uses `p + 1` network solves:
one outcome solve, reused for the profile, and one per control column. Sparse
storage and streaming do not remove that work. ExactCholesky final coefficient
reconstruction reuses the final likelihood factorization.

The shared streamed kernel also supports `HutchSLQ` on its existing supported
model/error combinations and safely consumes solvers' reusable output buffers.
However, **final coefficient reconstruction after a HutchSLQ fit still uses a
direct Cholesky factorization**. This implementation therefore does not provide an
end-to-end matrix-free mean fit on arbitrarily large graphs. AR(1) and multiple
error classes still require `ExactCholesky`.

### Reproducible synthetic benchmark and runtime gate

`benchmark/mean_profiling.jl` is opt-in and uses only deterministic synthetic data.
It accepts firm count, coefficient count, connected-component count, error kind,
and input storage. For example, from the repository root:

```sh
# Connected graph; two nonzero controls per observation; estimated AR(1).
OPENBLAS_NUM_THREADS=1 julia --project=. benchmark/mean_profiling.jl 1000 128 1 ar1 sparse

# Same dimensions, twenty disconnected components.
OPENBLAS_NUM_THREADS=1 julia --project=. benchmark/mean_profiling.jl 1000 128 20 ar1 sparse

# Small dense design, and a node-count comparison at fixed p.
OPENBLAS_NUM_THREADS=1 julia --project=. benchmark/mean_profiling.jl 1000 3 1 iid dense
OPENBLAS_NUM_THREADS=1 julia --project=. benchmark/mean_profiling.jl 2000 128 1 iid sparse
```

Instantiate the package environment first. For a historical comparison, run the
same script with a temporary Julia environment developing that historical
checkout; the script prints the source path and Git revision actually loaded.
No application data or additional benchmark dependency is needed.

Each operation is warmed before timing. The output separates preparation,
observation-statistic combination, one network solve, numeric factorization,
one block product, the dense coefficient solve, complete profiling, complete
objective evaluation, final reconstruction, and a short five-iteration fit. The
short fit is a runtime pilot, not a convergence or statistical-recovery claim.
It also records dimensions, structural and numerical sparsity, retained sizes,
Julia/package versions, CPU/RAM, seed, block size, and Julia/BLAS threads.

Three different memory quantities must not be conflated:

- `summarysize` estimates retained Julia objects; it excludes transient arrays
  and does not reliably account for native factorization memory.
- Reported allocations are cumulative bytes allocated during an operation,
  **not** simultaneously live memory.
- Fresh-process peak resident memory includes Julia, compilation, native
  libraries, factors, and allocator high-water marks. On macOS, prefix the
  command with `/usr/bin/time -l`; on Linux, use `/usr/bin/time -v`.

The script rejects cases exceeding a conservative 512 MiB estimated dense-array
budget, including when comparing the old implementation. This is not a process
RSS guarantee or permission to extrapolate to production scale. Begin with small
cases, vary `p` at fixed graph and sparse row support, then vary `n` at fixed
`p`. Measure both connected and disconnected graphs.

Before attempting a large graph with thousands of controls, set an explicit RAM
and wall-clock budget. Use a representative pilot's actual `obj_evals` and
warmed complete-objective time, then add preparation and final reconstruction.
The benchmark prints this core-time projection alongside measured pilot time;
it excludes optimizer and cache-initialization overhead. Finite-difference
evaluations are already counted in `obj_evals`, so do not multiply them in again.
Reassess the budget as dimensions, conditioning, or graph fill-in change.

Coefficient-space conjugate gradients may eventually reduce the number of
network solves when it converges in far fewer than `p` iterations. It is not
enabled here: a potential speedup must first be checked against direct profiling
for coefficient, objective, and finite-difference accuracy. No control columns
are dropped or regularized to make the solve cheaper.

### Scope of the measured improvement

Synthetic comparisons on Julia 1.12.6, GaussianMarkovRandomFields 0.12.4, an
Apple M3 Pro with 18 GiB RAM, and one Julia/BLAS thread used the script above
with seed `20260929`. The baseline was v0.5.3 plus the isolated borrowed-buffer
correction (`85b8490`), which does not change the exact route. Both connected
and twenty-component cycle graphs were checked, with `n = K = 2000`,
`p = 32, 128, 256`; additional cases used `n = 4000, p = 128`, small dense
controls, and a connected AR(1) graph with `n = 10000, p = 512`. These are
synthetic memory/runtime checks, not production-scale or recovery evidence.

For the connected AR(1) cases, retained statistics changed from 2.38 to 1.06 MB
at `p = 32`, and from 14.69 to 2.61 MB at `p = 256` (`n = 2000`, decimal MB).
Across the small comparison grid, fixed-parameter NLL differences were at most
`2.3e-13`. The package's independent dense-GLS tests, rather than agreement with
the historical implementation alone, establish numerical correctness.

Sparse storage is not always smaller. In the `p = 3` fixture, the node-by-control
product was fully populated: representing it sparsely increased retained
statistics from 0.500 to 0.548 MB. The corresponding genuinely dense input
retained the same 0.500 MB on both revisions. Vector solve callbacks also still
allocate, so cumulative allocation can remain substantial despite bounded live
solution storage. Neither these timings nor a sparse input guarantee a speedup
for every graph or design.

### Interpretation and persistence

The coefficient system is checked for finite values, symmetry, positive
definiteness, and severe numerical ill-conditioning before its solution is
accepted. A diagnostic can therefore expose a previously hidden collinearity,
poor column scaling, or inaccurate iterative solve. The package does not repair
such a problem with a ridge penalty, pseudoinverse, or dropped columns; inspect
the design and scaling instead.

Coefficient order and the original-outcome-unit convention are unchanged. With
`Xobs` aligned to the package's observation mapping, the fitted mean is
`result.stats.y_mean .+ Xobs * result.beta`; do not multiply `beta` by `y_std`
again. Grouped matches continue to select the first member's control row rather
than averaging member rows. Supplying constant-within-match controls makes that
choice immaterial; this storage change does not impose new constancy validation.

Julia's `Serialization` is not a stable cross-version interchange format for
package structs. New-version result round trips are tested. A dense mean-statistic
and fitted-result pair written by the pre-change v0.5.3 code was also loaded on
Julia 1.12.6 and refitted successfully: its NLL was unchanged and the largest
coefficient difference was `5.6e-17`. That limited check is not a guarantee for
every historical artifact or Julia version. Keep existing files and the
environment that created them. If an old artifact cannot be loaded, recreate
statistics and refit from the original inputs; do not overwrite the only copy.

## Warm starts

The default starting point is a fixed heuristic — `sigma_a = 0.7`,
`sigma_z = 0.04`, `sigma_epsilon = 0.4` on a unit-variance outcome — and is not
adapted to the data. `init` replaces it, field by field, in the outcome's own
units:

```julia
r  = fit_mle(BipartiteVarianceStableModel, f, w, y)
r2 = fit_mle(BipartiteVarianceStableModel, f, w, y;
             error_groups = firm, init = params(r))
```

Two uses. The first is cost: several error models fitted on the same graph can
share one cheap pilot estimate instead of each climbing from the same
dataset-agnostic default. The second is diagnostic: when a fit lands on a
boundary — `rho` pinned at `rho_limit`, a variance collapsed to zero — refitting
from a different, informed starting point is what separates a genuine feature of
the likelihood from an artifact of where the optimizer began.

A warm start on its own is not guaranteed to be faster, and the reasons are
worth knowing before reading too much into a timing.

### Size the simplex to match the start

Nelder-Mead does not begin at a point but at a simplex around it: vertex `j+1`
differs from the start in coordinate `j` only, and equals
`(1 + simplex_scale)·x_j + simplex_shift`. At the default `simplex_scale = 0.5`
that is a 50% relative perturbation — enormous on the unconstrained scale, where
`log σ = −0.9` reaches `−1.325`, i.e. `σ` down 35%. So a warm-started fit with
default settings still searches a wide region, and most of the benefit of `init`
is thrown away. Shrink the simplex to keep the search local:

```julia
# search ±5% around a good starting point
fit_mle(BipartiteVarianceStableModel, f, w, y;
        solver = ExactCholesky(simplex_scale = 0.05),
        init   = params(pilot))

# or a uniform absolute box, independent of coordinate magnitude
ExactCholesky(simplex_scale = 0.0, simplex_shift = 0.05)
```

`simplex_shift` is the absolute term, and it is what rescues coordinates near
zero: at exactly zero the relative term vanishes and only the shift remains. A
warm start produces exactly those coordinates — `log omega = 0`,
`atanh(eta) ≈ 0`, `log sigma ≈ 0` at `sigma ≈ 1`, `atanh(rho/rho_limit) ≈ 0` at
`rho ≈ 0` — so a simplex that is degenerate in one of them can satisfy the
convergence test having barely moved. Setting both knobs to zero is rejected for
that reason. `ExactCholesky` recovers through its L-BFGS polish; `HutchSLQ` has
no polish, so check that a warm-started `HutchSLQ` fit actually moved.

### The stopping tolerance also depends on the start

`g_reltol` is scaled by `max(1, |NLL(x₀)|)`, so a better starting point yields a
tighter absolute threshold. `ExactCholesky` floors it at `1e-3`, so it usually
does not move there; under `HutchSLQ` the scaling passes through and a better
start can cost extra iterations rather than fewer. (For a negative NLL the
direction reverses.) This is deliberate — pinning it would silently change when
existing fits stop.
