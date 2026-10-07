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

## AR(1) HutchSLQ: accuracy and scaling

`benchmark/hutch_ar1.jl` measures the iterative AR(1) path without application
data or extra dependencies. Its connected tree has grouped co-managers,
graph-only leaves, successive observed ranks, raw weighting, and sparse
controls. Simulation uses tree recursions, not a network factorization.
Accuracy mode compares with both ExactCholesky and an independently assembled
observation-space Gaussian likelihood. Scaling mode never constructs either
reference and includes eta-changing statistics and final coefficient solves.

From the repository root, after instantiating the package environment:

```sh
# Fixed-parameter sensitivity, independent seeds, two starts and rho profiles.
OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=. benchmark/hutch_ar1.jl accuracy

# Larger probe and optimizer budgets on the same small fixture.
OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=. benchmark/hutch_ar1.jl accuracy fit-iters=400 probes=512 probe-counts=128,512

# Fresh-process peak RSS, warmed evaluations, and a bounded full fit on macOS.
/usr/bin/time -l env OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=. benchmark/hutch_ar1.jl scaling firms=1000 observations=3000 nodes=15000 controls=8 probes=16 steps=16 tol=1e-8 fit-iters=2
```

Use `/usr/bin/time -v` instead on Linux. Other `key=value` options include
`data-seed`, `seed`, `lanczos-steps`, `solve-tols`, `independent-seeds`, and
`cg-maxiter`. Output records the loaded source/revision, dimensions, hardware,
threads, seeds, solver settings, every measured likelihood difference and fit
candidate, and all timing repetitions. It distinguishes retained object size,
cumulative allocations, and external peak process RSS. The accuracy reference
is limited to 512 nodes/observations to avoid accidental dense large runs.

### Measured approximation quality

The following synthetic measurements used Julia 1.12.6, an Apple M3 Pro with
18 GiB RAM, and one Julia/BLAS thread, on 2026-10-07. The small fixture has
64 nodes, 32 grouped observations, eight firms and three controls; data seed
`20261006`, common probe seed `20261007`. The fixed-parameter grid visits
rho `(-0.3, 0, 0.3)` and eta `(-0.45, 0, 0.45, -0.45)`, including a repeated
eta after crossing zero in the same cache. Exact and dense NLLs agree within
`6.3e-14`. NLLs omit the shared Gaussian constant.

With 32 Lanczos steps and `cg_tol=1e-10`:

| Probes | Maximum absolute NLL error | RMS NLL error |
|--:|--:|--:|
| 8 | 1.48780 | 0.81059 |
| 32 | 0.65134 | 0.36606 |
| 128 | 0.34593 | 0.17277 |
| 512 | 0.06520 | 0.04079 |

At 512 probes, raising Lanczos steps from 6 to 16 to 32 gives maximum errors
`0.06525`, `0.06520`, `0.06520`. With 32 steps, tightening `cg_tol` from
`1e-4` to `1e-7` to `1e-10` gives `0.06478`, `0.06520`, `0.06520`.
Probe error dominates on this fixture; numerical errors can cancel, so a
slightly smaller total error with a looser solve is not evidence of a better
solve. The table is a measured seed-specific trend, not a monotonicity guarantee.

The 400-iteration fit comparison shows why optimizer convergence alone is
insufficient. At 128 probes, both starts report convergence but yield
`(rho, eta) = (-0.4509, 0.8120)` and `(0.1916, -0.0658)`. The first candidate's
exact NLL is `14.4583`, versus `13.7997` for the corresponding exact-fit
candidate: an objective gap of `0.6586`.

At 512 probes, the two starts agree closely within each probe seed, but not
across seeds:

| Probe seed | Rho from the two starts | Eta (approximately) | Reported convergence |
|--:|:--|--:|:--|
| 20261007 | -0.093913, -0.093896 | -0.0644 | Both true |
| 20261008 | +0.097961, +0.097978 | -0.0646 | Both hit 400 iterations |
| 20261009 | -0.032117, -0.032103 | -0.0643 | First hits 400; second true |

The respective NLL approximation errors at these candidates are about
`-0.0161`, `-0.0202`, and `-0.00223`. Firm and residual scales are similar
across these runs, while the worker scale approaches zero. This is a weakly
identified, near-boundary example, **not** parameter-recovery evidence.
Exact fixed-rho candidates at `(-0.3, 0, 0.3)` have NLLs
`(13.8938, 13.7966, 13.9022)`: a profile span of only `0.1057`, comparable to
approximation error. The common-seed Hutch profile prefers `+0.3` at 128
probes and `0` at 512 probes on this three-point grid.

Exact-fit candidates are finite optimizer runs, not certified global minima;
ExactCholesky's `1e-3` absolute simplex-spread floor is still active. Some
Hutch candidates therefore have a slightly lower exact NLL than their exact
reference candidate. A negative candidate gap is not a correctness failure.
For a flat application profile, use common probes for comparisons, then
independent seeds and larger budgets; require stability in likelihood
differences and parameters before interpreting the optimum.

### Measured process memory and evaluation time

Fresh, serial processes on the same machine used eight sparse controls,
16 probes, 16 Lanczos steps, `cg_tol=1e-8`, `cg_maxiter=2000`, and the seeds
above. Each process prepares the graph, warms the objective, records three
evaluation repetitions and three eta-cycle repetitions, and runs a full fit
capped at **two optimizer iterations**. The fits include estimated eta and
final beta reconstruction. All produce finite likelihoods, perform 13
objective evaluations, and report `converged=false`; they are execution and
memory checks, not converged estimates.

| Latent nodes | Grouped observations | Firms | Warm evaluation (minimum of 3) | Bounded fit | Peak RSS | Peak footprint |
|--:|--:|--:|--:|--:|--:|--:|
| 15,000 | 3,000 | 1,000 | 0.0549 s | 1.19 s | 1.012 GiB | 0.781 GiB |
| 75,000 | 15,000 | 5,000 | 0.2719 s | 4.17 s | 1.116 GiB | 0.878 GiB |
| 300,000 | 60,000 | 20,000 | 1.0865 s | 15.12 s | 1.417 GiB | 1.187 GiB |
| 1,679,537 | 329,642 | 50,000 | 7.0161 s | 94.81 s | 2.503 GiB | 2.976 GiB |

Change `firms`, `observations`, and `nodes` in the scaling command above to
reproduce each row. RSS is macOS `/usr/bin/time -l`'s **maximum resident set
size**, converted from bytes to GiB; the last column separately reports that
tool's "peak memory footprint" metric. Neither metric should be substituted
for the other. The process includes Julia, JIT compilation, simulation,
preparation, all evaluations, finalization and allocator high-water marks.
The script creates an evaluation cache before starting the separate fit;
these are combined-benchmark process peaks, not isolated fit-only memory.
The 300,000-node case retains approximately 75.0 MB of statistics and 97.4 MB
of cache objects (decimal units; shared references mean these are not simply
additive). These quantities are not substitutes for measured process RSS.

The largest case deliberately matches the application's node and observation
counts, but uses **synthetic** firms, matches and links. It retains 1,249,896
graph-only leaf rows. Its three warmed evaluations take 7.0445, 7.0161 and
7.1128 seconds; a three-eta cycle takes 21.0993 seconds at minimum. The whole
fresh process takes 247.75 seconds. The bounded optimizer remains at its
initial parameter values in this case, so it demonstrates execution of the
entire fitting path, not successful parameter estimation.

The tree fixture is connected and retains graph-only leaves, but has limited
fill-in and relatively benign iterative conditioning. These are **measured
synthetic results**, not a DE/AT measurement, a comparison with its reported
16.3G exact-solver footprint, or an extrapolation to a production optimum.
The low probe budget used for scaling is deliberately separate from the
accuracy study; it is not recommended as sufficient for a flat rho profile.

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

The shared streamed kernel also supports `HutchSLQ` and safely consumes solvers'
reusable output buffers. Final HutchSLQ coefficients now use convergence-checked
PCG with the fitted observation statistics, including AR(1), rather than a
network Cholesky factorization. Sparse-product fill-in, iterative conditioning,
per-control solves, and the dense coefficient system remain limits; this is
not a guarantee of production feasibility on arbitrary graphs. Multiple error
classes still require `ExactCholesky`. Optional covariance extraction still
factors network matrices, and AR(1) decomposition is explicitly unsupported.

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

The opt-in `benchmark/mean_profile_cg.jl` preserves the exploratory comparison:

```sh
OPENBLAS_NUM_THREADS=1 julia --project=. benchmark/mean_profile_cg.jl
```

On its single well-conditioned `n = 2000, p = 256` exact-solve fixture, CG used
38 iterations (40 total network solves including the outcome and achieved
correction) instead of 257 direct-profile solves. Relative coefficient error
was about `1.1e-12`; the NLL finite-difference discrepancy in eta was about
`1.4e-9`. This is promising evidence for a follow-up, not validation of a
production solver across conditioning, priors, or approximate inner solves.

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

The larger connected AR(1) case (`n = K = 10000`, `p = 512`) gave the following
representative results. Timings are warmed minima except the separately warmed
pilot; memory units here are decimal MB/GB.

| Measurement | Dense baseline | Sparse, streamed profile |
|:--|--:|--:|
| Retained statistics | 133.23 MB | 11.40 MB |
| Preparation | 424 ms | 5.0 ms |
| Complete objective | 170 ms | 78.6 ms |
| Final coefficient reconstruction | 170 ms | 78.0 ms |
| Five-iteration pilot (16 objective evaluations) | 3.74 s | 1.48 s |
| Fresh-process peak RSS | 1.64 GB | 1.17 GB |

Both pilots were deliberately iteration-limited and did not converge. Their NLLs
and the fixed-parameter NLLs matched; maximum coefficient differences were below
`2e-15`. The new profiling workspace occupied 2.85 MB. The final measurement used
the source change set through `c4a17ad`. Repeated fresh
processes varied in RSS and timing; the figures are evidence for this fixture,
not promised ratios for other graph structures. In particular, these cycle
graphs have limited factorization fill-in.

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

There is also a known cancellation-related limit: at extreme variance ratios,
forming the coefficient matrix subtracts nearly equal terms. The fixed relative
symmetry check can then reject a mathematically valid trial even when the small
coefficient matrix is reasonably conditioned. This can occur on small graphs;
it is not solely a production-scale concern. A verified reproducer and a
cancellation-aware numerical follow-up are tracked in
[issue #133](https://github.com/ulrichwohak/BipartiteGMRF.jl/issues/133).

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
