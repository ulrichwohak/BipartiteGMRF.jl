# Bootstrapping `BipartiteGMRF.jl` with `PkgTemplates` — Refined Plan

A companion to `2026-05-15-gmrfmle.jl.md`. The design doc specifies *what* the
library looks like; this document specifies *how* to bring the skeleton into
existence with `PkgTemplates v0.7.61` (the version currently installed), and
how to handle version control and dependency management around that skeleton.

This file refines §11–§13 of the design doc and supersedes nothing.

> **Current-repo note.** This document is a clean-start reference for creating
> a sibling `BipartiteGMRF/` package with `PkgTemplates`. The current working
> tree has instead been adapted in place: `src/BipartiteGMRF.jl` is the package
> entry point, the root `Project.toml` is kept library-only, and the older
> project-specific pipeline scripts have been removed from the reusable package
> API and archived with the reproducibility compendium. For this repo, use the
> dependency, docs, test, and open-source-readiness guidance below as an audit
> checklist rather than rerunning `PkgTemplates`.

## 1. Repository and version control

### 1.1 New repo, separate from `BipartiteGMRFMLE`

The library lives in its own working tree and its own git history. Suggested
location: `/Users/uw/code/projects/BipartiteGMRF/`, a sibling of the current
`BipartiteGMRFMLE/`.

Reasons:

- `PkgTemplates`' `Git()` plugin initializes a git repository in the generated
  directory. Nesting that inside another repository would either create a
  submodule (rarely desired for a library) or contaminate the parent's
  history with the package skeleton.
- §12 of the design doc explicitly says "this library work should not
  initially change the existing project scripts." A peer directory keeps
  that boundary clean.
- Phase 12 in §13 (project integration) is the only point at which the parent
  references the library, and at that point it does so through
  `Pkg.develop(path="../BipartiteGMRF")`, not through git submodules.

### 1.2 First-commit hygiene

`Git()` writes commit 0 with a generic "Initial commit" message containing the
templated tree. Treat that as the baseline:

1. Run `t("BipartiteGMRF")`.
2. Inspect the tree and CI workflows — do not edit yet.
3. Push the templated commit to a fresh GitHub remote (private is fine for
   phases 2–7).
4. Verify CI is green against the empty package on Julia 1.10 LTS and current
   stable on Linux. If anything fails at this stage it is a templating bug,
   not a library bug, and is much easier to diagnose against an unchanged
   baseline.
5. Only then begin phase 2 of §13.

### 1.3 Branching strategy

- `main` is the always-green branch. CI must pass before merge.
- Feature branches per phase of §13: `feat/02-prepare`, `feat/03-operators`,
  `feat/04-linalg`, etc. Opening a PR even for solo work gives CI a chance to
  catch regressions before they reach `main` and produces a paper trail
  aligned with the phase boundaries.
- Avoid long-lived feature branches. If a phase stalls, merge the partial
  work behind an `export`-omission or a docstring `!!! warning` rather than
  letting the branch drift.

### 1.4 `.gitignore`

`PkgTemplates` writes a sensible default. Add the following if not already
present:

- `/Manifest.toml` — package-root manifest. Libraries do not commit manifests;
  applications and pipelines do.
- `/docs/Manifest.toml`, `/test/Manifest.toml` — same rule.
- `/docs/build/`
- `*.cov`, `lcov.info`, `/coverage/`
- `.DS_Store` — already a habitual offender in this working copy.

### 1.5 Tagging and registration cadence

- Stay on `v0.1.x` while the public API is moving. §11 of the design doc says
  "register no earlier than v0.2/v0.3."
- `TagBot()` only acts in response to a registry merge; until you register,
  it sits idle. Manual `git tag v0.1.x` is allowed and harmless.
- Bump to `v0.2.0` only when one external-style workflow (a notebook, a
  script in `BipartiteGMRFMLE/`, or a test fixture) has exercised the
  full `gmrf_mle → prior_decomposition → posterior_decomposition →
  cov_block` chain.

### 1.6 Relationship with `BipartiteGMRFMLE`

Phase 12 of §13 is when this repo starts depending on the library. The
transition:

1. In `BipartiteGMRFMLE/Project.toml`, add the library as a dependency.
2. `julia --project=. -e 'using Pkg; Pkg.develop(path="../BipartiteGMRF")'`
   so the parent picks up the local checkout, not a registered version.
3. Keep project-specific CLI wrappers, Parquet readers, and `estimates.txt`
   writers outside the reusable library package.
4. Once the library reaches v0.2 and is registered, switch
   `Pkg.develop(...)` to `Pkg.add("BipartiteGMRF")` in the parent.

## 2. Dependency management

### 2.1 Minimum Julia version

`julia = "1.10"` (LTS). The current scripts use no features beyond 1.10 and
SparseArrays/CHOLMOD behavior is well-tested there. Going lower buys nothing
and complicates compat-resolver outcomes for downstream users.

### 2.2 What to add, and when

**Required from phase 1** (immediately after `t("BipartiteGMRF")`):

- `DataFrames` — primary input to `gmrf_mle` and `GMRFProblem`.

**Required from phase 5 onward**:

- `Optim` — Nelder-Mead for `HutchSLQ`; L-BFGS for `ExactCholesky` warm-up.
- `FiniteDiff` — used for `ExactCholesky`'s finite-difference gradient path
  and for the post-convergence perturbation check in both solvers.

**Stdlibs** — imported, not `Pkg.add`-ed:

- `SparseArrays`, `LinearAlgebra`, `Statistics`, `Random`, `Printf`,
  `Logging`.

**Conditional**:

- `ADTypes` — only needed if the library retains an `autodiff = :forward`
  or `:reverse` codepath. Defer the decision; v0.1.x can ship with
  `ExactCholesky(autodiff = :finitediff)` as the only supported mode and
  not depend on `ADTypes` at all.
- `StableRNGs` — test-only. Put it in `test/Project.toml`, not the root
  `Project.toml`.

**Explicitly dropped** (do not migrate from the current scripts):

- `Parquet2`, `CSV`, `JSON` — moved to project glue (`BipartiteGMRFMLE`).
- `Plots`, `Makie`, `Kezdi`, `Graphs`.

### 2.3 Compat bounds

After adding each direct dependency, write an explicit `[compat]` entry.
For phase 1:

```toml
[compat]
julia      = "1.10"
DataFrames = "1"
Optim      = "1"
FiniteDiff = "2"
```

Bound to the major version. Libraries should err loose; tighten only when an
upstream actually breaks you. Tight bounds at v0.1.x lead to resolver pain
for downstream users with no benefit.

### 2.4 Manifest discipline

- `Manifest.toml` at the package root: untracked. `Git(manifest = false)` in
  the template invocation enforces this from commit 0.
- `test/Manifest.toml` and `docs/Manifest.toml`: untracked.
- CI regenerates manifests on each run — that is the intended behavior.

### 2.5 The AD caveat from design §14.6

`gmrfmle_exact.jl` currently tries three autodiff backends in sequence because
the Optim API changed between releases:

```
autodiff = :finite               # old Optim
   → autodiff = AutoFiniteDiff() # new Optim via ADTypes
   → autodiff = (Optim default)  # fallback
```

For the library, decouple from this:

- `ExactCholesky(autodiff = :finitediff)` should mean "use `FiniteDiff.jl`
  directly to build the gradient closure, then hand the closure to Optim
  without using its `autodiff` kwarg." This pins behavior to the library's
  own dep on `FiniteDiff` and removes the Optim-version branch.
- Do not advertise `:forward` or `:reverse` in v0.1.x. Sparse CHOLMOD
  factorization is not safely differentiable through the standard
  `ForwardDiff`/`Zygote` paths; offering the kwarg would mislead.

### 2.6 Threading

§9 and §14.9 of the design doc both prohibit `BLAS.set_num_threads(N)` as a
load-time side effect inside the library. Document the recommendation
(`BLAS.set_num_threads(1)` for the typical workload) in
`docs/src/performance.md`. Do not call it.

The current scripts do call it at module top-level. That is a script-level
choice; it does not belong in the library.

## 3. Bootstrapping with PkgTemplates

### 3.1 Recommended template invocation

Verified against `PkgTemplates v0.7.61` (the version currently in your depot):

```julia
using PkgTemplates

t = Template(;
    user    = "uw",                                      # GitHub user or org
    dir     = "/Users/uw/code/projects",                 # sibling of BipartiteGMRFMLE
    authors = ["Ulrich Wohak <ulrich@wohak.eu>"],
    julia   = v"1.10",
    plugins = [
        ProjectFile(; version = v"0.1.0"),
        SrcDir(),
        Readme(),
        License(; name = "MIT"),
        Git(; ssh = false, manifest = false),
        Tests(; project = true),
        GitHubActions(;
            extra_versions = ["1.10", "1"],
            linux   = true,
            osx     = false,
            windows = false,
            coverage = true,
        ),
        Codecov(),
        Documenter{GitHubActions}(),
        CompatHelper(),
        TagBot(),
        Citation(; readme = true),
    ],
)

t("BipartiteGMRF")
```

Decide each of the following before running:

1. **GitHub user/org.** Affects URLs embedded in workflows and badges.
2. **SSH vs HTTPS.** `Git(ssh = true)` if your GitHub credential flow is
   SSH-only. Affects the remote URL embedded by TagBot and Documenter.
3. **Codecov.** Drop the `Codecov()` plugin and `coverage = true` if you have
   no Codecov account. Easy to add later without re-templating.
4. **Package name.** §2 of the design doc recommends `BipartiteGMRF.jl`;
   §14.1 leaves it open. Verify the General registry has nothing under
   that name (search JuliaHub) before phase 12.

### 3.2 Generated layout

```
BipartiteGMRF/
├── .git/                       (Git)
├── .github/workflows/
│   ├── CI.yml                  (GitHubActions)
│   ├── CompatHelper.yml
│   ├── TagBot.yml
│   └── Documenter.yml          (Documenter{GitHubActions})
├── .gitignore                  (Git)
├── CITATION.bib                (Citation)
├── LICENSE                     (License — MIT)
├── Project.toml                (ProjectFile, with [compat] julia)
├── README.md                   (Readme — install + citing sections)
├── docs/
│   ├── Project.toml
│   ├── make.jl
│   └── src/index.md
├── src/
│   └── BipartiteGMRF.jl        (one-line stub: module BipartiteGMRF end)
└── test/
    ├── Project.toml            (Tests with project=true)
    └── runtests.jl
```

### 3.3 Gap analysis: skeleton vs design doc §8

| Design §8 element | Skeleton state | Action / phase |
|---|---|---|
| `src/BipartiteGMRF.jl` (top-level module) | Stub | Edit to `include` each submodule and `export` the public API |
| `src/types.jl` | Missing | Phase 2 |
| `src/prepare.jl` | Missing | Phase 2 |
| `src/util.jl` (`leading_singular_value`, `is_forest`, `safe_id`) | Missing | Phase 2 |
| `src/operators/{qop,qop_vs,mop}.jl` | Missing | Phase 3 |
| `src/linalg/{pcg,slq}.jl` | Missing | Phase 4 |
| `src/solvers/exact.jl` | Missing | Phase 5 |
| `src/solvers/hutch.jl` | Missing | Phase 6 |
| `src/api.jl` (`gmrf_mle`, `solve`, accessors, `Base.show`) | Missing | Phase 7 |
| `src/decomposition/prior.jl` | Missing | Phase 8 |
| `src/decomposition/posterior.jl` | Missing | Phase 9 |
| `src/covariance/{operator,extract}.jl` | Missing | Phase 10 |
| `test/runtests.jl` | Stub | Edit in phase 2 |
| `test/fixtures/synthetic.jl` (`synthetic_bipartite`) | Missing | Phase 2 — gates all later test files |
| `test/test_*.jl` | Missing | One per phase, 2 through 10 |
| `test/integration.jl` | Missing | Phase 10 |
| `docs/src/index.md` | Stub | Phase 11 |
| `docs/src/api.md`, `priors.md`, `solvers.md`, `weighting.md`, `examples.md` | Missing | Phase 11 |
| `CITATION.cff` | Missing (only `.bib` generated) | Phase 12, handwrite — see §3.4 below |

### 3.4 Citation files: `.bib` vs `.cff`

The `Citation()` plugin emits `CITATION.bib` (BibTeX) and adds a "Citing this
package" section to `README.md`. The design doc (§11) calls out
`CITATION.cff`, which is a different format — YAML, rendered by GitHub's
"Cite this repository" UI. No PkgTemplates plugin generates `.cff`.

Resolution: keep both. After phase 12, handwrite `CITATION.cff` next to the
auto-generated `CITATION.bib`. The two coexist cleanly: `.cff` for GitHub's
UI, `.bib` for paste-in academic use.

### 3.5 Workflow file sanity checks

Open `.github/workflows/CI.yml` after generation and verify:

- The `julia-version` matrix is `["1.10", "1"]` (LTS plus current stable).
- The `os` matrix contains only `ubuntu-latest`.
- A coverage step is present and uploads to Codecov (if you kept the
  `Codecov()` plugin).
- The `julia-actions/setup-julia` and `julia-actions/julia-runtest` actions
  are pinned to a version, not `@master`.

Add macOS / Windows to the matrix later, when SuiteSparse behavior has been
audited (per §11 of the design doc).

### 3.6 Optional: Aqua.jl

Aqua catches ambiguities, unbound type-vars, stale deps, and similar
lint-level issues. Not in the default plugin set, but trivial to retrofit:

- Add `Aqua = "0.8"` to `test/Project.toml`.
- In `test/runtests.jl`:

  ```julia
  using Aqua, BipartiteGMRF
  Aqua.test_all(BipartiteGMRF; ambiguities = false)
  ```

Disable `ambiguities` for libraries with overloaded `Base.show` — the test
tends to false-positive on display methods. Decide before phase 2;
retrofitting later is harmless but slightly noisier in PRs.

## 4. Phase plan, mapped to the templated skeleton

§13 of the design doc lists 12 implementation phases. With `PkgTemplates`,
phase 1 collapses to a single command. The remaining phases proceed against
the skeleton:

| Phase | Concrete work |
|---|---|
| **1. Skeleton** | Run `t("BipartiteGMRF")`. Push to GitHub. Verify CI green on the empty package. No code edits. |
| **2. Data prep** | `Pkg.add("DataFrames")`. Create `src/types.jl`, `src/prepare.jl`, `src/util.jl`. Create `test/fixtures/synthetic.jl` with `synthetic_bipartite(...)`. Create `test/test_prepare.jl`. Wire `include` statements in `src/BipartiteGMRF.jl`. Add `[compat] DataFrames = "1"`. |
| **3. Operators** | Create `src/operators/{qop,qop_vs,mop}.jl`. Create `test/test_operators.jl` covering all four prior models against dense reference. |
| **4. Linalg** | Create `src/linalg/{pcg,slq}.jl`. Create `test/test_pcg.jl`, `test/test_slq.jl`. Local RNGs only — no global `MersenneTwister` state. |
| **5. Exact solver** | `Pkg.add(["Optim", "FiniteDiff"])`. Create `src/solvers/exact.jl`. Create `test/test_exact.jl`. Add `[compat]` entries. |
| **6. Hutch solver** | Create `src/solvers/hutch.jl`, `test/test_hutch.jl`. |
| **7. Public API** | Create `src/api.jl` with `gmrf_mle`, `solve(::GMRFProblem, ::AbstractGMRFSolver)`, accessors, `Base.show(::GMRFResult)`. Export from top-level module. |
| **8. Prior decomp** | Create `src/decomposition/prior.jl`. Extend `test/test_decomposition.jl`. Wire `decompose = N` through `gmrf_mle`. |
| **9. Posterior decomp** | Create `src/decomposition/posterior.jl`. Make explicit `posterior_decomposition(result; probes, seed)`. |
| **10. Covariance** | Create `src/covariance/{operator,extract}.jl`. Create `test/test_covariance.jl`. Create `test/integration.jl` running the full chain. |
| **11. Docs** | Fill `docs/src/{api,priors,solvers,weighting,examples}.md`. Build locally with `julia --project=docs docs/make.jl`. |
| **12. OS-ready** | Handwrite `CITATION.cff`. Audit badges in `README.md`. `Pkg.develop` the parent project against the library and convert the three scripts to thin wrappers. Decide on registration. |

## 5. Risks specific to the bootstrap

A short list of things that have bitten other Julia libraries at this stage:

- **`Pkg.develop` cycles.** If the parent project is `Pkg.develop`-ed against
  the library while the library has the parent as a path-dep, Pkg resolves
  but actions like `Pkg.update` behave unpredictably. The library must have
  no dependency on `BipartiteGMRFMLE`, ever. Keep the dependency arrow
  one-directional.
- **Manifest committed by accident.** `Git(manifest = false)` prevents this
  on commit 0, but a stray `git add -A` later can sneak it in. Watch for
  `Manifest.toml` in `git status` before committing.
- **`autodiff` kwarg drift.** Optim has changed its `autodiff` API at least
  twice. The library should not pass through that kwarg; build the gradient
  closure itself via `FiniteDiff`. See §2.5.
- **Sparse Cholesky reproducibility.** SuiteSparse's CHOLMOD on macOS vs
  Linux has produced numerically different log-determinants in past
  releases. Until phase 12 the test matrix is Linux-only for this reason.
- **CI on an empty package.** Some plugins emit workflow steps that assume
  source files exist (e.g. doctests). If CI fails on commit 0 with a "no
  files to test" error, comment out the doctest step until phase 2 lands a
  first docstring, then re-enable it.

## 6. Open decisions to settle before running

1. **Final package name.** `BipartiteGMRF.jl` is the recommendation in §2 of
   the design doc and is used throughout this document.
2. **GitHub user / org** for the `user=` template arg.
3. **Visibility of the remote at start.** Private through phase 7 is the
   suggested default; public no later than phase 11 so README badges
   resolve.
4. **Codecov account.** Keep the plugin only if the account exists.
5. **SSH or HTTPS** for the embedded git remote URL.
6. **Aqua.jl** at phase 2, or skip.
