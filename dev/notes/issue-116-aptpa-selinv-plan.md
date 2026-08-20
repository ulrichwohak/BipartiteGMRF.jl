# Issue #116 — `_aptpa` does O(n) dense solves per block row

**Status:** done; committed on `feat/issue-114-init-warm-start` (user's call —
ships with #114/#115, novelty signalled in the version number)

## The problem, restated

`_aptpa` (`src/solvers/emblocks.jl:152`) computes the per-firm posterior
correction `Aᵢ P⁻¹ Aᵢ'` with one **full** solve per block row:

```julia
X = Matrix{Float64}(undef, n, m)        # n = nf + nw, the whole graph
for a in 1:m
    X[:, a] = workspace_solve(ws, Vector(Vi[a, :]))
end
return Matrix(Vi * X)
```

Called once per firm per EM iteration (`emiwblocks.jl:221`), so one E-step pass
costs `K_tot` full solves — 2.1M of them on the reporter's data, each
allocating a dense length-2.34M vector. The reported symptom (8 h, zero
completed iterations, 148M allocations, one thread in `daxpy` and eleven idle in
GC) is consistent.

## What the issue misses

The Gram form is **deliberate**, and the comment at `emblocks.jl:147-151` says
why: it "avoids the cancellation of the 4-term selected-inverse sum when M is
ill-conditioned; each entry is an inner product of *solved* vectors, so the
result stays PSD to roundoff."

That property is load-bearing, not cosmetic. `Seps[i]` feeds
`svals[i] = _equicorr_trinv(Seps[i], r)/φ` (`emiwblocks.jl:227`), which is
`E_q[rᵢ'Ψᵢ⁻¹rᵢ] ≥ 0` by construction and becomes the Gamma rate
`brate[i] = (δ + svals[i])/2` (`:231`), which is then passed to `log` (`:233`)
and `digamma`. A loss of PSD does not degrade the answer — it produces `NaN`
and silently poisons the ELBO.

So a naive swap to `selinv_dot` / a termwise 4-block sum, as the issue proposes,
trades a performance bug for a numerical one. The fix has to be fast **and**
keep the PSD guarantee.

## Proposed fix: one selected inversion per E-step, then a *local* Gram form

Four facts make this work; each is checked below and must be verified before the
rewrite is trusted.

- **F1 — the block's node set is inside `P`'s pattern.** For firm `i` let
  `Sᵢ = {firm_i} ∪ {workers of firm i}`. `assemble_block_precision`
  (`emblocks.jl:79-93`) pushes `(firm, firm)`, every `(worker_a, worker_c)` pair
  in the block (`:79-84`, a dense loop over `a, c`), and both `(firm, worker)`
  directions (`:90-91`). So `Sᵢ × Sᵢ ⊆ pattern(P)`, densely.
- **F2 — `P`'s pattern is inside the factor's.** `ws_M` factors `Q + P`, whose
  pattern is a superset of `P`'s, and Cholesky fill-in only adds. So
  `selinv_extract_at` returns *exact* values on `Sᵢ × Sᵢ`, never the
  "zero where outside the factor's pattern" case its docstring warns about.
  This is the one claim that, if wrong, silently biases `S_{ε,i}` downward.
- **F3 — `Σ_SS` is PD.** It is a principal submatrix of `P⁻¹`, and principal
  submatrices of a PD matrix are PD.
- **F4 — the Gram form survives.** With `Σ_SS = LLᵀ` from a small dense
  Cholesky, `Aᵢ[:,S] Σ_SS Aᵢ[:,S]ᵀ = (Aᵢ[:,S] L)(Aᵢ[:,S] L)ᵀ` — still a Gram
  product of an explicitly formed matrix, so still PSD to roundoff. The
  property the current code buys with `m` global solves is recovered from one
  `s × s` Cholesky, `s = |Sᵢ| ≤ m + 1`.

Cost per E-step goes from `K_tot` global solves to **one** `selinv_extract_at`
plus, per firm, an `s × s` Cholesky and an `m × s` product — `O(m³)` on the
block's own size, with no dependence on `n`.

## Worksteps

- [x] **1. Verify F1/F2 empirically, not just by reading.** On an existing
      fixture, compare `selinv_extract_at(ws_M, pattern)` against a dense
      `inv(Matrix(M))` at every `Sᵢ × Sᵢ` position. If any position comes back
      structurally zero, F2 is false and the whole approach collapses — so this
      gate comes first, before any rewrite.
- [x] **2. Precompute what is constant.** The selinv pattern (structural ones on
      `∪ᵢ Sᵢ × Sᵢ`) and each firm's `cols = Sᵢ` index vector do not change
      across EM iterations or across `θ`. Build them once in
      `make_em_blocks_workspace` and store on `EMBlocksWorkspace`. Build the
      pattern structurally rather than reusing `P`, so a numerically-zero entry
      can never drop out of it.
- [x] **3. Rewrite.** `_aptpa_blocks(Σ, cols, Vi)` doing the F4 congruence, with
      a documented fallback to `Asub * Symmetric(Σ_SS) * Asub'` (symmetrized) if
      the small Cholesky fails. Keep the existing `_aptpa` as the **reference
      implementation**, used by tests only, and say so in its comment.
- [x] **4. Numerical agreement.** Assert the new path matches `_aptpa` to tight
      tolerance on every fixture in `test_error_blocks_iw.jl`, and that
      `Seps[i]` stays PSD (min eigenvalue ≥ −tol) on a deliberately
      ill-conditioned fixture.
- [x] **5. Scaling regression.** The issue asks for a 100k-firm fixture; that is
      too slow for CI. Instead assert that the E-step's cost does *not* scale
      with `n`: two graphs with identical block structure but different node
      counts must take comparable time/allocations. That is the actual invariant
      the bug violated, and it is cheap to test.
- [x] **6. CHANGELOG** under v0.5.0; note in the `EMIWBlocks` docstring that
      block count is no longer the bottleneck.

## Explicitly not done

- No public API change (agrees with the issue).
- The acceptance criterion — one EM iteration in under a minute at `n ≈ 2.3M`,
  `B ≈ 1M` — cannot be verified in this repo's test suite. Worksteps 4 and 5
  establish correctness and the removal of the `n`-dependence; confirming the
  wall-clock target needs the reporter's dataset.

## Results

**Gate (workstep 1) passed.** On three fixtures (cyclic multi-size blocks, a
forest, a dense random graph with blocks up to size 12): zero structurally
missing positions and max `|selinv − inv|` of 5.6e-17. F2 holds — the reads are
exact, not silently zeroed.

**Agreement (workstep 4).** `_aptpa_local` matches the reference `_aptpa` to
1.8e-15 absolute in the worst case, including a deliberately ill-conditioned
fixture (`Ω` scaled by 1e-9, `ρ = 0.95`, large `σ`). PSD holds throughout:
minimum eigenvalue ≥ −2.2e-17, and +3.5e-10 on the ill-conditioned case.

**Scaling (workstep 5).** One E-step correction pass, old path vs new:

| n | blocks | old | new | speedup |
|---|---|---|---|---|
| 1,000 | 400 | 69 ms / 93 MiB | 0.5 ms / 0.8 MiB | ×133 |
| 5,000 | 2,000 | 3.25 s / 2.4 GiB | 6.4 ms / 6.3 MiB | ×510 |
| 20,000 | 8,000 | 61 s / 25 GiB | 32 ms / 23 MiB | ×1883 |

The old path grows roughly as `n × blocks`; the new one is linear in block
count alone. That is the invariant the bug violated, and the CI test asserts it
directly (allocation per block must not grow when the graph does) rather than
asserting a wall-clock number.

Extrapolating linearly in block count from the 8,000-block row, the reporter's
`B ≈ 1.04M` should land in the seconds per E-step pass. That is consistent with
the issue's acceptance criterion but is **not** a substitute for running it —
confirming the target needs their dataset.
