# Issue #120 — match-grouped designs under `error_eta` and `error_blocks=:iw`

**Status:** approved 2026-08-21; executing. Steps 1–2 implemented and green
(prepare 140/140, error_ar1 29/29, error_blocks_iw 148/148). Advisor
re-reviewed the step-1 diff: approved; its amendments (union-pattern
docstring fix incl. a pre-existing dangling sentence, rank-vs-appearance
test fixture, extra component assertions, error-message paren) are applied
— the paren fix came via the shared `per_observation_value` helper both
builders now use.
**Branch:** `feat/issue-120-match-error-units`, cut from `3d01414` (tip of
`feat/issue-114-init-warm-start` = `release/v0.5.0`).

## Goal

Allow `match_id` together with `error_eta` (AR(1)) and `error_blocks=:iw`.
Collapse to match-level observations first (`1/F_s` firm loading, `1/M_s`
worker loading, one observation per match — the existing
`observation_rows`/`build_match_V_stats` semantics), then define the error
process **directly on the match units**:

- AR(1): `Corr(ε_s, ε_t) = η^|rank_s − rank_t|` over a firm's matches ranked
  by `edge_index`. One error draw per spell; no `G R G'`.
- IW: the firm block is its **matches** (`m_i` = matches at firm i),
  `Ψ_{m_i} = φ[(1−r)I + r·11']` on match errors, integrated out exactly as now.

Every estimator then shares one observation equation
`y_s = (1/F_s) Σ a_i + (1/M_s) Σ z_m + ε_s`, differing only in `Cov(ε)`.

## Key code facts the plan builds on

- `observation_rows` (`src/prepare.jl:396`) already produces match-collapsed
  `V, yv, src` when given `match_ids` (used by `error_cov` and
  `error_groups`); `src` maps each observation to its first input row —
  the established convention for reading per-row side data (`error_cov`
  reads `R` at each match's first row).
- `build_ar1_V_stats` (`prepare.jl:710`) is generic sparse algebra after
  `observation_rows`; nothing in `V'V / V'S_adj V / V'S_int V` or the
  structural union-pattern construction assumes one worker per row. The only
  raw-row assumptions are (a) it passes `match_ids = nothing`, (b) it groups
  rows by `f_obs[s]` per raw row, (c) `edge_index_obs` is per raw row.
- The IW path has real one-firm-one-worker-per-row assumptions, all in
  `src/solvers/emblocks.jl`: `_edge_nodes` / `_edge_values` (exactly two
  nonzeros per V row), `assemble_block_precision` (unit loadings, one worker
  per row), `_block_selinv_pattern` (`(m+1)²` sizehint), `_aptpa_local`
  (one firm + one worker per row when building the block-local `A`).
  `_aptpa` (reference implementation, tests only) is already generic.
- The two blockers to remove: `src/stats.jl:139` ("error_eta does not support
  match_id") and `src/stats.jl:177` ("error_blocks=:iw does not yet support
  match_id").

## Contracts (settle before code)

- **Single-firm matches.** Under both error paths a match spanning more than
  one distinct firm is a hard `ArgumentError` (the firm's spell timeline is
  the error-process index; the interval construction guarantees one firm per
  match on real data). Plain `match_id` (iid errors) keeps allowing
  multi-firm matches — no behavior change there.
- **`edge_index`**: stays per input row (same length as `y`, uniform with
  every other argument). New validation with `match_id`: all rows of a match
  must carry the same value (read at `src` after the check); per firm, the
  per-match values must be a strict permutation of `1:m_i` — ties are
  structurally impossible, so validation stays strict.
- **`firm_group`**: stays per input row and keeps the existing
  `firm_group == f_obs` check; combined with single-firm matches this makes
  the per-match firm well-defined.
- **`X`** composes: read at each match's first row via `src`, matching the
  `error_cov` convention, and documented as such.

## Worksteps

- [x] **1. AR(1) builder.** `build_ar1_V_stats` gains a
      `match_ids::Union{Nothing,Vector{Int}}` argument: pass it through to
      `observation_rows`; derive the per-observation firm (validate
      single-firm matches), per-observation `edge_index` (validate
      within-match agreement). **Advisor-flagged, implementation-critical:**
      `K` must become `length(yv)` (matches, not raw rows) — it is
      load-bearing three times: `stats.K` enters the NLL as
      `K·2·log(σ_ε)` (exact.jl), it drives
      `logdet_R = (K − n_blocks)·log(1−η²)` (`ar1_observation_stats`), and
      it fills `WeightStats`. The firm-grouping loop (`f_obs[s]`,
      prepare.jl:730) must index the derived per-observation firm, not raw
      rows. The rest is generic sparse algebra and stays.
      Rewrite the docstring: delete the "match_id is deliberately NOT
      supported" paragraph, state the match-level AR(1) definition; fix the
      "for a firm with $m rows" validation message (rows → matches).
      Unit tests at the builder level (test_prepare.jl style).
- [x] **2. IW builder.** `build_block_V_stats` gains `match_ids`: collapse via
      `observation_rows`, keep the `firm_group == f_obs` row-level check, add
      the single-firm-per-match assertion, emit `block_of`/`sizes` over
      matches (sizes = matches per firm). Done via the shared
      `per_observation_value` helper (prepare.jl, next to
      `observation_rows`).
- [x] **3. EM internals.** Done: `_edge_nodes`/`_edge_values` replaced by
      `_row_supports` (per-row node/value lists + firm), generic
      `assemble_block_precision` accumulation, node-set union
      `_block_selinv_pattern`, `_aptpa_local` over row supports; tests
      assert dense `A'Ω⁻¹A` equality and `_aptpa` agreement on a fixture
      with a shared worker and co-managed matches (iw 188/188).
      **Sequencing constraint (advisor):** this step
      MUST land before step 4 removes the `suffstats` guard —
      `build_block_V_stats` can now produce multi-worker rows, and until
      emblocks.jl is generalized, `_edge_nodes` does last-worker-wins
      overwriting (silently wrong, no error); only the stats.jl
      ArgumentError blocks the path today. Also add a fixture with a worker
      shared across two matches of the same firm (stresses the `A[a,pos] +=`
      accumulation and the block node-set union).
      Generalize `emblocks.jl` to weighted multi-worker
      observation rows: per-row node/value lists read from `V` (replacing the
      scalar `edge_firm/edge_worker/edge_vf/edge_vw` quartet on
      `EMBlocksWorkspace`), `assemble_block_precision` accumulating
      `A_i'Ω_i⁻¹A_i` from the actual row supports, `_block_selinv_pattern`
      node sets `{firm} ∪ {workers of the block's matches}` with the sizehint
      from the actual node-set size, `_aptpa_local`'s block-local `A` built
      from the per-row lists. The dense-verified reference `_aptpa` is the
      agreement oracle for all of this (it solves per design row with no
      support assumption). Fix the now-stale "match_id is rejected"
      comments in **both** `_edge_values` (emblocks.jl:162) and
      `_aptpa_local` (emblocks.jl:229), plus the `FirmBlockStats.V` field
      comment in types.jl ("edge-level design rows, NOT collapsed"). Note:
      `make_em_blocks_workspace` seeds `ws_M`'s symbolic pattern from
      `P0 = assemble_block_precision(fb, _init_blocks(fb), ...)`, so the
      pattern generalizes for free once assembly is generalized — the
      equicorr reference blocks produce no exact-zero entries in the block
      clique for positive `1/M` weights, and `_block_selinv_pattern`
      builds from explicit ones independently of `P0`. Keep reading actual
      row supports even where the single-firm constraint makes the firm
      loading exactly 1 — do not "simplify" back to unit loadings.
- [ ] **4. `suffstats` wiring.** Drop the two `ArgumentError`s; on both paths
      collapse when `match_id` is present (pass `match_id_obs` down), run the
      new validations, `K` = number of matches. On the IW branch the
      placeholder edge-level design becomes `build_match_V_stats` when
      grouped, so non-EM consumers of `stats.design` see the match design.
      AR(1)+`X` reads `X` at `ar1_aux.src` (already does — now that means
      first row of match). Update both docstring sections. Pre-existing
      asymmetry to note (not fix): `optimize_emiw` ignores `mean_stats`
      entirely and returns `beta = nothing`, so X + IW is a silent no-op
      today with or without match_id — out of scope here, but say so in
      the docstring rather than leaving it implicit.
- [ ] **5. Tests** (extend `test_error_ar1.jl`, `test_error_blocks_iw.jl`):
      - *Invariance* (the property that currently fails): duplicating a
        member row of a co-managed match leaves the fit exactly unchanged,
        AR(1) and IW. Compare **fit outputs** (params/NLL), not stats
        structs — `A_prior`, `personyear_rows`, `duplicate_rows`, `base`
        legitimately change; invariance holds under the default
        `model_adjacency=:binary` only (`:counts` weights the prior by row
        multiplicity by design).
      - *Equivalence*: on a fixture with one row per match, the
        match-grouped fit equals the current raw-row fit, AR(1) and IW.
        Use `standardize=false` (or `≈` with tight atol): the match path
        computes `y_mean/y_std` over Dict-ordered match outcomes, which
        can differ from the raw-path row order in the last ulp. Give match
        ids in row order so `observation_rows`' first-appearance ordering
        matches the raw path.
      - *Loading*: a two-member match contributes `(1/2, 1/2)` worker
        loadings under both error models (mirror the `build_match_V_stats`
        loading tests in `test_prepare.jl`).
      - *Validation*: multi-firm match errors under both paths; within-match
        `edge_index` disagreement errors; per-firm rank permutation enforced
        over matches.
      - *IW numerics*: `_aptpa_local` agrees with `_aptpa` on a
        match-grouped fixture; `S_ε` stays PSD.
      - *Mean structure*: AR(1)+`X` on a match fixture runs and matches the
        one-row-per-match raw fit.
- [ ] **6. Docs + CHANGELOG.** Replace the two "not supported" notes in the
      `suffstats` docstring with the match-level error-process definition and
      the semantic note (error draws live on spells, not member rows); update
      `assemble_block_precision`/`observation_rows` docstrings where they
      mention the restriction; revisit the `optimize_emiw` mmax==1
      `init.r` error wording ("single observation" — observations are now
      matches); CHANGELOG entry (next minor, v0.6.0).

## Advisor review (2026-08-21)

Verdict: plan sound, no design-level blocker. Its findings are folded in
above; the load-bearing one is the `K = length(yv)` change in step 1.
Facts the advisor verified that reviewers might wrongly flag:

- `logdet_R = (K − n_blocks)·log(1−η²)` with K = matches is **correct
  unchanged**: each firm's match-level block of size `m_i` contributes
  `(m_i−1)·log(1−η²)`.
- The singleton `S_int = −1` correction applies per observation — after
  collapse, per match. Correct as designed.
- Standardization needs no work: match-outcome validation and
  match-weighted `y_mean/y_std` run before the error-model branches.
- `mmax`/r-domain/`Ktot`/φ in `emiwblocks.jl` acquire the correct
  match-level meaning with zero code change; only `emblocks.jl` needs
  generalizing.
- Graph-only rows cannot leak into an error block: the finite/non-finite
  mixing check forbids straddling matches, and obs_mask subsetting happens
  before any collapse.
- `error_groups` already validates within-match group agreement the same
  way (prepare.jl ~606-613) — a template to copy for `edge_index`.

## Acceptance criteria (from the issue)

`fit_mle(...; match_id=..., error_eta=:estimate)` and
`fit_mle(...; match_id=..., error_blocks=:iw, firm_group=...)` run, satisfy
the duplication-invariance test, and reproduce today's raw-row results on
data with one row per match.

## Explicitly out of scope

- Caller-side changes (deleting the keep-first-member workaround,
  `--group-matches=true`, re-ranking `edge_index`) — different repo, after
  this lands.
- Multi-firm matches under the structured error models (hard error, by
  design).
- `Weighting(observations=:raw)` remains required, as for all error models.
