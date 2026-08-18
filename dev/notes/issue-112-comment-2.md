## Direction chosen: integrate Ωᵢ out (random-effects IW mixing) — supersedes the MAP recommendation above

Maintainer decision on the remedy: **the Ωᵢ are nuisance parameters and are never estimated — they are integrated out** over a population law. This keeps everything the issue wanted (edge-level data, no within-firm averaging, cross-firm covariance intact, arbitrary PD realizations per firm) while replacing the non-existent profile MLE with a well-posed mixture MLE — the Kiefer–Wolfowitz resolution: maximize over the mixing distribution, not the realizations.

### Model

Ωᵢ ~ iid **Inverse-Wishart** across firms, parametrized by size-free scalars so blocks of different sizes are coherent:

- scale `Ψ_m = ω̄·[(1−r)I_m + r𝟙𝟙']`, built at any size from ω̄ (mean error variance — takes over σ_ε²'s reporting role) and r (mean within-firm error correlation);
- dispersion via the **marginal Student-t dof** δ: per-firm `νᵢ = δ + mᵢ − 1`, so each firm's integrated error block is `t_δ(0, Ψ_mᵢ)` with the same tail parameter regardless of degree.

This family is **projective**: a k×k principal submatrix of an `IW(δ+m−1, Ψ_m)` draw is `IW(δ+k−1, Ψ_k)` — the same law a size-k firm gets directly. Firm size is bookkeeping, not modeling. Equivalently (scalar gamma scale mixture): `uᵢ ~ Gamma(δ/2, δ/2)` iid — the identical law for every firm — with `εᵢ | uᵢ ~ N(0, Ψ_mᵢ/uᵢ)`.

Epistemic status: same as `α ~ N(0, K⁻¹)` — a random-effects distribution with ML-estimated hyperparameters, not a fixed subjective prior. Estimands: `(ρ, σ_a, σ_z, ω̄, r, δ)` — all finite, all pooled; `δ → ∞` recovers fixed equal blocks, small δ = heterogeneous firms. Only realizations stay arbitrary PD; the population mean is exchangeable within firm (AR-in-tenure Ψ is a later option). One constraint to state in docs: the mixing law must be **proper** — a flat prior over PD matrices diverges exactly like the profile likelihood does.

### Algorithm

EM/variational EM with latents `(α, {uᵢ})`. One honesty note: because α is shared across firms (the network), the uᵢ are coupled through it and the exact joint E-step is not closed-form; we use the mean-field factorization `q(α)·Πᵢq(uᵢ)` (coordinate ascent on the ELBO, monotone by construction):

- `q(α)` Gaussian with precision `K + Σᵢ ūᵢ Aᵢ'Ψᵢ⁻¹Aᵢ` — the already-verified workspace machinery at known blocks `Ψᵢ/ūᵢ`;
- `q(uᵢ) = Gamma((δ+mᵢ)/2, (δ+sᵢ)/2)` with `sᵢ = E_q[rᵢ'Ψᵢ⁻¹rᵢ]` read off the verified posterior moments (`Sᵢ = r̂ᵢr̂ᵢ' + AᵢP⁻¹Aᵢ'`);
- θ-step: same Fisher-identity score as now (`½tr(∂K(K⁻¹ − S_α))`); ω̄ closed-form given r; r and δ by 1-D searches.

If the MC gate shows the mean-field bound biting, the fallback is MCEM (Gibbs: α | u Gaussian, uᵢ | α gamma — both trivial draws), but VEM is deterministic and the natural v1.

### Acceptance criteria, updated

- Dense verification of every ELBO component on the toy problem (as before, machine precision).
- ELBO ≤ exact marginal (checked against brute-force integration over u on a 2-block toy).
- Large-δ reduction: recovers the fixed-blocks (`error_cov`-style) fit.
- **Criterion 4 rerun** (the real gate): caterpillar + dense-graph MC — `(ρ̂, σ̂_a, σ̂_z, ω̄̂, r̂)` centered at truth, where the profile-MLE version provably collapsed.
- Criterion 3 (reduction to `error_groups`) as re-scoped earlier.

Implementation starting now on `feat/free-error-blocks`; suffstats/API surface (`error_blocks=:free`, `firm_group`) unchanged — only the solver changes.
