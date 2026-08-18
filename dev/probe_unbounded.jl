# Criterion-1 discriminator (issue #112): is the σ_a collapse an EM artifact,
# or is the free-Ω likelihood itself unbounded?
#
# The free-block model is y ~ N(0, A K(θ)⁻¹ A' + Ω), Ω = blkdiag(Ω_i) with each
# Ω_i free PD. Walk the *degenerate ray*: σ_a, σ_z → 0 (so K⁻¹ → 0) while
# Ω_i → r_i r_i' + δ I with r_i = y_i (the residual at α = 0). If the dense
# marginal NLL diverges to −∞ along this ray, there is no MLE and no EM fix.
using BipartiteGMRF, LinearAlgebra, SparseArrays, Printf

f = [1, 1, 1, 2, 2]
w = [1, 2, 3, 2, 4]
y = [0.5, -0.3, 0.9, 0.2, -0.5]

ss = suffstats(BipartiteVarianceStableModel, f, w, y;
    weighting = Weighting(observations = :raw), standardize = false,
    error_blocks = :free, firm_group = f)
fb = ss.free_blocks
nf, nw = ss.N_firms, ss.N_workers
model = BipartiteVarianceStableModel(ss.A_prior; rho_limit = 0.99)
ew = BipartiteGMRF.make_emfree_workspace(model, fb, nf, nw)
A = Matrix(fb.V)
rows = [findall(==(i), fb.block_of) for i in 1:length(fb.sizes)]

# Dense marginal NLL, same convention as `emfree_nll` (constants dropped).
function dense_nll(Ωs, rho, sa, sz)
    Om = zeros(length(y), length(y))
    for (i, rr) in enumerate(rows)
        Om[rr, rr] = Ωs[i]
    end
    Q = Matrix(BipartiteGMRF.model_precision(model, rho, sa, sz))
    Σ = A * inv(Q) * A' + Om
    return 0.5 * (logdet(Symmetric(Σ)) + dot(fb.y, Symmetric(Σ) \ fb.y))
end

println("=== degenerate ray: σ_a = σ_z = s, Ω_i = y_i y_i' + δI ===")
@printf("%8s %8s %16s %16s\n", "s", "δ", "dense NLL", "emfree_nll")
for k in 0:8
    s = 0.7 * 10.0^(-k)
    δ = 10.0^(-k)
    Ωs = [fb.y[rr] * fb.y[rr]' + δ * Matrix{Float64}(I, length(rr), length(rr))
          for rr in rows]
    nd = dense_nll(Ωs, 0.3, s, s)
    ne = BipartiteGMRF.emfree_nll(model, fb, ew, Ωs, 0.3, s, s, nf, nw)
    @printf("%8.1e %8.1e %16.6f %16.6f\n", s, δ, nd, ne)
end

# Control: hold σ_a, σ_z at sane values and shrink δ only. If the NLL still
# diverges, the degeneracy does not even need the σ → 0 limit.
println("\n=== δ → 0 only, σ_a = σ_z = 0.7 held fixed ===")
@printf("%8s %16s\n", "δ", "dense NLL")
for k in 0:8
    δ = 10.0^(-k)
    Ωs = [fb.y[rr] * fb.y[rr]' + δ * Matrix{Float64}(I, length(rr), length(rr))
          for rr in rows]
    @printf("%8.1e %16.6f\n", δ, dense_nll(Ωs, 0.3, 0.7, 0.7))
end

# Reference: NLL at a well-behaved interior point, for scale.
Ωref = [0.2^2 * Matrix{Float64}(I, m, m) for m in fb.sizes]
@printf("\ninterior reference NLL (Ω = 0.04·I, σ = 0.7): %.6f\n",
        dense_nll(Ωref, 0.3, 0.7, 0.7))
