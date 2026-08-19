using Statistics: mean, std
using Random: MersenneTwister, rand, randn

# Marsaglia–Tsang Gamma(shape, 1) sampler (shape ≥ 1) — test-local helper so
# the suite does not need Distributions' samplers.
function rand_gamma(rng, shape::Float64)
    d = shape - 1 / 3
    c = 1 / sqrt(9d)
    while true
        x = randn(rng)
        v = (1 + c * x)^3
        v <= 0 && continue
        u = rand(rng)
        log(u) < 0.5 * x^2 + d - d * v + d * log(v) && return d * v
    end
end

@testset "integrated error blocks (EMIWBlocks, issue #112 remedy E)" begin
    f = [1, 1, 1, 2, 2]
    w = [1, 2, 3, 2, 4]
    y = [0.5, -0.3, 0.9, 0.2, -0.5]

    @testset "ELBO u-terms: stable form matches naive, vanishes as δ → ∞" begin
        δ, m, s = 7.0, 3.0, 1.2
        a, b = (δ + m) / 2, (δ + s) / 2
        dg, lg = BipartiteGMRF.digamma, BipartiteGMRF.loggamma
        naive = ((δ / 2) * log(δ / 2) - lg(δ / 2) +
                 (δ / 2 - 1) * (dg(a) - log(b)) - (δ / 2) * (a / b)) +
                (a - log(b) + lg(a) + (1 - a) * dg(a))
        @test BipartiteGMRF._elbo_u_terms(a, b, δ) ≈ naive atol = 1e-10
        for δ_ in (10.0, 1e4, 1e8, 1e12)
            v = BipartiteGMRF._elbo_u_terms((δ_ + 3) / 2, (δ_ + 3.0) / 2, δ_)
            @test v <= 1e-12
            δ_ >= 1e8 && @test abs(v) < 1e-6
        end
    end

    @testset "ψ→θ codec stays strictly inside the ρ domain at saturation" begin
        for lim in (0.99, 0.26887747717394245), s in (-30.0, 30.0)
            ρ, _, _ = BipartiteGMRF._em_blocks_θ_from_ψ([s, 0.0, 0.0], lim)
            @test abs(ρ) < lim
        end
    end

    @testset "equicorrelation closed forms match dense" begin
        for (m, r) in ((1, 0.0), (3, 0.4), (4, -0.2), (5, 0.9))
            R = BipartiteGMRF._equicorr(m, r)
            @test R ≈ (1 - r) * Matrix{Float64}(LinearAlgebra.I, m, m) .+ r atol = 1e-14
            @test BipartiteGMRF._equicorr_logdet(m, r) ≈ logdet(Symmetric(R)) atol = 1e-12
            S = randn(MersenneTwister(m), m, m); S = S * S' + LinearAlgebra.I
            @test BipartiteGMRF._equicorr_trinv(S, r) ≈ tr(inv(R) * S) atol = 1e-10
        end
    end

    @testset "solver type validation" begin
        @test_throws ArgumentError EMIWBlocks(delta = 2.0)
        @test_throws ArgumentError EMIWBlocks(delta = :wrong)
        @test_throws ArgumentError EMIWBlocks(r = 1.5)
        @test EMIWBlocks(delta = 5.0, r = 0.3) isa EMIWBlocks
        ss = suffstats(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = f)
        model = BipartiteVarianceStableModel(ss.A_prior; rho_limit = 0.99)
        @test_throws ArgumentError BipartiteGMRF.validate_capability(model, ss, ExactCholesky())
        @test BipartiteGMRF.validate_capability(model, ss, EMIWBlocks()) === nothing
        # fixed r below the data-dependent PD lower bound must be rejected, not
        # silently clamped (mmax = 3 here ⇒ r_lo ≈ -0.5)
        @test_throws ArgumentError BipartiteGMRF.optimize_emiw(model, ss, EMIWBlocks(r = -0.9))
    end

    @testset "firm_group guardrails" begin
        # firm_group must be the rowwise firm mapping (one latent u_i per firm)
        @test_throws ArgumentError suffstats(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = [1, 1, 2, 3, 3])  # splits firm 1
        @test_throws ArgumentError suffstats(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = [1, 1, 1, 1, 1])  # merges two firms
    end

    @testset "ELBO is a valid lower bound on the integrated log-likelihood" begin
        ss = suffstats(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = f)
        model = BipartiteVarianceStableModel(ss.A_prior; rho_limit = 0.99)
        fit = BipartiteGMRF.optimize_emiw(model, ss,
            EMIWBlocks(max_iter = 300, ftol = 1e-10))
        elbo = -fit.nll

        A = Matrix(ss.error_blocks.V)
        Q = Matrix(BipartiteGMRF.model_precision(model, fit.rho, fit.sigma_a, fit.sigma_z))
        AKA = A * inv(Q) * A'
        rows = [[1, 2, 3], [4, 5]]
        δ, φ, r = fit.delta, fit.phi, fit.r
        Ψs = [φ .* BipartiteGMRF._equicorr(length(rr), r) for rr in rows]

        rng = MersenneTwister(7)
        ndraw = 200_000
        logps = Vector{Float64}(undef, ndraw)
        gshape = δ / 2
        for t in 1:ndraw
            u1 = rand_gamma(rng, gshape) / gshape
            u2 = rand_gamma(rng, gshape) / gshape
            Σ = copy(AKA)
            Σ[rows[1], rows[1]] .+= Ψs[1] ./ u1
            Σ[rows[2], rows[2]] .+= Ψs[2] ./ u2
            C = cholesky(Symmetric(Σ))
            logps[t] = -0.5 * (length(y) * log(2π) + logdet(C) + dot(y, C \ y))
        end
        mx = maximum(logps)
        logp_mc = mx + log(mean(exp.(logps .- mx)))
        wgt = exp.(logps .- mx)
        se = std(wgt) / (mean(wgt) * sqrt(ndraw))
        @test elbo <= logp_mc + 4 * se
        @test elbo >= logp_mc - 0.5
    end

    @testset "ELBO trace is monotone non-decreasing" begin
        ss = suffstats(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = f)
        model = BipartiteVarianceStableModel(ss.A_prior; rho_limit = 0.99)
        fit = BipartiteGMRF.optimize_emiw(model, ss,
            EMIWBlocks(max_iter = 40, ftol = 1e-14))
        @test all(diff(fit.elbo_trace) .>= -1e-7)
    end

    @testset "large-δ, r=0 reduction: iid MLE is a VEM fixed point" begin
        Np = 20
        fp = Int[]; wp = Int[]
        for i in 1:Np
            push!(fp, i); push!(wp, i)
            i < Np && (push!(fp, i + 1); push!(wp, i))
        end
        Ap = sparse(fp, wp, ones(length(fp)), Np, Np)
        mp = BipartiteVarianceStableModel(Ap; rho_limit = 0.99)
        Qp = Matrix(BipartiteGMRF.model_precision(mp, 0.4, 0.7, 0.5))
        rng = MersenneTwister(11)
        θl = transpose(cholesky(Symmetric(Qp)).L) \ randn(rng, 2Np)
        yp = [θl[fp[k]] + θl[Np + wp[k]] + 0.2 * randn(rng) for k in eachindex(fp)]

        ss0 = suffstats(BipartiteVarianceStableModel, fp, wp, yp;
            weighting = Weighting(observations = :raw), standardize = false)
        res0 = fit_mle(BipartiteVarianceStableModel, ss0; solver = ExactCholesky())
        @test res0.sigma_a > 0.1

        ss = suffstats(BipartiteVarianceStableModel, fp, wp, yp;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = fp)
        model = BipartiteVarianceStableModel(ss.A_prior; rho_limit = 0.99)
        fit = BipartiteGMRF.optimize_emiw(model, ss,
            EMIWBlocks(max_iter = 500, ftol = 1e-12, delta = 1e8, r = 0.0);
            init = (rho = res0.rho, sigma_a = res0.sigma_a, sigma_z = res0.sigma_z,
                    phi = res0.sigma_epsilon^2))
        @test fit.rho ≈ res0.rho atol = 1e-3
        @test fit.sigma_a ≈ res0.sigma_a atol = 1e-3
        @test fit.sigma_z ≈ res0.sigma_z atol = 1e-3
        @test sqrt(fit.omega_bar) ≈ res0.sigma_epsilon atol = 1e-3
    end

    @testset "fit_mle end-to-end surface and metadata" begin
        res = fit_mle(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = true,
            error_blocks = :iw, firm_group = f,
            solver = EMIWBlocks(max_iter = 300, ftol = 1e-9))
        @test res.solver isa EMIWBlocks
        @test res.metadata.objective == :elbo
        @test res.metadata.t_dof_delta > 2
        @test length(res.metadata.u_bar) == 2
        @test res.metadata.omega_bar ≈ res.sigma_epsilon^2 rtol = 1e-8
        @test -1 < res.metadata.error_corr_r < 1
        @test dof(res) == 6    # ρ, σ_a, σ_z, φ, r, δ
        resfix = fit_mle(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = true,
            error_blocks = :iw, firm_group = f,
            solver = EMIWBlocks(max_iter = 50, ftol = 1e-9, delta = 10.0, r = 0.0))
        @test dof(resfix) == 4
    end
end