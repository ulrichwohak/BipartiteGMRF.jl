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

    @testset "match-grouped fit (issue #120)" begin
        # ── one match per row reproduces the raw-row fit ──
        res_raw = fit_mle(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = f,
            solver = EMIWBlocks(max_iter = 80, ftol = 1e-9))
        res_mid = fit_mle(BipartiteVarianceStableModel, f, w, y;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = f, match_id = collect(1:length(y)),
            solver = EMIWBlocks(max_iter = 80, ftol = 1e-9))
        @test res_mid.stats.K == res_raw.stats.K
        for field in (:rho, :sigma_a, :sigma_z, :sigma_epsilon, :nll)
            @test getproperty(res_mid, field) ≈ getproperty(res_raw, field) atol = 1e-10
        end
        @test res_mid.metadata.error_corr_r ≈ res_raw.metadata.error_corr_r atol = 1e-10
        @test res_mid.metadata.t_dof_delta ≈ res_raw.metadata.t_dof_delta atol = 1e-6

        # ── duplicating a member row of a co-managed match: fit unchanged ──
        # Firm 1: match 1 co-managed (workers 1, 2), match 2 (worker 3).
        # Firm 2: match 3 (worker 2), match 4 (worker 4).
        fm = [1, 1, 1, 2, 2]
        wm = [1, 2, 3, 2, 4]
        ym = [0.7, 0.7, 0.9, 0.2, -0.5]
        midm = [1, 1, 2, 3, 4]
        res0 = fit_mle(BipartiteVarianceStableModel, fm, wm, ym;
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = fm, match_id = midm,
            solver = EMIWBlocks(max_iter = 60, ftol = 1e-9))
        @test res0.stats.K == 4                       # matches, not rows
        @test length(res0.metadata.u_bar) == 2        # one u_i per firm
        resd = fit_mle(BipartiteVarianceStableModel,
            vcat(fm, 1), vcat(wm, 2), vcat(ym, 0.7);
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = vcat(fm, 1), match_id = vcat(midm, 1),
            solver = EMIWBlocks(max_iter = 60, ftol = 1e-9))
        for field in (:rho, :sigma_a, :sigma_z, :sigma_epsilon, :nll)
            @test getproperty(resd, field) ≈ getproperty(res0, field) atol = 1e-12
        end
        @test resd.metadata.error_corr_r ≈ res0.metadata.error_corr_r atol = 1e-12
        @test resd.metadata.u_bar ≈ res0.metadata.u_bar atol = 1e-12

        # ── a match spanning two firms is a hard error through suffstats ──
        @test_throws ArgumentError suffstats(BipartiteVarianceStableModel,
            [1, 2], [1, 1], [1.0, 1.0];
            weighting = Weighting(observations = :raw), standardize = false,
            error_blocks = :iw, firm_group = [1, 2], match_id = [1, 1])
    end

    # The E-step's per-firm posterior correction A_i M^-1 A_i' used to cost one
    # full solve against the whole factorization per block row (issue #116:
    # K_tot global solves per iteration, hours per iteration at n ~ 2M). It now
    # reads one selected inverse per E-step. `_aptpa` is retained as the
    # reference implementation and these tests pin the new path to it.
    @testset "A M^-1 A' by selected inversion (issue #116)" begin
        function setup(fs, ws_; rho = 0.4, sa = 0.8, sz = 0.3, scale = 1.0)
            ys = randn(MersenneTwister(5), length(fs))
            ss = suffstats(BipartiteVarianceStableModel, fs, ws_, ys;
                weighting = Weighting(observations = :raw), standardize = false,
                error_blocks = :iw, firm_group = fs)
            model = BipartiteVarianceStableModel(ss.A_prior; rho_limit = 0.99)
            fb, nf, nw = ss.error_blocks, ss.N_firms, ss.N_workers
            ew = BipartiteGMRF.make_em_blocks_workspace(model, fb, nf, nw)
            Om = [scale .* (Matrix{Float64}(0.7LinearAlgebra.I, m, m) .+ 0.1) for m in fb.sizes]
            P, _ = BipartiteGMRF.assemble_block_precision(fb, Om, nf, nw)
            Q = BipartiteGMRF.model_precision(model, rho, sa, sz)
            BipartiteGMRF.GaussianMarkovRandomFields.update_precision_values!(
                ew.ws_M, BipartiteGMRF._align_to_ws(Q, P, ew.ws_M))
            BipartiteGMRF.GaussianMarkovRandomFields.ensure_numeric!(ew.ws_M)
            return ss, fb, ew, Q + P
        end

        fs = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6]
        wsx = [1, 2, 3, 2, 4, 5, 1, 6, 7, 3, 7, 5, 2]

        @testset "the selinv pattern covers every block, exactly" begin
            ss, fb, ew, M = setup(fs, wsx)
            Sig = BipartiteGMRF.GaussianMarkovRandomFields.selinv_extract_at(ew.ws_M, ew.selinv_pattern)
            Minv = inv(Matrix(M))
            # Every S_i x S_i position must be present *and* exact. If selinv
            # returned a structural zero here the correction would be biased
            # downward with no error raised anywhere.
            for cols in ew.block_cols, p in cols, q in cols
                @test Sig[p, q] ≈ Minv[p, q] atol = 1e-12
            end
        end

        @testset "agrees with the reference, and stays PSD" begin
            for (label, args, kw) in (
                    ("cyclic", (fs, wsx), NamedTuple()),
                    ("path", ([1, 1, 2, 2, 3, 3, 4], [1, 2, 2, 3, 3, 4, 4]), NamedTuple()),
                    ("ill-conditioned", (fs, wsx),
                     (scale = 1e-9, sa = 50.0, sz = 40.0, rho = 0.95)))
                ss, fb, ew, _ = setup(args...; kw...)
                rows = BipartiteGMRF._block_rows(fb)
                Sig = BipartiteGMRF.GaussianMarkovRandomFields.selinv_extract_at(ew.ws_M, ew.selinv_pattern)
                for i in eachindex(rows)
                    rr = rows[i]
                    ref = BipartiteGMRF._aptpa(ew.ws_M, fb.V[rr, :])
                    new = BipartiteGMRF._aptpa_local(Sig, ew.block_cols[i], rr, ew)
                    @test maximum(abs.(new .- ref)) <=
                          1e-10 * max(1.0, maximum(abs.(ref)))
                    # PSD is load-bearing: S_eps feeds the Gamma rate, which is
                    # then passed to log. Losing it gives NaN, not a small error.
                    @test minimum(eigvals(Symmetric(new))) > -1e-10
                end
            end
        end

        # The actual invariant the bug violated: cost must not scale with the
        # node count. Two graphs with identical block structure, one padded with
        # extra nodes, must cost about the same.
        @testset "cost does not scale with the node count" begin
            function estep_alloc(pad::Int)
                fs2 = vcat(fs, collect(7:(6 + pad)), collect(7:(6 + pad)))
                ws2 = vcat(wsx, collect(8:(7 + pad)), collect((8 + pad):(7 + 2pad)))
                ss, fb, ew, _ = setup(fs2, ws2)
                rows = BipartiteGMRF._block_rows(fb)
                run() = begin
                    Sig = BipartiteGMRF.GaussianMarkovRandomFields.selinv_extract_at(
                        ew.ws_M, ew.selinv_pattern)
                    s = 0.0
                    for i in eachindex(rows)
                        s += sum(BipartiteGMRF._aptpa_local(
                            Sig, ew.block_cols[i], rows[i], ew))
                    end
                    s
                end
                run()                       # compile
                return @allocated(run()), length(rows)
            end
            a_small, b_small = estep_alloc(20)
            a_big, b_big = estep_alloc(200)
            # Blocks grew ~10x; allocation per block must not grow with n. The
            # old implementation allocated a dense length-n vector per row, so
            # per-block cost grew linearly in n and this would fail loudly.
            @test a_big / b_big < 3 * (a_small / b_small)
        end
    end

    @testset "match-collapsed rows through the EM internals (issue #120)" begin
        # Firm 1: match 1 co-managed (workers 1, 2), match 2 (worker 2 again —
        # a worker shared across two matches of the same firm), match 3
        # (worker 3). Firm 2: match 4 co-managed (workers 3, 4). Firm 3:
        # match 5 (worker 1), a singleton block.
        fm = [1, 1, 1, 1, 2, 2, 3]
        wm = [1, 2, 2, 3, 3, 4, 1]
        ym = [0.5, 0.5, -0.3, 0.9, 0.2, 0.2, -0.5]
        mid = [1, 1, 2, 3, 4, 4, 5]
        nf, nw = 3, 4
        Vm, yvm, bom, szm =
            BipartiteGMRF.build_block_V_stats(fm, wm, ym, fm, nf, nw, mid)
        fb = BipartiteGMRF.FirmBlockStats(Vm, yvm, bom, szm)
        @test szm == [3, 1, 1]   # matches per firm

        @testset "assembly equals the dense A'Ω⁻¹A on weighted rows" begin
            Om = [Matrix{Float64}(0.7LinearAlgebra.I, m, m) .+ 0.1 for m in szm]
            P, b = BipartiteGMRF.assemble_block_precision(fb, Om, nf, nw)
            A = Matrix(Vm)
            K = length(yvm)
            Oinv = zeros(K, K)
            rows = BipartiteGMRF._block_rows(fb)
            for i in eachindex(rows)
                Oinv[rows[i], rows[i]] = inv(Om[i])
            end
            @test Matrix(P) ≈ A' * Oinv * A atol = 1e-12
            @test b ≈ A' * Oinv * yvm atol = 1e-12
        end

        @testset "workspace node sets and the local correction" begin
            A_prior = sparse(fm, wm, ones(Float64, length(fm)), nf, nw)
            A_prior.nzval .= 1.0
            model = BipartiteVarianceStableModel(A_prior; rho_limit = 0.99)
            ew = BipartiteGMRF.make_em_blocks_workspace(model, fb, nf, nw)
            # Block 1's node set is firm 1 plus the workers of ALL its
            # matches, deduplicated: {1} ∪ {nf+1, nf+2, nf+3}.
            @test ew.block_cols[1] == [1, nf + 1, nf + 2, nf + 3]
            @test ew.block_cols[3] == [3, nf + 1]

            Om = [Matrix{Float64}(0.7LinearAlgebra.I, m, m) .+ 0.1 for m in szm]
            P, _ = BipartiteGMRF.assemble_block_precision(fb, Om, nf, nw)
            Q = BipartiteGMRF.model_precision(model, 0.4, 0.8, 0.3)
            BipartiteGMRF.GaussianMarkovRandomFields.update_precision_values!(
                ew.ws_M, BipartiteGMRF._align_to_ws(Q, P, ew.ws_M))
            BipartiteGMRF.GaussianMarkovRandomFields.ensure_numeric!(ew.ws_M)
            rows = BipartiteGMRF._block_rows(fb)
            Sig = BipartiteGMRF.GaussianMarkovRandomFields.selinv_extract_at(
                ew.ws_M, ew.selinv_pattern)
            Minv = inv(Matrix(Q + P))
            for cols in ew.block_cols, p in cols, q in cols
                @test Sig[p, q] ≈ Minv[p, q] atol = 1e-12
            end
            for i in eachindex(rows)
                rr = rows[i]
                ref = BipartiteGMRF._aptpa(ew.ws_M, fb.V[rr, :])
                new = BipartiteGMRF._aptpa_local(Sig, ew.block_cols[i], rr, ew)
                @test maximum(abs.(new .- ref)) <=
                      1e-10 * max(1.0, maximum(abs.(ref)))
                @test minimum(eigvals(Symmetric(new))) > -1e-10
            end
        end

        @testset "chained co-managed matches keep the clique in ws_M's pattern" begin
            # Three chained two-worker matches at one firm: {w1,w2}, {w2,w3},
            # {w3,w4}. At the reference blocks the assembled P0 entry at the
            # (w2, w3) position cancels ANALYTICALLY (1/4 − 1/4·1·1 = 0); only
            # roundoff in inv(cholesky(·)) keeps it nonzero. The workspace must
            # therefore seed ws_M's symbolic pattern structurally — every
            # S_i × S_i position stored — because the runtime M is genuinely
            # nonzero there once r moves off the reference value.
            fc = [1, 1, 1, 1, 1, 1]
            wc = [1, 2, 2, 3, 3, 4]
            yc = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]
            midc = [1, 1, 2, 2, 3, 3]
            Vc, yvc, boc, szc =
                BipartiteGMRF.build_block_V_stats(fc, wc, yc, fc, 1, 4, midc)
            fbc = BipartiteGMRF.FirmBlockStats(Vc, yvc, boc, szc)
            Ap = sparse(fc, wc, ones(Float64, length(fc)), 1, 4)
            Ap.nzval .= 1.0
            modelc = BipartiteVarianceStableModel(Ap; rho_limit = 0.99)
            ewc = BipartiteGMRF.make_em_blocks_workspace(modelc, fbc, 1, 4)
            stored(A, i, j) = insorted(i, @view rowvals(A)[nzrange(A, j)])
            for p in ewc.block_cols[1], q in ewc.block_cols[1]
                @test stored(ewc.ws_M.Q, p, q)
            end
        end
    end
end
