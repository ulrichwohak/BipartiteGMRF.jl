@testset "operators" begin
    for model_type in (BipartiteNormalizedModel, BipartiteUnnormalizedModel, BipartiteSpectralModel)
        ss = suffstats(model_type, synthetic_df())
        model = BipartiteGMRF._build_model(model_type, ss.A_prior, 0.99)
        rho, sa, sz = 0.25, 0.7, 0.4
        qop = BipartiteGMRF.q_operator(model, rho, sa, sz)
        Q = BipartiteGMRF.model_precision(model, rho, sa, sz)
        x = collect(range(-0.4, 0.6; length=ss.N_firms + ss.N_workers))
        y = similar(x)
        qop(y, x)
        @test y ≈ Q * x atol=1e-10 rtol=1e-10
        mop = BipartiteGMRF.MOp(qop, ss.VtV, similar(x), 1.0)
        @test mop isa BipartiteGMRF.MOp{typeof(qop)}
    end
    ss_spectral = suffstats(BipartiteSpectralModel, synthetic_df())
    model_spectral = BipartiteSpectralModel(ss_spectral.A_prior; seed=7)
    @test model_spectral isa BipartiteSpectralModel

    ss_vs = suffstats(BipartiteVarianceStableModel, synthetic_df())
    model_vs = @test_warn "variance-stable model no longer guarantees" BipartiteVarianceStableModel(
        ss_vs.A_prior,
    )
    rho, sa, sz = 0.2, 0.6, 0.5
    qop = BipartiteGMRF.q_operator(model_vs, rho, sa, sz)
    Q = BipartiteGMRF.model_precision(model_vs, rho, sa, sz)
    x = collect(range(0.1, 0.7; length=ss_vs.N_firms + ss_vs.N_workers))
    y = similar(x)
    qop(y, x)
    @test y ≈ Q * x atol=1e-10 rtol=1e-10

    lambda = 1.7
    scale = vcat(fill(sa, ss_vs.N_firms), fill(sz, ss_vs.N_workers))
    bop = BipartiteGMRF.make_qop_vs(model_vs, rho, 1.0, 1.0)
    kop = BipartiteGMRF.ScaledMOp(
        bop,
        ss_vs.VtV,
        copy(scale),
        similar(x),
        similar(x),
        lambda,
    )
    kop(y, x)
    B = BipartiteGMRF.model_precision(model_vs, rho, 1.0, 1.0)
    S = Diagonal(scale)
    @test y ≈ (B + lambda .* S * ss_vs.VtV * S) * x atol=1e-10 rtol=1e-10
end
