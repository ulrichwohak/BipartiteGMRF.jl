@testset "operators" begin
    for model_type in (BipartiteNormalizedModel, BipartiteUnnormalizedModel, BipartiteSpectralModel)
        p = GMRFProblem(synthetic_df(); model_type=model_type)
        rho, sa, sz = 0.25, 0.7, 0.4
        qop = BipartiteGMRF.q_operator(p, rho, sa, sz)
        Q = BipartiteGMRF.model_precision(p.model, rho, sa, sz)
        x = collect(range(-0.4, 0.6; length=p.N_firms + p.N_workers))
        y = similar(x)
        qop(y, x)
        @test y ≈ Q * x atol=1e-10 rtol=1e-10
        mop = BipartiteGMRF.MOp(qop, p.VtV, similar(x), 1.0)
        @test mop isa BipartiteGMRF.MOp{typeof(qop)}
    end
    p_spectral = GMRFProblem(synthetic_df(); model_type=BipartiteSpectralModel, seed=7)
    @test p_spectral.model isa BipartiteSpectralModel

    pvs = @test_warn "variance-stable model no longer guarantees" GMRFProblem(
        synthetic_df();
        model_type=BipartiteVarianceStableModel,
    )
    rho, sa, sz = 0.2, 0.6, 0.5
    qop = BipartiteGMRF.q_operator(pvs, rho, sa, sz)
    Q = BipartiteGMRF.model_precision(pvs.model, rho, sa, sz)
    x = collect(range(0.1, 0.7; length=pvs.N_firms + pvs.N_workers))
    y = similar(x)
    qop(y, x)
    @test y ≈ Q * x atol=1e-10 rtol=1e-10

    lambda = 1.7
    scale = vcat(fill(sa, pvs.N_firms), fill(sz, pvs.N_workers))
    bop = BipartiteGMRF.make_qop_vs(pvs, rho, 1.0, 1.0)
    kop = BipartiteGMRF.ScaledMOp(
        bop,
        pvs.VtV,
        copy(scale),
        similar(x),
        similar(x),
        lambda,
    )
    kop(y, x)
    B = BipartiteGMRF.model_precision(pvs.model, rho, 1.0, 1.0)
    S = Diagonal(scale)
    @test y ≈ (B + lambda .* S * pvs.VtV * S) * x atol=1e-10 rtol=1e-10
end
