@testset "operators" begin
    for prior in (NormalizedPrior(), UnnormalizedPrior(), SpectralPrior())
        p = GMRFProblem(synthetic_df(); prior=prior)
        rho, sa, sz = 0.25, 0.7, 0.4
        qop = BipartiteGMRF.q_operator(p, rho, sa, sz)
        Q = BipartiteGMRF.precision_matrix(p, rho, sa, sz)
        x = collect(range(-0.4, 0.6; length=p.N_firms + p.N_workers))
        y = similar(x)
        qop(y, x)
        @test y ≈ Q * x atol=1e-10 rtol=1e-10
    end

    pvs = GMRFProblem(synthetic_df(); prior=VarianceStablePrior())
    rho, sa, sz = 0.2, 0.6, 0.5
    qop = BipartiteGMRF.q_operator(pvs, rho, sa, sz)
    Q = BipartiteGMRF.precision_matrix(pvs, rho, sa, sz)
    x = collect(range(0.1, 0.7; length=pvs.N_firms + pvs.N_workers))
    y = similar(x)
    qop(y, x)
    @test y ≈ Q * x atol=1e-10 rtol=1e-10
end
