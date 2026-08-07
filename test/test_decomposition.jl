@testset "decomposition" begin
    result = fitted_exact()
    md = decompose(result; kind=:model, probes=3, seed=1)
    @test md.kind == :model
    @test md.V_firm > 0
    @test md.V_total ≈ md.V_firm + md.V_worker + md.V_cross + md.V_epsilon

    fd = decompose(result; kind=:fitted, probes=3, seed=1)
    @test fd.kind == :fitted
    @test fd.V_total ≈ fd.V_firm + fd.V_worker + fd.V_cross + fd.V_epsilon
    @test isfinite(fd.metadata.covariance)

    edge_target = decompose(result; kind=:model, probes=3, seed=2, target=:edge)
    @test edge_target.target == :edge
    @test isfinite(edge_target.V_total)
end
