@testset "decomposition" begin
    result = fitted_exact(; decompose=3)
    prior = result.prior_decomposition
    @test prior !== nothing
    @test prior.kind == :prior
    @test prior.V_total ≈ prior.V_firm + prior.V_worker + prior.V_cross + prior.V_epsilon

    posterior = posterior_decomposition(result; probes=3, seed=1)
    @test posterior.kind == :posterior
    @test posterior.V_total ≈ posterior.V_firm + posterior.V_worker + posterior.V_cross + posterior.V_epsilon
    @test isfinite(posterior.metadata.covariance)

    edge_target = prior_decomposition(result; probes=3, seed=2, target=:edge)
    @test edge_target.target == :edge
    @test isfinite(edge_target.V_total)
end
