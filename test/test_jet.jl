using JET

function setup_jet_fixture()
    ss = suffstats(BipartiteNormalizedModel, synthetic_df())
    model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.99)
    result = solve(
        model,
        ss,
        ExactCholesky(optim_iters=2, polish=false);
        decompose=nothing,
    )
    return model, ss, result
end

function jet_exact_solve_flow(model, ss)
    Q = BipartiteGMRF.model_precision(model, 0.3, 0.8, 0.6)
    M = Q + (1.0 / 0.25^2) .* ss.VtV
    return cholesky(Symmetric(M)) \ ss.projected_y
end

function jet_covariance_flow(result)
    op = covariance(result; kind=:model)
    return cov_block(op; firms=(1,), workers=(10,))
end

@testset "JET" begin
    model, ss, result = setup_jet_fixture()

    JET.@test_opt target_modules=(BipartiteGMRF,) jet_exact_solve_flow(model, ss)
    JET.@test_opt target_modules=(BipartiteGMRF,) jet_covariance_flow(result)
end
