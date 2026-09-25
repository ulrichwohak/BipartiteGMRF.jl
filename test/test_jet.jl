using JET

function setup_jet_fixture()
    ss = suffstats_synthetic()
    model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.99)
    result = solve(model, ss, ExactCholesky(optim_iters=2, polish=false))
    return model, ss, result
end

function jet_exact_solve_flow(model, ss)
    Q = BipartiteGMRF.model_precision(model, 0.3, 0.8, 0.6)
    M = Q + (1.0 / 0.25^2) .* ss.design.VtV
    return cholesky(Symmetric(M)) \ ss.design.projected_y
end

function jet_covariance_flow(result)
    op = covariance(result; kind=:model)
    return cov_block(op; firms=(1,), workers=(1,))
end

function jet_batched_covariance_flow(result, batch_size)
    op = covariance(result; kind=:model)
    return cov_block(op; firms=(3, 1, 2), workers=(4, 1, 3, 2), batch_size)
end

@testset "JET" begin
    model, ss, result = setup_jet_fixture()

    JET.@test_opt target_modules=(BipartiteGMRF,) jet_exact_solve_flow(model, ss)
    JET.@test_opt target_modules=(BipartiteGMRF,) jet_covariance_flow(result)
    JET.@test_opt target_modules=(BipartiteGMRF,) jet_batched_covariance_flow(result, 2)
end
