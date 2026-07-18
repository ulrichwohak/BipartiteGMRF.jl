@testset "covariance" begin
    result = fitted_exact()
    stats = result.stats
    sigma_a = result.sigma_a / stats.y_std
    sigma_z = result.sigma_z / stats.y_std
    Q = BipartiteGMRF.model_precision(result.model, result.rho, sigma_a, sigma_z)
    Sigma = inv(Matrix(Q)) .* stats.y_std^2

    op = covariance(result; kind=:model, units=:original)
    @test op isa CovarianceOperator
    block = cov_block(op; firms=[1, 2], workers=[10])
    idx = [
        stats.firm_to_index[1],
        stats.firm_to_index[2],
        stats.N_firms + stats.worker_to_index[10],
    ]
    @test block.matrix ≈ Sigma[idx, idx] atol=1e-8 rtol=1e-8
    @test block.kind == :model
    @test block.units == :original

    rect = cov_block(op; row_firms=[1], col_workers=[10, 11])
    @test size(rect.matrix) == (1, 2)
    @test rect.rows[1].side == :firm
    @test rect.cols[1].side == :worker

    post = covariance(result; kind=:fitted, units=:scaled)
    post_block = cov_block(post; firms=[1])
    @test size(post_block.matrix) == (1, 1)
    @test post_block.units == :scaled
end
