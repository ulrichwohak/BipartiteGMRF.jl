@testset "covariance" begin
    result = fitted_exact()
    stats = result.stats
    sigma_a = result.sigma_a / stats.y_std
    sigma_z = result.sigma_z / stats.y_std
    Q = BipartiteGMRF.model_precision(result.model, result.rho, sigma_a, sigma_z)
    Sigma = inv(Matrix(Q)) .* stats.y_std^2

    op = covariance(result; kind=:model, units=:original)
    @test op isa CovarianceOperator
    block = cov_block(op; firms=[1, 2], workers=[1])
    idx = [1, 2, stats.N_firms + 1]
    @test block.matrix ≈ Sigma[idx, idx] atol=1e-8 rtol=1e-8
    @test block.kind == :model
    @test block.units == :original

    rect = cov_block(op; row_firms=[1], col_workers=[1, 2])
    @test size(rect.matrix) == (1, 2)
    @test rect.rows[1].side == :firm
    @test rect.cols[1].side == :worker

    post = covariance(result; kind=:fitted, units=:scaled)
    post_block = cov_block(post; firms=[1])
    @test size(post_block.matrix) == (1, 1)
    @test post_block.units == :scaled

    @testset "batched extraction" begin
        # Exercise full batches, a short final batch, repeated/reordered indices,
        # and both orientations of the rectangular extraction optimization.
        for batch_size in (1, 2, 3, 5, 16)
            all_nodes = vcat([3, 1, 2], stats.N_firms .+ [4, 1, 3, 2])
            full = cov_block(op; firms=[3, 1, 2], workers=[4, 1, 3, 2], batch_size)
            @test full.matrix ≈ Sigma[all_nodes, all_nodes] atol=1e-8 rtol=1e-8

            rows = [stats.N_firms + 4, stats.N_firms + 1, stats.N_firms + 4]
            cols = [2, 1]
            tall = cov_block(op; row_workers=[4, 1, 4], col_firms=[2, 1], batch_size)
            wide = cov_block(op; row_firms=[2, 1], col_workers=[4, 1, 4], batch_size)
            @test tall.matrix ≈ Sigma[rows, cols] atol=1e-8 rtol=1e-8
            @test wide.matrix ≈ Sigma[cols, rows] atol=1e-8 rtol=1e-8
        end
        @test_throws ArgumentError cov_block(op; firms=[1], batch_size=0)
    end
end
