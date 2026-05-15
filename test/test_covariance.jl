@testset "covariance" begin
    result = fitted_exact()
    problem = result.problem
    sigma_a = result.sigma_a / problem.y_std
    sigma_z = result.sigma_z / problem.y_std
    Q = BipartiteGMRF.precision_matrix(problem, result.rho, sigma_a, sigma_z)
    Sigma = inv(Matrix(Q)) .* problem.y_std^2

    op = prior_covariance(result; units=:original)
    block = cov_block(op; firms=[1, 2], workers=[10])
    idx = [
        problem.firm_to_index[1],
        problem.firm_to_index[2],
        problem.N_firms + problem.worker_to_index[10],
    ]
    @test block.matrix ≈ Sigma[idx, idx] atol=1e-8 rtol=1e-8
    @test block.kind == :prior
    @test block.units == :original

    rect = cov_block(op; row_firms=[1], col_workers=[10, 11])
    @test size(rect.matrix) == (1, 2)
    @test rect.rows[1].side == :firm
    @test rect.cols[1].side == :worker

    post = posterior_covariance(result; units=:scaled)
    post_block = cov_block(post; firms=[1])
    @test size(post_block.matrix) == (1, 1)
    @test post_block.units == :scaled
end
