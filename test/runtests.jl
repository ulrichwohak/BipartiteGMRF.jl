using Test
using DataFrames
using BipartiteGMRF

function synthetic_df()
    return DataFrame(
        firm_id = [1, 1, 2, 2, 3, 3, 1, 2],
        worker_id = [10, 11, 11, 12, 12, 13, 13, 10],
        y = [1.2, 0.7, 0.9, 1.5, 1.1, 0.4, 1.0, 0.8],
    )
end

@testset "prepare" begin
    p = GMRFProblem(synthetic_df(); prior=NormalizedPrior(), weighting=Weighting())
    @test p.N_firms == 3
    @test p.N_workers == 4
    @test p.K == 8
    @test p.firm_to_index[1] == 1
    @test p.worker_to_index[10] == 1
end

@testset "exact fit and covariance" begin
    result = gmrf_mle(
        synthetic_df();
        solver=ExactCholesky(optim_iters=5, polish=false),
        decompose=3,
        seed=1,
        verbose=false,
    )
    @test isfinite(result.nll)
    @test result.prior_decomposition !== nothing

    post = posterior_decomposition(result; probes=3, seed=1)
    @test isfinite(post.V_total)

    op = prior_covariance(result)
    block = cov_block(op; firms=[1, 2], workers=[10])
    @test size(block.matrix) == (3, 3)
    @test block.kind == :prior
    @test block.units == :original
end

@testset "capabilities" begin
    p = GMRFProblem(synthetic_df(); prior=VarianceStablePrior())
    @test_throws ArgumentError solve(p, ExactCholesky(optim_iters=2); decompose=false)
end
