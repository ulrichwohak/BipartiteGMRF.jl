import CommonSolve
import Distributions
import Optim

@testset "StatsAPI / Distributions alignment" begin
    @testset "extended external bindings" begin
        @test fit_mle === Distributions.fit_mle
        @test suffstats === Distributions.suffstats
        @test SufficientStats === Distributions.SufficientStats
        @test params === Distributions.params
        @test solve === CommonSolve.solve
        @test converged === Optim.converged
        @test coef === StatsAPI.coef
        @test coefnames === StatsAPI.coefnames
        @test BipartiteGMRFStats <: Distributions.SufficientStats
    end

    result = fitted_exact()

    @testset "coef / coefnames / params" begin
        @test coef(result) ==
            [result.rho, result.sigma_a, result.sigma_z, result.sigma_epsilon]
        @test coefnames(result) == ["rho", "sigma_a", "sigma_z", "sigma_epsilon"]
        p = params(result)
        @test p.rho == result.rho
        @test p.sigma_epsilon == result.sigma_epsilon
        @test p.rho_eps === nothing
    end

    @testset "dof / isfitted / islinear / aic / bic" begin
        @test dof(result) == 4
        @test isfitted(result)
        @test !islinear(result)
        @test aic(result) ≈ -2.0 * loglikelihood(result) + 2.0 * dof(result)
        @test bic(result) ≈ -2.0 * loglikelihood(result) + dof(result) * log(nobs(result))
        @test isfinite(aic(result)) && isfinite(bic(result))
        @test converged(result) isa Bool
    end

    @testset "dof excludes fixed parameters" begin
        ss = suffstats(BipartiteNormalizedModel, synthetic_df())
        model = BipartiteNormalizedModel(ss.A_prior)
        fixed = solve(model, ss, ExactCholesky(optim_iters=2, polish=false);
            fix_rho=0.2, decompose=nothing, seed=1)
        @test dof(fixed) == 3
        @test length(coef(fixed)) == 4  # coef reports all parameters, fixed included
    end

    @testset "fixed rho_eps fit" begin
        eff = fit_mle(BipartiteNormalizedModel, repeated_df();
            solver=ExactCholesky(optim_iters=3, polish=false),
            weighting=Weighting(observations=:effective, rho_eps=0.3),
            decompose=nothing, seed=1)
        @test isfinite(eff.nll)
        @test eff.rho_eps == 0.3
        @test dof(eff) == 4          # rho_eps fixed, not estimated
        @test length(coef(eff)) == 5
        @test coefnames(eff)[end] == "rho_eps"
        @test params(eff).rho_eps == 0.3
        @test isfinite(loglikelihood(eff))
    end

    @testset "estimated rho_eps dof" begin
        eff = fit_mle(BipartiteNormalizedModel, repeated_df();
            solver=ExactCholesky(optim_iters=3, polish=false),
            weighting=Weighting(observations=:effective, rho_eps=:estimate),
            decompose=nothing, seed=1)
        @test dof(eff) == 5
        @test length(coef(eff)) == 5
    end

    @testset "loglikelihood: dense reference, standardization-invariant" begin
        for standardize in (true, false)
            r = fit_mle(BipartiteNormalizedModel, synthetic_df();
                solver=ExactCholesky(optim_iters=5, polish=false),
                standardize=standardize, decompose=nothing, seed=1)
            stats = r.stats
            k = stats.K
            n = stats.N_firms + stats.N_workers
            obs_rows = repeat(1:k, 2)
            entity_cols = vcat(stats.base_f_rows, stats.N_firms .+ stats.base_w_cols)
            V = sparse(obs_rows, entity_cols, ones(Float64, 2k), k, n)
            # Original-units parameters and mean-centered original-units outcome.
            Q = BipartiteGMRF.model_precision(r.model, r.rho, r.sigma_a, r.sigma_z)
            Sigma_y = Matrix(V * inv(Matrix(Q)) * transpose(V)) +
                r.sigma_epsilon^2 * Matrix{Float64}(I, k, k)
            yc = stats.base_y .* stats.y_std
            ll_dense = -0.5 * (k * log(2pi) + logdet(Symmetric(Sigma_y)) + dot(yc, Sigma_y \ yc))
            @test loglikelihood(r) ≈ ll_dense atol=1e-6 rtol=1e-6
        end
    end

    @testset "fit_mle(model, ss) overload matches solve" begin
        ss = suffstats(BipartiteNormalizedModel, synthetic_df())
        model = BipartiteNormalizedModel(ss.A_prior; rho_limit=0.9)
        r1 = fit_mle(model, ss; solver=ExactCholesky(optim_iters=3, polish=false), seed=1)
        r2 = solve(model, ss, ExactCholesky(optim_iters=3, polish=false); seed=1)
        @test r1.rho == r2.rho
        @test r1.nll == r2.nll
        @test BipartiteGMRF.rho_limit(r1.model) == 0.9
    end

    @testset "argument validation" begin
        ss = suffstats(BipartiteNormalizedModel, synthetic_df())
        @test_throws ArgumentError fit_mle(BipartiteNormalizedModel, ss; rho_limit=:auto)
        @test_throws ArgumentError fit_mle(BipartiteNormalizedModel, ss;
            solver=ExactCholesky(optim_iters=2, polish=false), decompose=0)
    end
end

@testset "decomposition and covariance edge cases" begin
    result = fitted_exact()

    @testset "invalid kinds and units" begin
        @test_throws ArgumentError decompose(result; kind=:bogus)
        @test_throws ArgumentError covariance(result; kind=:bogus)
        @test_throws ArgumentError covariance(result; kind=:model, units=:bogus)
    end

    @testset "personyear target" begin
        py = decompose(result; kind=:model, probes=3, seed=2, target=:personyear)
        @test py.target == :personyear
        @test isfinite(py.V_total)
        @test py.V_total ≈ py.V_firm + py.V_worker + py.V_cross + py.V_epsilon
    end

    @testset "decomposition invariant to verbose" begin
        hutch = fit_mle(BipartiteNormalizedModel, synthetic_df();
            solver=HutchSLQ(logdet_probes=2, lanczos_iters=3, optim_iters=2, cg_maxiter=200),
            decompose=nothing, seed=1)
        quiet = decompose(hutch; kind=:model, probes=5, seed=9, verbose=false)
        loud = decompose(hutch; kind=:model, probes=5, seed=9, verbose=true)
        @test quiet.V_firm == loud.V_firm
        @test quiet.V_worker == loud.V_worker
        @test quiet.V_cross == loud.V_cross
        fq = decompose(hutch; kind=:fitted, probes=5, seed=9, verbose=false)
        fl = decompose(hutch; kind=:fitted, probes=5, seed=9, verbose=true)
        @test fq.V_firm == fl.V_firm
        @test fq.V_total == fl.V_total
    end

    @testset "cov_block validation and batching" begin
        op = covariance(result; kind=:model)
        @test_throws ArgumentError cov_block(op)
        @test_throws ArgumentError cov_block(op; firms=[1], row_firms=[1])
        @test_throws ArgumentError cov_block(op; firms=[999])
        @test_throws ArgumentError cov_block(op; workers=["nope"])
        full = cov_block(op; firms=[1, 2], workers=[10, 11])
        small = cov_block(op; firms=[1, 2], workers=[10, 11], batch_size=1)
        @test full.matrix ≈ small.matrix atol=1e-12
        rect = cov_block(op; row_firms=[1, 2, 3], col_workers=[10])
        @test size(rect.matrix) == (3, 1)
    end
end
