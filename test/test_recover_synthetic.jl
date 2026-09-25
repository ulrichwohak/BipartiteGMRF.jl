using Statistics: mean, std

@testset "Monte Carlo synthetic recovery" begin
    # Rho is weakly identified in an individual panel: a valid finite-sample
    # MLE need not lie within 0.10 of its generating value. Seed 203 happened
    # to pass on newer Julia, but the RNG stream produces a different panel
    # on Julia 1.10. Follow the aggregate-recovery framing discussed in #58:
    # use the complete contiguous range, retain the per-panel scale checks,
    # and check average rho recovery with the original numerical tolerance.
    # This is a bounded regression check, not a coverage or consistency claim.
    seeds = 201:208
    truth = (rho=0.45, sigma_a=0.8, sigma_z=0.5, sigma_epsilon=0.25)
    estimates = map(seeds) do seed
        data, _ = simulate_gmrf_panel(seed; truth=truth)
        ss = suffstats(BipartiteNormalizedModel, data.f, data.w, data.y;
            standardize=false)
        estimate = fit_mle(
            BipartiteNormalizedModel, ss;
            solver=ExactCholesky(optim_iters=160, polish=true),
            seed=seed,
        )
        truth_params = [atanh(truth.rho / BipartiteGMRF.rho_limit(estimate.model)),
            log(truth.sigma_a), log(truth.sigma_z), log(truth.sigma_epsilon)]
        truth_obs = BipartiteGMRF.objective_stats(estimate.model, ss, truth_params)
        truth_nll = BipartiteGMRF.nll_exact_value(estimate.model, ss, truth_params, truth_obs)
        @info "Synthetic recovery" seed rho=estimate.rho sigma_a=estimate.sigma_a sigma_z=estimate.sigma_z sigma_epsilon=estimate.sigma_epsilon nll=estimate.nll truth_nll converged=estimate.converged
        @testset "seed $seed" begin
            @test estimate.converged
            @test all(isfinite, (estimate.rho, estimate.sigma_a, estimate.sigma_z,
                                estimate.sigma_epsilon, estimate.nll, truth_nll))
            @test max(estimate.nll, truth_nll) < BipartiteGMRF.BIG_NLL
            @test abs(estimate.rho) < BipartiteGMRF.rho_limit(estimate.model)
            # Sampling variation may move the MLE away from truth, but the
            # fitted objective should improve on that feasible candidate.
            @test estimate.nll <= truth_nll + 1e-6
            @test abs(log(estimate.sigma_a) - log(truth.sigma_a)) < 0.15
            @test abs(log(estimate.sigma_z) - log(truth.sigma_z)) < 0.15
            @test abs(log(estimate.sigma_epsilon) - log(truth.sigma_epsilon)) < 0.10
        end
        estimate
    end

    rhos = [estimate.rho for estimate in estimates]
    mean_rho = mean(rhos)
    sd_rho = std(rhos)
    mcse_rho = sd_rho / sqrt(length(rhos))
    @info "Aggregate rho recovery" seeds truth_rho=truth.rho mean_rho sd_rho mcse_rho
    @test abs(mean_rho - truth.rho) < 0.10
end
