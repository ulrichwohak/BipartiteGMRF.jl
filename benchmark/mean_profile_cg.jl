# Opt-in numerical experiment, NOT a supported mean solver. Run with:
# OPENBLAS_NUM_THREADS=1 julia --project=. benchmark/mean_profile_cg.jl
# One small, well-conditioned fixture with exact inner network solves cannot
# establish reliability on other graphs or with iterative inner solves.
include(joinpath(@__DIR__, "mean_profiling.jl"))

function coefficient_cg_probe(apply_G, c, diagonal; rtol=1e-12)
    b = zeros(length(c)); r = copy(c); z = r ./ diagonal; d = copy(z)
    norm_c = norm(c)
    norm_c == 0 && return 0.0, b, 0
    rz = dot(r, z)
    for iteration in 1:(2length(c))
        Gd = apply_G(d)
        denominator = dot(d, Gd)
        isfinite(denominator) && denominator > 0 || error("Nonpositive CG curvature")
        alpha = rz / denominator
        b .+= alpha .* d
        r .-= alpha .* Gd
        if norm(r) <= rtol * norm_c
            # Evaluate the achieved likelihood, not just c'b for approximate b.
            correction = 2dot(c, b) - dot(b, apply_G(b))
            return correction, b, iteration
        end
        z .= r ./ diagonal
        rz_next = dot(r, z)
        d .= z .+ (rz_next / rz) .* d
        rz = rz_next
    end
    error("Coefficient CG did not converge")
end

function evaluate_cg_probe(rho, eta)
    data = fixture(1000, 256, 1, "sparse")
    stats = suffstats(BipartiteNormalizedModel, data.f, data.w, data.y;
        X=data.X, weighting=Weighting(observations=:raw), standardize=false,
        error_eta=:estimate, edge_index=data.edge_index)
    model = BipartiteNormalizedModel(stats.A_prior)
    theta = [atanh(rho / BG.rho_limit(model)), log(0.8), log(0.6), log(0.7),
             BG.eta_to_unconstrained(eta)]
    obs = BG.objective_stats(model, stats, theta)
    cache = BG.make_nll_cache(ExactCholesky(), model, stats)
    BG.nll_exact_value(model, stats, theta, obs, cache)
    ms = obs.mean_stats; lambda = 1 / 0.7^2
    solve_M(v) = GM.workspace_solve(cache.ws_M, v)
    apply_G(v) = lambda .* (ms.XtX * v) -
        lambda^2 .* (ms.VtX' * solve_M(ms.VtX * v))
    workspace = BG.MeanProfileWorkspace(ms)
    direct() = BG.mean_profile_correction(ms, lambda, obs.design.projected_y, solve_M, workspace)
    function iterative()
        c = lambda .* ms.Xty - lambda^2 .* (ms.VtX' * solve_M(obs.design.projected_y))
        return coefficient_cg_probe(apply_G, c, lambda .* diag(ms.XtX))
    end
    exact_correction, exact_beta = direct()
    cg_correction, cg_beta, iterations = iterative()
    @assert isapprox(cg_beta, exact_beta; rtol=1e-9, atol=1e-10)
    @assert isapprox(cg_correction, exact_correction; rtol=1e-10, atol=1e-10)
    println((; rho, eta, iterations, solves=iterations+2,
        coefficient_relative_error=norm(cg_beta-exact_beta)/norm(exact_beta),
        correction_error=cg_correction-exact_correction,
        direct=measured(direct), coefficient_cg=measured(iterative)))
    return exact_correction, cg_correction
end

function main_cg_probe()
    println((; experiment="coefficient CG with exact network solves",
        julia=VERSION, source=pathof(BG), git=git_provenance(),
        gmrf_version=Base.pkgversion(GM), seed=SEED, nodes=2000, controls=256,
        julia_threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads()))
    evaluate_cg_probe(0.2, 0.2)
    h = 1e-5
    rp = evaluate_cg_probe(0.2+h, 0.2); rm = evaluate_cg_probe(0.2-h, 0.2)
    ep = evaluate_cg_probe(0.2, 0.2+h); em = evaluate_cg_probe(0.2, 0.2-h)
    rho_error = ((rp[2]-rm[2])-(rp[1]-rm[1]))/(2h)
    eta_error = ((ep[2]-em[2])-(ep[1]-em[1]))/(2h)
    @assert abs(rho_error) < 1e-6 && abs(eta_error) < 1e-6
    # The profiled NLL subtracts half the achieved correction.
    println((; rho_correction_fd_error=rho_error, eta_correction_fd_error=eta_error,
        rho_nll_fd_error=-rho_error / 2, eta_nll_fd_error=-eta_error / 2))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cg_probe()
end
