# Real-data test of EMIWBlocks (issue #112 remedy E) on a connected subsample
# of the choo-siow-calvo firm×CEO edgelist (lnR outcome, exported by the
# project's /tmp/csc_export_subsample.jl). Compares the iid VS baseline with
# the integrated free-block fit on the same graph.
using BipartiteGMRF, DelimitedFiles, Printf

raw, _ = readdlm("/tmp/csc_subsample.csv", ','; header = true)
f = Int.(raw[:, 1])
w = Int.(raw[:, 2])
y = Float64.(raw[:, 3])
println("edges=", length(y), " firms=", maximum(f), " workers=", maximum(w))

t0 = time()
res0 = fit_mle(BipartiteVarianceStableModel, f, w, y;
    weighting = Weighting(observations = :raw), standardize = true,
    rho_limit = :auto, solver = ExactCholesky())
@printf("\n--- iid baseline (ExactCholesky, %.1fs) ---\n", time() - t0)
@printf("rho=%.4f  sigma_a=%.4f  sigma_z=%.4f  sigma_eps=%.4f  converged=%s\n",
        res0.rho, res0.sigma_a, res0.sigma_z, res0.sigma_epsilon, res0.converged)

t0 = time()
res = fit_mle(BipartiteVarianceStableModel, f, w, y;
    weighting = Weighting(observations = :raw), standardize = true,
    rho_limit = :auto, error_blocks = :free, firm_group = f,
    solver = EMIWBlocks(max_iter = 400, ftol = 1e-8))
@printf("\n--- integrated free blocks (EMIWBlocks, %.1fs, %d iters) ---\n",
        time() - t0, res.iterations)
@printf("rho=%.4f  sigma_a=%.4f  sigma_z=%.4f  sigma_eps(=sqrt omega_bar)=%.4f  converged=%s\n",
        res.rho, res.sigma_a, res.sigma_z, res.sigma_epsilon, res.converged)
@printf("phi=%.4f  r=%.4f  delta=%.1f%s\n",
        res.metadata.error_scale_phi, res.metadata.error_corr_r,
        res.metadata.t_dof_delta,
        res.metadata.t_dof_delta >= 9999 ? " (cap: no detectable block heterogeneity)" : "")
ub = res.metadata.u_bar
@printf("u_bar: min=%.3f  median=%.3f  max=%.3f\n",
        minimum(ub), sort(ub)[(length(ub) + 1) ÷ 2], maximum(ub))
