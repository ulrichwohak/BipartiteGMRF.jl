# ═══════════════════════════════════════════════════════════════════════════
# Per-firm error blocks, integrated-likelihood version (issue #112, remedy E):
# Ω_i are nuisance realizations drawn iid from an inverse-Wishart population
# law and integrated out — never estimated. Scalar scale-mixture form:
# u_i ~ Gamma(δ/2, δ/2) per firm, ε_i | u_i ~ N(0, Ψ_{m_i}/u_i) with
# Ψ_m = φ·[(1−r)I + r𝟙𝟙']. Variational EM under q(α)·∏q(u_i); every update is
# closed-form (r and δ by 1-D searches) and reuses the dense-verified block
# machinery (assembly, workspaces, posterior moments, θ score).
# ═══════════════════════════════════════════════════════════════════════════

# ─── Equicorrelation R(r) = (1−r)I + r𝟙𝟙' closed forms ─────────────────────

_equicorr_logdet(m::Int, r::Float64) = (m - 1) * log1p(-r) + log1p((m - 1) * r)

# tr(R(r)⁻¹ S) via R⁻¹ = [I − r/(1+(m−1)r)·𝟙𝟙']/(1−r)
function _equicorr_trinv(S::Matrix{Float64}, r::Float64)
    m = size(S, 1)
    return (tr(S) - (r / (1 + (m - 1) * r)) * sum(S)) / (1 - r)
end

_equicorr(m::Int, r::Float64) =
    Matrix{Float64}((1 - r) * I, m, m) .+ r

# Lower bound of the r domain: R(r) is PD for r ∈ (−1/(m−1), 1) at the largest
# block; singleton-only data leave r unidentified (R ≡ [1]) and it stays at 0.
_equicorr_rlo(mmax::Int) = mmax <= 1 ? -1.0 + 1e-8 : -1.0 / (mmax - 1) + 1e-8

# Upper cap for the estimated t dof δ: past this the mixing distribution is a
# point mass to numerical precision and the block errors are exactly Gaussian.
const DELTA_CAP = 1e4

# ─── ELBO pieces ────────────────────────────────────────────────────────────

# logΓ(x+h) − logΓ(x), stable for large x. The naive difference cancels
# catastrophically once x ≳ 1e6; the Stirling form is exact to O(h³/x²) there.
function _loggamma_diff(x::Float64, h::Float64)
    x < 1e6 && return loggamma(x + h) - loggamma(x)
    return h * log(x) + h * (h - 1) / (2x)
end

# E_q[log p(u)] + H[q(u)] = −KL(q ‖ p) for one block: q = Gamma(a, b)
# (shape/rate), prior Gamma(δ/2, δ/2), so a = (δ+m)/2, b = (δ+s)/2. Written in
# the cancellation-free form (log1p + stable loggamma difference) so it stays
# accurate as δ → ∞, where the term vanishes.
function _elbo_u_terms(a::Float64, b::Float64, δ::Float64)
    h = a - δ / 2       # m/2
    s = 2b - δ          # E_q[r'Ψ⁻¹r] ≥ 0
    return -h * digamma(a) + _loggamma_diff(δ / 2, h) -
           (δ / 2) * log1p(s / δ) + a * (s / 2) / b
end

"""
    emiw_init_status(solver, mmax)

`init` availability on the EMIWBlocks path (see [`init_status`](@ref) for the
`optimize_problem` counterpart). A solver-fixed `r` or `delta` reports the
pinned value. `r` is estimated only when the largest block holds more than one
observation; with all-singleton blocks it is `:absent`, and `optimize_emiw`
rejects `init.r` with a message that says why.
"""
function emiw_init_status(solver::EMIWBlocks, mmax::Int)
    return (
        rho = :free,
        rho_eps = :absent,
        eta = :absent,
        omega = :absent,
        phi = :free,
        r = solver.r isa Float64 ? solver.r : (mmax > 1 ? :free : :absent),
        delta = solver.delta isa Float64 ? solver.delta : :free,
    )
end

"""
    optimize_emiw(model, stats, solver; verbose, init) -> NamedTuple

Run the [`EMIWBlocks`](@ref) variational EM to convergence. Latents are the
GMRF field `α` and one scalar mixing weight `u_i` per firm block; the
hyperparameters `(ρ, σ_a, σ_z, φ, r, δ)` maximize the ELBO (a lower bound on
the integrated log-likelihood with the `Ω_i` marginalized out). Each iteration:

1. `q(α) = N(μ, P⁻¹)` with `P = K + (1/φ)Σ_i ū_i A_i'R_i⁻¹A_i` (exact
   coordinate maximizer of the ELBO given `q(u)`).
2. `q(u_i) = Gamma((δ+m_i)/2, (δ+s_i)/2)` with
   `s_i = E_q[r_i'Ψ_i⁻¹r_i]` read off `S_{ε,i} = r̂_ir̂_i' + A_iP⁻¹A_i'`.
3. One safeguarded score step on `(ρ, σ_a, σ_z)`; the θ-dependent ELBO part is
   `½logdet K − ½tr(K S_α)`, whose gradient is the Fisher-identity form.
4. `φ` closed-form given `r`; `r` by 1-D profile search; `δ` by 1-D search on
   its exact E-step objective (both optional per solver settings).

Returns `optimize_problem`-shaped fields plus `(phi, r, delta, omega_bar,
elbo_trace, u_bar)`. `omega_bar = φδ/(δ−2)` is the mean error variance
(the `sigma_epsilon²` analogue).

`init` is the warm-start NamedTuple documented on [`fit_mle`](@ref), already
rescaled to standardized units. Note what each field can and cannot do here:
`rho`, `sigma_a`, `sigma_z` and `phi` are genuine starting values, but `r` and
`δ` are re-estimated by *bracketed* searches over their whole domain
(`optimize(negprof, r_lo, 1-1e-8)` and `optimize(negdof, log(2+1e-6),
log(DELTA_CAP))`), which take no starting point. `init.r` therefore seeds only
the first E-step, and `init.delta` is very nearly a no-op — δ is re-minimized
on the first pass of the inner loop before it influences anything but one
`Ωeff` build.
"""
function optimize_emiw(
    model::BipartiteVarianceStableModel,
    stats::BipartiteGMRFStats,
    solver::EMIWBlocks;
    verbose::Bool=false,
    init::Union{Nothing,NamedTuple}=nothing,
)
    fb = stats.error_blocks
    fb !== nothing || throw(ArgumentError("optimize_emiw requires error_blocks=:iw statistics."))
    nf = stats.N_firms
    nw = stats.N_workers
    limit = rho_limit(model)
    ew = make_em_blocks_workspace(model, fb, nf, nw)
    rows = _block_rows(fb)
    B = length(rows)
    ms = fb.sizes
    Ktot = sum(ms)
    n = nf + nw
    mmax = maximum(ms)
    r_lo = _equicorr_rlo(mmax)

    # Validate what the caller wrote, so the message quotes their numbers, then
    # move the scale parameters onto the standardized scale the EM works in.
    # r's own absence has a reason the generic hint cannot state.
    if mmax == 1 && !(solver.r isa Float64) && init_field(init, :r) !== nothing
        throw(ArgumentError(
            "init.r was given, but r is not estimated when every error block holds " *
            "a single observation (largest block size is 1) — it would become a " *
            "permanent value rather than a starting one. Pass EMIWBlocks(r=...) to " *
            "fix it deliberately."))
    end
    validate_init(init, emiw_init_status(solver, mmax); rho_limit=limit)
    init = rescale_init_sigmas(init, stats.y_std)

    # θ: field-by-field over the same heuristic optimize_problem uses. Domains
    # were checked above, so _em_blocks_ψ_from_θ's bare atanh/log are safe.
    rho0 = init_field(init, :rho)
    sa0 = init_field(init, :sigma_a)
    sz0 = init_field(init, :sigma_z)
    ψ = _em_blocks_ψ_from_θ(
        rho0 === nothing ? default_rho_start(limit) : Float64(rho0),
        sa0 === nothing ? 0.7 : Float64(sa0),
        sz0 === nothing ? 0.04 : Float64(sz0),
        limit,
    )

    δ = if solver.delta isa Float64
        solver.delta
    else
        d0 = init_field(init, :delta)
        d0 === nothing ? 10.0 : Float64(d0)
    end

    # EMIW has no sigma_epsilon coordinate: the error scale is φ, tied to it by
    # omega_bar = φδ/(δ−2) = σ_ε². Prefer an explicit φ; fall back to σ_ε.
    φ0 = init_field(init, :phi)
    se0 = init_field(init, :sigma_epsilon)
    φ0 === nothing || se0 === nothing || @warn(
        "init.sigma_epsilon is ignored: init.phi sets the error scale directly " *
        "(they are tied by omega_bar = phi*delta/(delta-2) = sigma_epsilon^2)."
    )
    φ = if φ0 !== nothing
        Float64(φ0)          # positivity already checked by validate_init
    elseif se0 !== nothing
        Float64(se0)^2 * (δ - 2.0) / δ
    else
        0.5
    end

    r = if solver.r isa Float64
        r_lo <= solver.r < 1.0 || throw(ArgumentError(
            "fixed r = $(solver.r) is outside the PD domain [$(r_lo), 1) for the " *
            "largest block size $mmax."))
        solver.r
    else
        r0 = init_field(init, :r)
        if r0 === nothing
            0.0
        else
            r_lo <= Float64(r0) < 1.0 || throw(ArgumentError(
                "init.r = $(r0) is outside the PD domain [$(r_lo), 1) for the " *
                "largest block size $mmax."))
            Float64(r0)
        end
    end
    ubar = ones(Float64, B)
    elogu = zeros(Float64, B)
    ashape = zeros(Float64, B)   # q(u_i) shape/rate, set each iteration
    brate = zeros(Float64, B)

    V = fb.V
    Seps = Vector{Matrix{Float64}}(undef, B)
    μ = zeros(Float64, n)
    elbo_trace = Float64[]
    elbo_curr = -Inf
    converged = false
    iterations = 0

    while iterations < solver.max_iter
        rho, sa, sz = _em_blocks_θ_from_ψ(ψ, limit)

        # 1. q(α): P = K + A'Ω⁻¹A at effective blocks Ω_i = (φ/ū_i)·R_i.
        Ωeff = [(_equicorr(ms[i], r) .* (φ / ubar[i])) for i in 1:B]
        P, b = assemble_block_precision(fb, Ωeff, nf, nw)
        Q = model_precision(model, rho, sa, sz)
        GaussianMarkovRandomFields.update_precision_values!(ew.ws_Q, _align_to_ws(Q, nothing, ew.ws_Q))
        GaussianMarkovRandomFields.ensure_numeric!(ew.ws_Q)
        GaussianMarkovRandomFields.update_precision_values!(ew.ws_M, _align_to_ws(Q, P, ew.ws_M))
        GaussianMarkovRandomFields.ensure_numeric!(ew.ws_M)
        μ = GaussianMarkovRandomFields.workspace_solve(ew.ws_M, b)

        # Posterior residual second moments S_{ε,i} = r̂r̂' + A_i P⁻¹ A_i'
        # (dense-verified in the block EM; q(α)-expectations, θ-free).
        resid = fb.y - V * μ
        for i in 1:B
            rr = rows[i]
            ri = resid[rr]
            S = ri * ri' + _aptpa(ew.ws_M, V[rr, :])
            Seps[i] = 0.5 .* (S .+ S')
        end

        # 2. q(u_i) = Gamma(a_i, b_i) — exact mean-field update — alternated
        # with the δ update to their joint fixed point.
        svals = [_equicorr_trinv(Seps[i], r) / φ for i in 1:B]
        for _ in 1:25
            for i in 1:B
                ashape[i] = (δ + ms[i]) / 2
                brate[i] = (δ + svals[i]) / 2
                ubar[i] = ashape[i] / brate[i]
                elogu[i] = digamma(ashape[i]) - log(brate[i])
            end
            solver.delta == :estimate || break
            sum_e = sum(elogu) - sum(ubar)
            negdof = lδ -> begin
                d = exp(lδ)
                return -(B * ((d / 2) * log(d / 2) - loggamma(d / 2)) + (d / 2) * sum_e)
            end
            res = optimize(negdof, log(2.0 + 1e-6), log(DELTA_CAP))
            δ_new = min(exp(minimizer(res)), DELTA_CAP)
            settled = abs(δ_new - δ) <= 1e-8 * max(1.0, δ)
            δ = δ_new
            settled && break
        end

        # 3. θ step: ELBO's θ-part is F(θ) = ½logdet K − ½tr(K S_α); its
        # gradient is the Fisher-identity form _em_blocks_score computes.
        for _ in 1:4
            gψ = _em_blocks_score(model, ew, μ, ψ, limit)
            ψn = _emiw_theta_step(model, ew, μ, ψ, gψ, limit)
            moved = ψn != ψ
            ψ = ψn
            rho, sa, sz = _em_blocks_θ_from_ψ(ψ, limit)
            Q = model_precision(model, rho, sa, sz)
            GaussianMarkovRandomFields.update_precision_values!(ew.ws_Q, _align_to_ws(Q, nothing, ew.ws_Q))
            GaussianMarkovRandomFields.ensure_numeric!(ew.ws_Q)
            moved || break
        end

        # 4. (φ, r): maximize E_q[log p(y|α,u)]. Given r, φ is closed-form;
        # r by 1-D profile search over its PD domain.
        if solver.r == :estimate && mmax > 1
            negprof = rr_ -> begin
                tsum = sum(ubar[i] * _equicorr_trinv(Seps[i], rr_) for i in 1:B)
                ldsum = sum(_equicorr_logdet(ms[i], rr_) for i in 1:B)
                return 0.5 * ldsum + (Ktot / 2) * log(max(tsum / Ktot, 1e-300))
            end
            res = optimize(negprof, r_lo, 1.0 - 1e-8)
            r = minimizer(res)
        end
        φ = sum(ubar[i] * _equicorr_trinv(Seps[i], r) for i in 1:B) / Ktot

        # ELBO at the end-of-iteration state (θ, φ, r, δ new; q fixed).
        ldK = -GaussianMarkovRandomFields.logdet_cov(ew.ws_Q)
        ldP = -GaussianMarkovRandomFields.logdet_cov(ew.ws_M)
        trKP = GaussianMarkovRandomFields.selinv_dot(ew.ws_M, Q)
        e1 = -(Ktot / 2) * log(2π) + 0.5 * sum(ms[i] * elogu[i] for i in 1:B) -
             0.5 * sum(ms[i] * log(φ) + _equicorr_logdet(ms[i], r) for i in 1:B) -
             (0.5 / φ) * sum(ubar[i] * _equicorr_trinv(Seps[i], r) for i in 1:B)
        e2 = -(n / 2) * log(2π) + 0.5 * ldK - 0.5 * (dot(μ, Q * μ) + trKP)
        h1 = (n / 2) * (1 + log(2π)) - 0.5 * ldP
        eu = sum(_elbo_u_terms(ashape[i], brate[i], δ) for i in 1:B)
        elbo_new = e1 + e2 + h1 + eu

        push!(elbo_trace, elbo_new)
        iterations += 1
        Δ = elbo_new - elbo_curr
        scale = max(1.0, abs(elbo_new))
        if isfinite(elbo_curr) && abs(Δ) < solver.ftol * scale
            elbo_curr = elbo_new
            converged = true
            break
        end
        isfinite(elbo_curr) && Δ < -1e-8 * scale &&
            @warn "non-monotone ELBO step (decreased by $(-Δ))."
        elbo_curr = elbo_new
        verbose && @info "EMIWBlocks iter $iterations: elbo=$(elbo_curr) φ=$(φ) r=$(r) δ=$(δ)"
    end

    rho, sa, sz = _em_blocks_θ_from_ψ(ψ, limit)
    converged || @warn(
        "EMIWBlocks did not converge in $(solver.max_iter) iterations (relative ELBO " *
        "change still above ftol = $(solver.ftol)); treat the estimate as non-settled."
    )
    abs(rho) < 1e-2 && @warn(
        "Integrated block fit landed at small |ρ| = $(abs(rho)): σ_a is identified " *
        "only off the cross-firm network — treat σ_a (and ρ) as weakly identified (issue #112 §4)."
    )
    omega_bar = φ * δ / (δ - 2)
    return (
        rho = rho,
        sigma_a = sa * stats.y_std,
        sigma_z = sz * stats.y_std,
        sigma_epsilon = sqrt(max(omega_bar, 0.0)) * stats.y_std,
        rho_eps = nothing,
        beta = nothing,
        nll = -elbo_curr,
        converged = converged,
        iterations = iterations,
        obj_evals = iterations,
        optimization_time = 0.0,
        theta_unconstrained = ψ,
        model = model,
        stats = stats,
        phi = φ * stats.y_std^2,
        r = r,
        delta = δ,
        omega_bar = omega_bar * stats.y_std^2,
        u_bar = copy(ubar),
        elbo_trace = elbo_trace,
    )
end

# One safeguarded score step on the θ-part of the ELBO with q fixed:
# maximize F(θ) = ½logdet K(θ) − ½(μ'K(θ)μ + tr(K(θ)P⁻¹)) by Armijo
# backtracking along the negative gradient. P (ws_M) is a fixed variational
# parameter here, so each candidate costs one numeric ws_Q update + one
# selinv_dot on the already-factored ws_M — no refactorization.
function _emiw_theta_step(
    model::BipartiteVarianceStableModel,
    ew::EMBlocksWorkspace,
    μ::Vector{Float64},
    ψ0::Vector{Float64},
    gψ::Vector{Float64},
    limit::Float64,
)
    negF = ψ -> begin
        try
            rho, sa, sz = _em_blocks_θ_from_ψ(ψ, limit)
            Q = model_precision(model, rho, sa, sz)
            GaussianMarkovRandomFields.update_precision_values!(ew.ws_Q, _align_to_ws(Q, nothing, ew.ws_Q))
            GaussianMarkovRandomFields.ensure_numeric!(ew.ws_Q)
            ldK = -GaussianMarkovRandomFields.logdet_cov(ew.ws_Q)
            return -0.5 * ldK + 0.5 * (dot(μ, Q * μ) +
                                       GaussianMarkovRandomFields.selinv_dot(ew.ws_M, Q))
        catch e
            e isa InterruptException && rethrow()
            return Inf
        end
    end
    g2 = dot(gψ, gψ)
    g2 > 0 || return ψ0
    f0 = negF(ψ0)
    step = 1.0
    for _ in 1:80
        ψn = ψ0 .- step .* gψ
        fn = negF(ψn)
        isfinite(fn) && fn <= f0 - 1e-4 * step * g2 && return ψn
        step *= 0.5
    end
    return ψ0
end