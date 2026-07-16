function build_V_stats(
    f_rows::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    n_firms::Int,
    n_workers::Int,
)
    k = length(y)
    length(f_rows) == k || throw(ArgumentError("f_rows length does not match y."))
    length(w_cols) == k || throw(ArgumentError("w_cols length does not match y."))

    proj_f = zeros(Float64, n_firms)
    proj_w = zeros(Float64, n_workers)
    cnt_f = zeros(Float64, n_firms)
    cnt_w = zeros(Float64, n_workers)

    @inbounds for i in 1:k
        f = f_rows[i]
        w = w_cols[i]
        val = y[i]
        cnt_f[f] += 1.0
        cnt_w[w] += 1.0
        proj_f[f] += val
        proj_w[w] += val
    end

    A_obs = sparse(f_rows, w_cols, ones(Float64, k), n_firms, n_workers)
    VtV = [spdiagm(0 => cnt_f) A_obs; copy(transpose(A_obs)) spdiagm(0 => cnt_w)]
    return (
        projected_y = vcat(proj_f, proj_w),
        VtV = VtV,
        cnt_f = cnt_f,
        cnt_w = cnt_w,
        A_obs = A_obs,
        At_obs = copy(transpose(A_obs)),
        ydot = dot(y, y),
    )
end

function build_weighted_V_stats(
    f_rows::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    weights::Vector{Float64},
    n_firms::Int,
    n_workers::Int,
)
    k = length(y)
    length(f_rows) == k || throw(ArgumentError("f_rows length does not match y."))
    length(w_cols) == k || throw(ArgumentError("w_cols length does not match y."))
    length(weights) == k || throw(ArgumentError("weights length does not match y."))

    proj_f = zeros(Float64, n_firms)
    proj_w = zeros(Float64, n_workers)
    cnt_f = zeros(Float64, n_firms)
    cnt_w = zeros(Float64, n_workers)
    ydot = 0.0

    @inbounds for i in 1:k
        f = f_rows[i]
        w = w_cols[i]
        wi = weights[i]
        val = y[i]
        cnt_f[f] += wi
        cnt_w[w] += wi
        proj_f[f] += wi * val
        proj_w[w] += wi * val
        ydot += wi * val * val
    end

    A_obs = sparse(f_rows, w_cols, weights, n_firms, n_workers)
    VtV = [spdiagm(0 => cnt_f) A_obs; copy(transpose(A_obs)) spdiagm(0 => cnt_w)]
    return (
        projected_y = vcat(proj_f, proj_w),
        VtV = VtV,
        cnt_f = cnt_f,
        cnt_w = cnt_w,
        A_obs = A_obs,
        At_obs = copy(transpose(A_obs)),
        ydot = ydot,
    )
end

function build_match_weight_stats(
    f_rows::Vector{Int},
    w_cols::Vector{Int},
    y::Vector{Float64},
    T::Vector{Int},
    n_firms::Int,
    n_workers::Int,
    rho_eps::Float64,
)
    weights = effective_match_weights(T, rho_eps)
    stats = build_weighted_V_stats(f_rows, w_cols, y, weights, n_firms, n_workers)
    return merge(stats, (
        log_weight_sum = sum(log, weights),
        effective_weight_sum = sum(weights),
        mean_effective_weight = mean(weights),
        max_effective_weight = maximum(weights),
        effective_weight_over_T_sum = sum(weights ./ Float64.(T)),
    ))
end

function filter_max_degree(df::DataFrame, max_degree::Int, firm_id::Symbol, worker_id::Symbol)
    max_degree > 0 || throw(ArgumentError("max_degree must be positive."))
    firm_deg = combine(groupby(df, firm_id), nrow => :deg)
    worker_deg = combine(groupby(df, worker_id), nrow => :deg)
    keep_firms = Set(firm_deg[firm_deg.deg .<= max_degree, firm_id])
    keep_workers = Set(worker_deg[worker_deg.deg .<= max_degree, worker_id])
    return filter(row -> row[firm_id] in keep_firms && row[worker_id] in keep_workers, df)
end

function prior_adjacency(prior::AbstractGMRFPrior)
    prior isa NormalizedPrior && return prior.prior_adjacency
    prior isa UnnormalizedPrior && return prior.prior_adjacency
    prior isa SpectralPrior && return prior.prior_adjacency
    prior isa VarianceStablePrior && return :binary
    error("Unknown prior type $(typeof(prior)).")
end

function prepare_prior_scaling(A_prior::SparseMatrixCSC{Float64,Int}, prior::AbstractGMRFPrior)
    n_firms, n_workers = size(A_prior)
    At_prior = copy(transpose(A_prior))
    d_f = vec(sum(A_prior; dims=2))
    d_w = vec(sum(A_prior; dims=1))
    if any(d_f .<= 0) || any(d_w .<= 0)
        throw(ArgumentError("Zero-degree node detected; check IDs or filtering."))
    end

    if prior isa NormalizedPrior
        df_is = 1.0 ./ sqrt.(d_f)
        dw_is = 1.0 ./ sqrt.(d_w)
        diag_f = ones(Float64, n_firms)
        diag_w = ones(Float64, n_workers)
    elseif prior isa SpectralPrior
        s1 = leading_singular_value(A_prior, At_prior; seed=prior.seed)
        s1 > 0 || throw(ArgumentError("Cannot spectral-normalize an empty adjacency matrix."))
        df_is = fill(1.0 / sqrt(s1), n_firms)
        dw_is = fill(1.0 / sqrt(s1), n_workers)
        diag_f = ones(Float64, n_firms)
        diag_w = ones(Float64, n_workers)
    elseif prior isa UnnormalizedPrior
        df_is = ones(Float64, n_firms)
        dw_is = ones(Float64, n_workers)
        diag_f = Float64.(d_f)
        diag_w = Float64.(d_w)
    elseif prior isa VarianceStablePrior
        df_is = ones(Float64, n_firms)
        dw_is = ones(Float64, n_workers)
        diag_f = ones(Float64, n_firms)
        diag_w = ones(Float64, n_workers)
    else
        error("Unknown prior type $(typeof(prior)).")
    end

    return (
        At_prior = At_prior,
        d_f = Float64.(d_f),
        d_w = Float64.(d_w),
        df_is = df_is,
        dw_is = dw_is,
        diag_f = diag_f,
        diag_w = diag_w,
    )
end

function GMRFProblem(; kwargs...)
    fields = fieldnames(GMRFProblem)
    missing = [field for field in fields if !haskey(kwargs, field)]
    isempty(missing) ||
        throw(ArgumentError("Missing GMRFProblem fields: $(join(String.(missing), ", "))."))
    field_set = Set(fields)
    unknown = [field for field in keys(kwargs) if field ∉ field_set]
    isempty(unknown) ||
        throw(ArgumentError("Unknown GMRFProblem fields: $(join(String.(unknown), ", "))."))
    values = map(field -> kwargs[field], fields)
    return GMRFProblem(values...)
end

function problem_fields(problem::GMRFProblem)
    fields = fieldnames(GMRFProblem)
    return NamedTuple{fields}(map(field -> getfield(problem, field), fields))
end

function GMRFProblem(
    df::DataFrame;
    outcome::Symbol=:y,
    firm_id::Symbol=:firm_id,
    worker_id::Symbol=:worker_id,
    prior::AbstractGMRFPrior=NormalizedPrior(),
    weighting::Weighting=Weighting(),
    max_degree::Union{Nothing,Int}=nothing,
    standardize::Bool=true,
    on_missing::Symbol=:drop,
    verbose::Bool=false,
)
    on_missing in (:drop, :error) ||
        throw(ArgumentError("on_missing must be :drop or :error; got $(on_missing)."))

    for c in (outcome, firm_id, worker_id)
        hasproperty(df, c) || throw(ArgumentError("Missing required column $(c)."))
    end
    nrow(df) > 0 || throw(ArgumentError("Empty dataset."))

    prior isa VarianceStablePrior && weighting.observations != :raw &&
        throw(ArgumentError("VarianceStablePrior currently supports only Weighting(observations=:raw)."))

    d0 = max_degree === nothing ? df : filter_max_degree(df, max_degree, firm_id, worker_id)
    nrow(d0) > 0 || throw(ArgumentError("No rows remain after max_degree filtering."))

    y_raw_all = map(to_float_nan, d0[!, outcome])
    keep = map(isfinite, y_raw_all)
    if on_missing == :error && !all(keep)
        throw(ArgumentError("Outcome $(outcome) contains missing or non-finite values."))
    end
    d = d0[keep, :]
    y_raw = Float64.(y_raw_all[keep])
    personyear_rows = length(y_raw)
    personyear_rows > 0 || throw(ArgumentError("No usable observations after missing filter."))

    y_mean = standardize ? mean(y_raw) : 0.0
    y_std = standardize ? std(y_raw) : 1.0
    isfinite(y_std) && y_std > 0 || throw(ArgumentError("Outcome has zero or invalid standard deviation."))

    work = DataFrame(
        firm = d[!, firm_id],
        worker = d[!, worker_id],
        y = y_raw,
    )
    collapsed_all = combine(groupby(work, [:firm, :worker]),
        :y => length => :T,
        :y => mean => :y_mean,
        :y => edge_ssw => :ssw_raw,
    )

    local ids_f, ids_w, n_firms, n_workers, k
    local f_rows, w_cols, y, T_edge
    local base_stats, A_prior_base, decomp_f_rows, decomp_w_cols, decomp_y, decomp_T
    local within_ss, within_df, personyear_within_ss, rho_eps_likelihood
    local log_weight_sum, effective_weight_sum, effective_weight_over_T_sum
    local mean_effective_weight, max_effective_weight
    obs = weighting.observations

    if obs == :raw
        ids_f = collect(unique(work.firm))
        ids_w = collect(unique(work.worker))
        n_firms = length(ids_f)
        n_workers = length(ids_w)
        k = nrow(work)
        firm_to_index = Dict(id => i for (i, id) in enumerate(ids_f))
        worker_to_index = Dict(id => i for (i, id) in enumerate(ids_w))
        f_rows = Vector{Int}(undef, k)
        w_cols = Vector{Int}(undef, k)
        @inbounds for i in 1:k
            f_rows[i] = firm_to_index[work.firm[i]]
            w_cols[i] = worker_to_index[work.worker[i]]
        end
        y = (Float64.(work.y) .- y_mean) ./ y_std
        T_edge = ones(Int, k)
        base_stats = build_V_stats(f_rows, w_cols, y, n_firms, n_workers)
        A_prior_base = copy(base_stats.A_obs)

        decomp_f_rows = Vector{Int}(undef, nrow(collapsed_all))
        decomp_w_cols = Vector{Int}(undef, nrow(collapsed_all))
        @inbounds for i in 1:nrow(collapsed_all)
            decomp_f_rows[i] = firm_to_index[collapsed_all.firm[i]]
            decomp_w_cols[i] = worker_to_index[collapsed_all.worker[i]]
        end
        decomp_y = (Float64.(collapsed_all.y_mean) .- y_mean) ./ y_std
        decomp_T = Int.(collapsed_all.T)
        personyear_within_ss = sum(Float64.(collapsed_all.ssw_raw)) / y_std^2
        within_ss = 0.0
        within_df = 0
        rho_eps_likelihood = nothing
        log_weight_sum = 0.0
        effective_weight_sum = Float64(k)
        effective_weight_over_T_sum = Float64(k)
        mean_effective_weight = 1.0
        max_effective_weight = 1.0
    else
        ids_f = collect(unique(collapsed_all.firm))
        ids_w = collect(unique(collapsed_all.worker))
        n_firms = length(ids_f)
        n_workers = length(ids_w)
        k = nrow(collapsed_all)
        firm_to_index = Dict(id => i for (i, id) in enumerate(ids_f))
        worker_to_index = Dict(id => i for (i, id) in enumerate(ids_w))
        f_rows = Vector{Int}(undef, k)
        w_cols = Vector{Int}(undef, k)
        @inbounds for i in 1:k
            f_rows[i] = firm_to_index[collapsed_all.firm[i]]
            w_cols[i] = worker_to_index[collapsed_all.worker[i]]
        end
        y = (Float64.(collapsed_all.y_mean) .- y_mean) ./ y_std
        T_edge = Int.(collapsed_all.T)
        A_prior_base = sparse(f_rows, w_cols, Float64.(T_edge), n_firms, n_workers)
        decomp_f_rows = copy(f_rows)
        decomp_w_cols = copy(w_cols)
        decomp_y = copy(y)
        decomp_T = copy(T_edge)
        personyear_within_ss = sum(Float64.(collapsed_all.ssw_raw)) / y_std^2

        if obs == :edge
            base_stats = build_weighted_V_stats(f_rows, w_cols, y, ones(Float64, k), n_firms, n_workers)
            within_ss = 0.0
            within_df = 0
            rho_eps_likelihood = nothing
            log_weight_sum = 0.0
            effective_weight_sum = Float64(k)
            effective_weight_over_T_sum = Float64(k)
            mean_effective_weight = 1.0
            max_effective_weight = 1.0
        else
            rho0 = weighting.rho_eps == :estimate ? 0.5 : Float64(weighting.rho_eps)
            base_stats = build_match_weight_stats(f_rows, w_cols, y, T_edge, n_firms, n_workers, rho0)
            within_ss = personyear_within_ss
            within_df = personyear_rows - k
            rho_eps_likelihood = rho0
            log_weight_sum = base_stats.log_weight_sum
            effective_weight_sum = base_stats.effective_weight_sum
            effective_weight_over_T_sum = base_stats.effective_weight_over_T_sum
            mean_effective_weight = base_stats.mean_effective_weight
            max_effective_weight = base_stats.max_effective_weight
        end
    end

    padj = prior_adjacency(prior)
    A_prior = copy(A_prior_base)
    padj == :binary && (A_prior.nzval .= 1.0)
    pscale = prepare_prior_scaling(A_prior, prior)
    vs_metadata = NamedTuple()
    if prior isa VarianceStablePrior
        resolved = prepare_vs_feasibility(prior, A_prior)
        prior = resolved.prior
        vs_metadata = resolved.metadata
    end

    unique_edges = nnz(sparse(f_rows, w_cols, ones(Float64, length(f_rows)), n_firms, n_workers))
    duplicate_rows = personyear_rows - unique_edges
    mean_edge_count = unique_edges > 0 ? Float64(personyear_rows) / Float64(unique_edges) : NaN
    max_edge_count = isempty(T_edge) ? NaN : Float64(maximum(T_edge))

    metadata = merge((
        outcome = outcome,
        firm_id = firm_id,
        worker_id = worker_id,
        max_degree = max_degree,
        prior_adjacency = padj,
        unique_edges = unique_edges,
        duplicate_rows = duplicate_rows,
        mean_edge_count = mean_edge_count,
        max_edge_count = max_edge_count,
        total_prior_weight = sum(A_prior.nzval),
        max_prior_degree_f = maximum(pscale.d_f),
        max_prior_degree_w = maximum(pscale.d_w),
    ), vs_metadata)

    return GMRFProblem(;
        y = y,
        ydot = Float64(base_stats.ydot),
        projected_y = base_stats.projected_y,
        VtV = base_stats.VtV,
        A_prior = A_prior,
        At_prior = pscale.At_prior,
        A_obs = base_stats.A_obs,
        At_obs = base_stats.At_obs,
        d_f = pscale.d_f,
        d_w = pscale.d_w,
        cnt_f = base_stats.cnt_f,
        cnt_w = base_stats.cnt_w,
        df_is = pscale.df_is,
        dw_is = pscale.dw_is,
        diag_f = pscale.diag_f,
        diag_w = pscale.diag_w,
        firm_ids = ids_f,
        worker_ids = ids_w,
        firm_to_index = firm_to_index,
        worker_to_index = worker_to_index,
        base_f_rows = f_rows,
        base_w_cols = w_cols,
        base_y = y,
        base_T = T_edge,
        decomp_f_rows = decomp_f_rows,
        decomp_w_cols = decomp_w_cols,
        decomp_y = decomp_y,
        decomp_T = decomp_T,
        N_firms = n_firms,
        N_workers = n_workers,
        K = k,
        personyear_rows = personyear_rows,
        y_mean = Float64(y_mean),
        y_std = Float64(y_std),
        standardize = standardize,
        prior = prior,
        model = to_model(prior, A_prior),
        weighting = weighting,
        rho_eps_likelihood = rho_eps_likelihood,
        within_ss = Float64(within_ss),
        within_df = Int(within_df),
        personyear_within_ss = Float64(personyear_within_ss),
        log_weight_sum = Float64(log_weight_sum),
        effective_weight_sum = Float64(effective_weight_sum),
        effective_weight_over_T_sum = Float64(effective_weight_over_T_sum),
        mean_effective_weight = Float64(mean_effective_weight),
        max_effective_weight = Float64(max_effective_weight),
        metadata = metadata,
    )
end

function with_observation_stats(problem::GMRFProblem, stats, rho_eps::Union{Nothing,Float64})
    return GMRFProblem(;
        merge(problem_fields(problem), (
            ydot = Float64(stats.ydot),
            projected_y = stats.projected_y,
            VtV = stats.VtV,
            A_obs = stats.A_obs,
            At_obs = stats.At_obs,
            cnt_f = stats.cnt_f,
            cnt_w = stats.cnt_w,
            rho_eps_likelihood = rho_eps,
            log_weight_sum = Float64(get(stats, :log_weight_sum, problem.log_weight_sum)),
            effective_weight_sum = Float64(get(stats, :effective_weight_sum, problem.effective_weight_sum)),
            effective_weight_over_T_sum = Float64(get(
                stats,
                :effective_weight_over_T_sum,
                problem.effective_weight_over_T_sum,
            )),
            mean_effective_weight = Float64(get(stats, :mean_effective_weight, problem.mean_effective_weight)),
            max_effective_weight = Float64(get(stats, :max_effective_weight, problem.max_effective_weight)),
        ))...
    )
end
