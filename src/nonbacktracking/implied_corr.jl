struct ImpliedCorrEdge{F,W}
    firm_id::F
    worker_id::W
    firm_index::Int
    worker_index::Int
    correlation::Float64
    stratum::Symbol
    distance_to_core_firm::Int
    distance_to_core_worker::Int
    nb_score::Float64
    core_component::Int
    component_lambda::Float64
    hotspot::Bool
    inclusion_probability::Float64
end

struct ImpliedCorrNode{I}
    id::I
    index::Int
    b_inverse_diagonal::Float64
    variance_inflation::Float64
    in_two_core::Bool
    distance_to_core::Int
    nb_score::Float64
    core_component::Int
    hotspot::Bool
end

"""
Exact selected-edge implied correlations under a variance-stable prior.

`edges` contains selected unique firm-worker edges with cyclicity strata and
hotspot flags. `firms` and `workers` contain every solved endpoint and its
variance inflation `(1-rho^2) * diag(inv(B))`. The edge-average functional is
stored in `functional` and marked as either `:exact` or `:stratified`.
"""
struct ImpliedCorrResult{F,W,P<:GMRFProblem}
    rho::Float64
    edges::Vector{ImpliedCorrEdge{F,W}}
    firms::Vector{ImpliedCorrNode{F}}
    workers::Vector{ImpliedCorrNode{W}}
    spectrum::NBSpectrum
    hotspot::BitVector
    selection::Symbol
    stratum_counts::NamedTuple{(:core,:adjacent,:tree),Tuple{Int,Int,Int}}
    selected_counts::NamedTuple{(:core,:adjacent,:tree),Tuple{Int,Int,Int}}
    functional::Float64
    functional_method::Symbol
    batch_size::Int
    seed::Int
    problem::P
end

function implied_edge_stratum(u::Int, v::Int, in_core::BitVector)
    in_core[u] && in_core[v] && return :core
    in_core[u] || in_core[v] ? :adjacent : :tree
end

function implied_hotspot(spectrum::NBSpectrum, mass::Float64)
    hotspot = falses(length(spectrum.node_scores))
    spectrum.binding_component == 0 && return hotspot
    nodes = spectrum.components[spectrum.binding_component].nodes
    order = sort(nodes; by=node -> (-spectrum.node_scores[node], node))
    cumulative = 0.0
    @inbounds for node in order
        hotspot[node] = true
        cumulative += spectrum.node_scores[node]
        cumulative >= mass && break
    end
    return hotspot
end

function implied_select_edges(
    strata::Vector{Symbol},
    selection::Symbol,
    outside_sample::Int,
    seed::Int,
)
    selection in (:stratified, :all) ||
        throw(ArgumentError("selection must be :stratified or :all; got $(selection)."))
    pools = (
        core=findall(==(:core), strata),
        adjacent=findall(==(:adjacent), strata),
        tree=findall(==(:tree), strata),
    )
    counts = (core=length(pools.core), adjacent=length(pools.adjacent), tree=length(pools.tree))
    if selection == :all
        selected = collect(eachindex(strata))
    else
        outside_sample > 0 || throw(ArgumentError(
            "outside_sample must be positive when selection=:stratified.",
        ))
        rng = MersenneTwister(seed)
        selected = copy(pools.core)
        for pool in (pools.adjacent, pools.tree)
            take = min(outside_sample, length(pool))
            take == 0 && continue
            permutation = randperm(rng, length(pool))
            append!(selected, pool[permutation[1:take]])
        end
        sort!(selected)
    end
    selected_counts = (
        core=count(index -> strata[index] == :core, selected),
        adjacent=count(index -> strata[index] == :adjacent, selected),
        tree=count(index -> strata[index] == :tree, selected),
    )
    return selected, counts, selected_counts
end

function implied_solve_entries(
    factor,
    n::Int,
    edge_u::Vector{Int},
    edge_v::Vector{Int},
    selected::Vector{Int},
    batch_size::Int,
)
    columns = sort!(unique(vcat(edge_u[selected], edge_v[selected])))
    jobs = Dict{Int,Vector{Tuple{Int,Int}}}()
    for (position, edge) in enumerate(selected)
        column = min(edge_u[edge], edge_v[edge])
        other = max(edge_u[edge], edge_v[edge])
        push!(get!(jobs, column, Tuple{Int,Int}[]), (position, other))
    end

    diagonal = fill(NaN, n)
    off_diagonal = fill(NaN, length(selected))
    width = min(batch_size, length(columns))
    rhs = zeros(n, width)
    for start in 1:width:length(columns)
        stop = min(start + width - 1, length(columns))
        chunk = @view columns[start:stop]
        m = length(chunk)
        @inbounds for (column_index, column) in enumerate(chunk)
            rhs[column, column_index] = 1.0
        end
        solution = m == width ? factor \ rhs : factor \ rhs[:, 1:m]
        @inbounds for (column_index, column) in enumerate(chunk)
            diagonal[column] = solution[column, column_index]
            if haskey(jobs, column)
                for (position, other) in jobs[column]
                    off_diagonal[position] = solution[other, column_index]
                end
            end
            rhs[column, column_index] = 0.0
        end
    end
    return columns, diagonal, off_diagonal
end

function implied_functional(
    correlations::Vector{Float64},
    selected_strata::Vector{Symbol},
    counts::NamedTuple,
    selected_count::Int,
    total_count::Int,
)
    if selected_count == total_count
        return mean(correlations), :exact
    end
    total = 0.0
    for stratum in (:core, :adjacent, :tree)
        population = getfield(counts, stratum)
        population == 0 && continue
        values = correlations[selected_strata .== stratum]
        isempty(values) && throw(ArgumentError(
            "No selected edges in nonempty $(stratum) stratum; increase outside_sample.",
        ))
        total += population * mean(values)
    end
    return total / total_count, :stratified
end

"""
    implied_correlations(problem, rho; selection=:stratified,
                         outside_sample=10_000, batch_size=32,
                         hotspot_mass=0.5, seed=12345)

Compute exact model-implied correlations for selected unique edges under a
`VarianceStablePrior`. The VS kernel `B(rho)` is factored once and requested
inverse entries are obtained with batched column solves.

The default selects every 2-core edge and deterministic samples of up to
`outside_sample` edges from each outside-core stratum (`:adjacent` and
`:tree`). Pass `selection=:all` for every edge. The returned edge-average
functional is exact when all edges are selected and otherwise uses stratum
population weights.
"""
function implied_correlations(
    problem::GMRFProblem,
    rho::Real;
    selection::Symbol=:stratified,
    outside_sample::Int=10_000,
    batch_size::Int=32,
    hotspot_mass::Real=0.5,
    seed::Int=12345,
)
    problem.prior isa VarianceStablePrior || throw(ArgumentError(
        "implied_correlations requires a problem prepared with VarianceStablePrior.",
    ))
    rho_float = Float64(rho)
    isfinite(rho_float) || throw(ArgumentError("rho must be finite."))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive."))
    0 < hotspot_mass <= 1 || throw(ArgumentError("hotspot_mass must lie in (0, 1]."))

    cached = get(problem.metadata, :nb_spectrum, nothing)
    spectrum = cached isa NBSpectrum ? cached : nb_spectrum(problem; seed=seed)
    spectrum.converged || throw(ArgumentError(
        "Non-backtracking eigensolver did not converge; implied correlations are unavailable.",
    ))
    ceiling = nb_rho_ceiling(spectrum.lambda_nb)
    abs(rho_float) < ceiling || throw(ArgumentError(@sprintf(
        "rho must satisfy abs(rho) < %.6f for this graph; got %.6f.",
        ceiling,
        rho_float,
    )))

    graph = nb_graph(problem.A_prior)
    strata = [
        implied_edge_stratum(graph.edge_u[edge], graph.edge_v[edge], spectrum.in_two_core)
        for edge in eachindex(graph.edge_u)
    ]
    selected, stratum_counts, selected_counts = implied_select_edges(
        strata,
        selection,
        outside_sample,
        seed,
    )
    isempty(selected) && throw(ArgumentError("No edges were selected."))

    B = precision_matrix(problem, rho_float, 1.0, 1.0)
    factor = try
        cholesky(Symmetric(B))
    catch
        throw(ArgumentError("B(rho) is not positive definite at rho=$(rho_float)."))
    end
    columns, diagonal, off_diagonal = implied_solve_entries(
        factor,
        size(B, 1),
        graph.edge_u,
        graph.edge_v,
        selected,
        batch_size,
    )
    correlations = [
        off_diagonal[position] /
        sqrt(diagonal[graph.edge_u[edge]] * diagonal[graph.edge_v[edge]])
        for (position, edge) in enumerate(selected)
    ]
    all(correlation -> isfinite(correlation) && abs(correlation) <= 1.0 + 1e-10, correlations) ||
        error("Computed a non-finite or invalid implied correlation.")

    hotspot = implied_hotspot(spectrum, Float64(hotspot_mass))
    selected_strata = strata[selected]
    functional, functional_method = implied_functional(
        correlations,
        selected_strata,
        stratum_counts,
        length(selected),
        length(graph.edge_u),
    )

    F = eltype(problem.firm_ids)
    W = eltype(problem.worker_ids)
    edge_records = Vector{ImpliedCorrEdge{F,W}}(undef, length(selected))
    @inbounds for (position, edge) in enumerate(selected)
        firm = graph.edge_u[edge]
        worker_global = graph.edge_v[edge]
        worker = worker_global - problem.N_firms
        stratum = strata[edge]
        component = max(
            spectrum.core_component[firm],
            spectrum.core_component[worker_global],
        )
        component_lambda = component == 0 ? 0.0 : spectrum.components[component].lambda_nb
        inclusion_probability = getfield(selected_counts, stratum) /
            getfield(stratum_counts, stratum)
        edge_records[position] = ImpliedCorrEdge{F,W}(
            problem.firm_ids[firm],
            problem.worker_ids[worker],
            firm,
            worker,
            correlations[position],
            stratum,
            spectrum.distance_to_core[firm],
            spectrum.distance_to_core[worker_global],
            spectrum.node_scores[firm] + spectrum.node_scores[worker_global],
            component,
            component_lambda,
            hotspot[firm] || hotspot[worker_global],
            inclusion_probability,
        )
    end

    firm_records = ImpliedCorrNode{F}[]
    worker_records = ImpliedCorrNode{W}[]
    scale = 1.0 - rho_float^2
    @inbounds for node in columns
        if node <= problem.N_firms
            push!(firm_records, ImpliedCorrNode{F}(
                problem.firm_ids[node],
                node,
                diagonal[node],
                scale * diagonal[node],
                spectrum.in_two_core[node],
                spectrum.distance_to_core[node],
                spectrum.node_scores[node],
                spectrum.core_component[node],
                hotspot[node],
            ))
        else
            worker = node - problem.N_firms
            push!(worker_records, ImpliedCorrNode{W}(
                problem.worker_ids[worker],
                worker,
                diagonal[node],
                scale * diagonal[node],
                spectrum.in_two_core[node],
                spectrum.distance_to_core[node],
                spectrum.node_scores[node],
                spectrum.core_component[node],
                hotspot[node],
            ))
        end
    end

    return ImpliedCorrResult{F,W,typeof(problem)}(
        rho_float,
        edge_records,
        firm_records,
        worker_records,
        spectrum,
        hotspot,
        selection,
        stratum_counts,
        selected_counts,
        functional,
        functional_method,
        batch_size,
        seed,
        problem,
    )
end

implied_correlations(result::GMRFResult; kwargs...) =
    implied_correlations(result.problem, result.rho; kwargs...)

"""
    implied_corr_functional(result)

Return the mean model-implied matched-edge correlation and whether it was
computed from all edges (`method=:exact`) or from stratum-weighted samples
(`method=:stratified`).
"""
implied_corr_functional(result::ImpliedCorrResult) = (
    value=result.functional,
    method=result.functional_method,
    total_edges=sum(values(result.stratum_counts)),
    selected_edges=length(result.edges),
)

implied_corr_functional(problem::GMRFProblem, rho::Real; kwargs...) =
    implied_corr_functional(implied_correlations(problem, rho; kwargs...))

implied_corr_functional(result::GMRFResult; kwargs...) =
    implied_corr_functional(implied_correlations(result; kwargs...))
