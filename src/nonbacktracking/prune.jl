struct NBPruneRound
    round::Int
    lambda_before::Float64
    lambda_after::Float64
    binding_component::Int
    candidates::Int
    dropped::Int
    unique_edges_remaining::Int
end

struct NBDroppedEdge{F,W}
    firm_id::F
    worker_id::W
    drop_round::Int
    score::Float64
end

"""
Connectivity-preserving result returned by [`nb_prune_edges`](@ref).

`keep` is aligned with the original input rows. Repeated rows for the same
firm-worker edge always have the same value. `dropped_edges` contains one typed
record per removed unique edge, while `audit` records each spectral pruning
round. A protected spanning forest preserves every input node and connected
component.
"""
struct NBPruneResult{F,W}
    keep::BitVector
    dropped_edges::Vector{NBDroppedEdge{F,W}}
    audit::Vector{NBPruneRound}
    lambda_before::Float64
    lambda_after::Float64
    target::Float64
    target_met::Bool
    unique_edges_before::Int
    unique_edges_after::Int
    protected_edges::Int
    components_before::Int
    components_after::Int
    firm_id::Symbol
    worker_id::Symbol
    seed::Int
end

function nb_unique_edges(df::DataFrame, firm_id::Symbol, worker_id::Symbol)
    firm_id == worker_id && throw(ArgumentError("firm_id and worker_id must be different columns."))
    for column in (firm_id, worker_id)
        hasproperty(df, column) || throw(ArgumentError("Missing required column $(column)."))
        any(ismissing, df[!, column]) &&
            throw(ArgumentError("Column $(column) contains missing IDs."))
    end
    nrow(df) > 0 || throw(ArgumentError("Empty dataset."))

    firm_ids = collect(unique(df[!, firm_id]))
    worker_ids = collect(unique(df[!, worker_id]))
    firm_to_index = Dict(id => index for (index, id) in enumerate(firm_ids))
    worker_to_index = Dict(id => index for (index, id) in enumerate(worker_ids))
    edge_index = Dict{Tuple{Int,Int},Int}()
    edge_firm = Int[]
    edge_worker = Int[]
    row_edge = Vector{Int}(undef, nrow(df))

    @inbounds for row in 1:nrow(df)
        firm = firm_to_index[df[row, firm_id]]
        worker = worker_to_index[df[row, worker_id]]
        key = (firm, worker)
        edge = get(edge_index, key, 0)
        if edge == 0
            push!(edge_firm, firm)
            push!(edge_worker, worker)
            edge = length(edge_firm)
            edge_index[key] = edge
        end
        row_edge[row] = edge
    end
    return (
        firm_ids=firm_ids,
        worker_ids=worker_ids,
        edge_firm=edge_firm,
        edge_worker=edge_worker,
        row_edge=row_edge,
    )
end

function nb_spanning_forest(edge_firm::Vector{Int}, edge_worker::Vector{Int}, n_firms::Int, n::Int)
    parent = collect(1:n)
    function root(node)
        while parent[node] != node
            parent[node] = parent[parent[node]]
            node = parent[node]
        end
        return node
    end

    protected = falses(length(edge_firm))
    @inbounds for edge in eachindex(edge_firm)
        firm_root = root(edge_firm[edge])
        worker_root = root(n_firms + edge_worker[edge])
        if firm_root != worker_root
            parent[worker_root] = firm_root
            protected[edge] = true
        end
    end
    return protected
end

function nb_alive_matrix(
    edge_firm::Vector{Int},
    edge_worker::Vector{Int},
    alive::BitVector,
    n_firms::Int,
    n_workers::Int,
)
    edges = findall(alive)
    return sparse(
        edge_firm[edges],
        edge_worker[edges],
        ones(Float64, length(edges)),
        n_firms,
        n_workers,
    )
end

function nb_component_count(A_prior::SparseMatrixCSC{Float64,Int})
    graph = nb_graph(A_prior)
    components, _ = nb_components(graph.adjacency)
    return length(components)
end

"""
    nb_prune_edges(df; firm_id=:firm_id, worker_id=:worker_id, target,
                   batch_size=1500, step_fraction=0.05, max_rounds=4000,
                   seed=12345, nev=6)

Remove cycle-closing firm-worker edges until the non-backtracking radius is
strictly below `target`. A deterministic spanning forest is protected, so no
node or connected component is removed. Only non-tree edges in the currently
binding 2-core component are eligible, ordered by endpoint localization score
with original edge order as the tie-breaker.

Input rows are not modified. Use [`pruned_dataframe`](@ref) to apply the
returned row mask. Repeated rows representing one unique edge move together.
"""
function nb_prune_edges(
    df::DataFrame;
    firm_id::Symbol=:firm_id,
    worker_id::Symbol=:worker_id,
    target::Real,
    batch_size::Int=1500,
    step_fraction::Real=0.05,
    max_rounds::Int=4000,
    seed::Int=12345,
    nev::Int=6,
)
    target_float = Float64(target)
    isfinite(target_float) && target_float > 0 ||
        throw(ArgumentError("target must be finite and positive."))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive."))
    0 < step_fraction <= 1 || throw(ArgumentError("step_fraction must lie in (0, 1]."))
    max_rounds > 0 || throw(ArgumentError("max_rounds must be positive."))
    nev > 0 || throw(ArgumentError("nev must be positive."))

    edges = nb_unique_edges(df, firm_id, worker_id)
    n_firms = length(edges.firm_ids)
    n_workers = length(edges.worker_ids)
    n = n_firms + n_workers
    protected = nb_spanning_forest(edges.edge_firm, edges.edge_worker, n_firms, n)
    alive = trues(length(edges.edge_firm))
    drop_round = zeros(Int, length(alive))
    drop_score = fill(NaN, length(alive))
    audit = NBPruneRound[]

    initial_matrix = nb_alive_matrix(
        edges.edge_firm,
        edges.edge_worker,
        alive,
        n_firms,
        n_workers,
    )
    components_before = nb_component_count(initial_matrix)
    spectrum = nb_spectrum(initial_matrix; seed=seed, nev=nev)
    spectrum.converged || throw(ArgumentError(
        "Non-backtracking eigensolver did not converge before pruning; no edges were removed.",
    ))
    lambda_before = spectrum.lambda_nb
    round = 0

    while spectrum.lambda_nb >= target_float && round < max_rounds
        binding = spectrum.binding_component
        binding > 0 || break
        component = spectrum.components[binding]
        in_binding = falses(n)
        in_binding[component.nodes] .= true
        candidates = Int[]
        scores = Float64[]
        @inbounds for edge in eachindex(alive)
            alive[edge] && !protected[edge] || continue
            firm = edges.edge_firm[edge]
            worker = n_firms + edges.edge_worker[edge]
            if in_binding[firm] && in_binding[worker]
                push!(candidates, edge)
                push!(scores, spectrum.node_scores[firm] + spectrum.node_scores[worker])
            end
        end
        isempty(candidates) && break

        round += 1
        drop_count = min(
            batch_size,
            max(1, ceil(Int, Float64(step_fraction) * length(candidates))),
        )
        order = sortperm(eachindex(candidates); by=index -> (-scores[index], candidates[index]))
        lambda_round = spectrum.lambda_nb
        @inbounds for index in order[1:drop_count]
            edge = candidates[index]
            alive[edge] = false
            drop_round[edge] = round
            drop_score[edge] = scores[index]
        end

        current_matrix = nb_alive_matrix(
            edges.edge_firm,
            edges.edge_worker,
            alive,
            n_firms,
            n_workers,
        )
        spectrum = nb_spectrum(current_matrix; seed=seed, nev=nev)
        spectrum.converged || throw(ArgumentError(
            "Non-backtracking eigensolver did not converge in pruning round $(round).",
        ))
        push!(audit, NBPruneRound(
            round,
            lambda_round,
            spectrum.lambda_nb,
            binding,
            length(candidates),
            drop_count,
            count(alive),
        ))
    end

    final_matrix = nb_alive_matrix(
        edges.edge_firm,
        edges.edge_worker,
        alive,
        n_firms,
        n_workers,
    )
    components_after = nb_component_count(final_matrix)
    components_after == components_before ||
        error("Internal error: protected spanning forest did not preserve connected components.")
    keep = BitVector(alive[edges.row_edge])

    dropped = findall(!, alive)
    sort!(dropped; by=edge -> (drop_round[edge], edge))
    F = eltype(edges.firm_ids)
    W = eltype(edges.worker_ids)
    dropped_edges = Vector{NBDroppedEdge{F,W}}(undef, length(dropped))
    @inbounds for (index, edge) in enumerate(dropped)
        dropped_edges[index] = NBDroppedEdge{F,W}(
            edges.firm_ids[edges.edge_firm[edge]],
            edges.worker_ids[edges.edge_worker[edge]],
            drop_round[edge],
            drop_score[edge],
        )
    end
    return NBPruneResult{F,W}(
        keep,
        dropped_edges,
        audit,
        lambda_before,
        spectrum.lambda_nb,
        target_float,
        spectrum.lambda_nb < target_float,
        length(alive),
        count(alive),
        count(protected),
        components_before,
        components_after,
        firm_id,
        worker_id,
        seed,
    )
end

"""
    pruned_dataframe(df, result)

Return a copy of the rows retained by an `NBPruneResult`. The input DataFrame
must have the same row count as the DataFrame passed to `nb_prune_edges`.
"""
function pruned_dataframe(df::DataFrame, result::NBPruneResult)
    nrow(df) == length(result.keep) || throw(DimensionMismatch(
        "DataFrame has $(nrow(df)) rows but prune mask has $(length(result.keep)).",
    ))
    return df[result.keep, :]
end
