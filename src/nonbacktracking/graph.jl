struct NBBipartiteGraph
    adjacency::SparseMatrixCSC{Float64,Int}
    edge_u::Vector{Int}
    edge_v::Vector{Int}
    n_firms::Int
    n_workers::Int
end

function nb_graph(A_prior::SparseMatrixCSC)
    n_firms, n_workers = size(A_prior)
    rows, cols, vals = findnz(A_prior)
    keep = findall(!iszero, vals)
    edge_u = Int.(rows[keep])
    edge_v = n_firms .+ Int.(cols[keep])
    n = n_firms + n_workers
    adjacency = sparse(
        vcat(edge_u, edge_v),
        vcat(edge_v, edge_u),
        ones(Float64, 2 * length(edge_u)),
        n,
        n,
    )
    return NBBipartiteGraph(adjacency, edge_u, edge_v, n_firms, n_workers)
end

function nb_two_core(adjacency::SparseMatrixCSC{Float64,Int})
    n = size(adjacency, 1)
    active = trues(n)
    degree = Vector{Int}(undef, n)
    queue = Int[]
    sizehint!(queue, n)
    @inbounds for node in 1:n
        degree[node] = adjacency.colptr[node + 1] - adjacency.colptr[node]
        degree[node] < 2 && push!(queue, node)
    end

    head = 1
    while head <= length(queue)
        node = queue[head]
        head += 1
        active[node] || continue
        active[node] = false
        @inbounds for ptr in adjacency.colptr[node]:(adjacency.colptr[node + 1] - 1)
            neighbor = adjacency.rowval[ptr]
            active[neighbor] || continue
            degree[neighbor] -= 1
            degree[neighbor] == 1 && push!(queue, neighbor)
        end
    end
    return active
end

function nb_components(
    adjacency::SparseMatrixCSC{Float64,Int},
    active::AbstractVector{Bool}=trues(size(adjacency, 1)),
)
    n = size(adjacency, 1)
    length(active) == n || throw(DimensionMismatch("active mask must have length $(n)."))
    labels = zeros(Int, n)
    components = Vector{Vector{Int}}()
    queue = Int[]

    for start in 1:n
        active[start] && labels[start] == 0 || continue
        label = length(components) + 1
        empty!(queue)
        push!(queue, start)
        labels[start] = label
        head = 1
        while head <= length(queue)
            node = queue[head]
            head += 1
            @inbounds for ptr in adjacency.colptr[node]:(adjacency.colptr[node + 1] - 1)
                neighbor = adjacency.rowval[ptr]
                if active[neighbor] && labels[neighbor] == 0
                    labels[neighbor] = label
                    push!(queue, neighbor)
                end
            end
        end
        push!(components, copy(queue))
    end
    return components, labels
end

function nb_distance_to_core(
    adjacency::SparseMatrixCSC{Float64,Int},
    in_core::AbstractVector{Bool},
)
    n = size(adjacency, 1)
    length(in_core) == n || throw(DimensionMismatch("core mask must have length $(n)."))
    distance = fill(-1, n)
    queue = findall(in_core)
    @inbounds for node in queue
        distance[node] = 0
    end

    head = 1
    while head <= length(queue)
        node = queue[head]
        head += 1
        @inbounds for ptr in adjacency.colptr[node]:(adjacency.colptr[node + 1] - 1)
            neighbor = adjacency.rowval[ptr]
            if distance[neighbor] == -1
                distance[neighbor] = distance[node] + 1
                push!(queue, neighbor)
            end
        end
    end
    return distance
end

function nb_edge_count(adjacency::SparseMatrixCSC{Float64,Int}, nodes::Vector{Int})
    in_component = falses(size(adjacency, 1))
    in_component[nodes] .= true
    twice_edges = 0
    @inbounds for node in nodes
        for ptr in adjacency.colptr[node]:(adjacency.colptr[node + 1] - 1)
            twice_edges += in_component[adjacency.rowval[ptr]]
        end
    end
    return div(twice_edges, 2)
end
