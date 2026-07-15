struct NBComponentSpectrum
    id::Int
    nodes::Vector{Int}
    n_edges::Int
    circuit_rank::Int
    lambda_nb::Float64
    eigenvalues::Vector{ComplexF64}
    node_scores::Vector{Float64}
    converged::Bool
end

"""
Non-backtracking spectrum diagnostics for a bipartite prior graph.

`lambda_nb` is the largest radius among connected 2-core components. Node-level
vectors use the latent ordering (firms followed by workers). `node_scores` are
normalized within each 2-core component, `core_component` is zero outside the
2-core, and `distance_to_core` is `-1` in components without a cycle.
`eigenvalues` contains at most the requested number of largest-modulus values
from each component.

If any iterative eigensolve fails, `converged` is false and `lambda_nb` is
`NaN`; the corresponding component also carries `NaN` scores.
"""
struct NBSpectrum
    lambda_nb::Float64
    eigenvalues::Vector{ComplexF64}
    node_scores::Vector{Float64}
    in_two_core::BitVector
    core_component::Vector{Int}
    distance_to_core::Vector{Int}
    components::Vector{NBComponentSpectrum}
    binding_component::Int
    circuit_rank::Int
    n_firms::Int
    n_workers::Int
    n_edges::Int
    seed::Int
    converged::Bool
end

function nb_companion(adjacency::SparseMatrixCSC{Float64,Int})
    n = size(adjacency, 1)
    degree = Float64.(diff(adjacency.colptr))
    identity = spdiagm(0 => ones(n))
    return [adjacency spdiagm(0 => 1.0 .- degree); identity spzeros(n, n)]
end

function nb_node_scores(vector::AbstractVector, n::Int)
    scores = Float64.(abs2.(@view vector[(n + 1):(2n)]))
    total = sum(scores)
    return isfinite(total) && total > 0 ? scores ./ total : fill(NaN, n)
end

function nb_dense_component(
    companion::SparseMatrixCSC{Float64,Int},
    n::Int,
    nev::Int,
)
    try
        decomposition = eigen(Matrix(companion))
        order = sortperm(abs.(decomposition.values); rev=true)
        keep = order[1:min(nev, length(order))]
        lead = first(order)
        scores = nb_node_scores(@view(decomposition.vectors[:, lead]), n)
        all(isfinite, decomposition.values) && all(isfinite, scores) || error("non-finite eigensolution")
        return (
            lambda_nb=Float64(abs(decomposition.values[lead])),
            eigenvalues=ComplexF64.(decomposition.values[keep]),
            node_scores=scores,
            converged=true,
        )
    catch
        return (
            lambda_nb=NaN,
            eigenvalues=ComplexF64[],
            node_scores=fill(NaN, n),
            converged=false,
        )
    end
end

function nb_arnoldi_component(
    companion::SparseMatrixCSC{Float64,Int},
    n::Int,
    nev::Int,
    seed::Int,
    tol::Float64,
    restarts::Int,
)
    dimension = size(companion, 1)
    k = clamp(nev, 1, max(1, dimension - 2))
    mindim = min(max(2k, 20), dimension)
    maxdim = min(max(4k, 60), dimension)
    start = randn(MersenneTwister(seed), dimension)

    try
        decomposition, history = partialschur(
            companion;
            v1=start,
            nev=k,
            which=:LM,
            tol=tol,
            mindim=mindim,
            maxdim=maxdim,
            restarts=restarts,
        )
        history.converged || return (
            lambda_nb=NaN,
            eigenvalues=ComplexF64[],
            node_scores=fill(NaN, n),
            converged=false,
        )
        values, vectors = partialeigen(decomposition)
        lead = argmax(abs.(values))
        scores = nb_node_scores(@view(vectors[:, lead]), n)
        all(isfinite, values) && all(isfinite, scores) || error("non-finite eigensolution")
        keep = partialsortperm(abs.(values), 1:min(k, length(values)); rev=true)
        sort!(keep)
        return (
            lambda_nb=Float64(abs(values[lead])),
            eigenvalues=ComplexF64.(values[keep]),
            node_scores=scores,
            converged=true,
        )
    catch
        return (
            lambda_nb=NaN,
            eigenvalues=ComplexF64[],
            node_scores=fill(NaN, n),
            converged=false,
        )
    end
end

function nb_component_spectrum(
    adjacency::SparseMatrixCSC{Float64,Int},
    nodes::Vector{Int},
    id::Int;
    nev::Int,
    seed::Int,
    tol::Float64,
    dense_threshold::Int,
    arnoldi_restarts::Int,
)
    subgraph = adjacency[nodes, nodes]
    n = length(nodes)
    n_edges = div(nnz(subgraph), 2)
    rank = n_edges - n + 1
    if n_edges == n
        return NBComponentSpectrum(
            id,
            nodes,
            n_edges,
            rank,
            1.0,
            ComplexF64[1.0, -1.0],
            fill(1.0 / n, n),
            true,
        )
    end

    companion = nb_companion(subgraph)
    result = size(companion, 1) <= dense_threshold ?
        nb_dense_component(companion, n, nev) :
        nb_arnoldi_component(companion, n, nev, seed, tol, arnoldi_restarts)
    return NBComponentSpectrum(
        id,
        nodes,
        n_edges,
        rank,
        result.lambda_nb,
        result.eigenvalues,
        result.node_scores,
        result.converged,
    )
end

"""
    nb_spectrum(problem_or_A; seed=12345, nev=6, tol=1e-7)

Compute non-backtracking spectral diagnostics from a `GMRFProblem` or directly
from its rectangular sparse prior adjacency matrix. Each connected 2-core
component is solved separately using an Ihara-Bass companion matrix. Forests
return `lambda_nb == 0`; simple cycle components return radius one.

Large components use `ArnoldiMethod` with a local, explicitly seeded start
vector. The input matrix is treated as binary support and is not modified.
"""
function nb_spectrum(
    A_prior::SparseMatrixCSC;
    seed::Int=12345,
    nev::Int=6,
    tol::Real=1e-7,
    dense_threshold::Int=2000,
    arnoldi_restarts::Int=800,
)
    nev > 0 || throw(ArgumentError("nev must be positive."))
    tol > 0 || throw(ArgumentError("tol must be positive."))
    dense_threshold >= 0 || throw(ArgumentError("dense_threshold must be nonnegative."))
    arnoldi_restarts >= 0 || throw(ArgumentError("arnoldi_restarts must be nonnegative."))

    graph = nb_graph(A_prior)
    n = graph.n_firms + graph.n_workers
    in_core = nb_two_core(graph.adjacency)
    core_components, labels = nb_components(graph.adjacency, in_core)
    distance = nb_distance_to_core(graph.adjacency, in_core)
    all_components, _ = nb_components(graph.adjacency)
    circuit_rank = length(graph.edge_u) - n + length(all_components)

    if isempty(core_components)
        return NBSpectrum(
            0.0,
            ComplexF64[],
            zeros(n),
            BitVector(in_core),
            labels,
            distance,
            NBComponentSpectrum[],
            0,
            circuit_rank,
            graph.n_firms,
            graph.n_workers,
            length(graph.edge_u),
            seed,
            true,
        )
    end

    components = NBComponentSpectrum[]
    node_scores = zeros(n)
    eigenvalues = ComplexF64[]
    for (id, nodes) in enumerate(core_components)
        component = nb_component_spectrum(
            graph.adjacency,
            nodes,
            id;
            nev=nev,
            seed=seed + id - 1,
            tol=Float64(tol),
            dense_threshold=dense_threshold,
            arnoldi_restarts=arnoldi_restarts,
        )
        push!(components, component)
        node_scores[nodes] .= component.node_scores
        append!(eigenvalues, component.eigenvalues)
    end

    did_converge = all(component -> component.converged, components)
    if did_converge
        _, binding_component = findmax(component.lambda_nb for component in components)
        lambda_nb = components[binding_component].lambda_nb
    else
        binding_component = 0
        lambda_nb = NaN
    end
    return NBSpectrum(
        lambda_nb,
        eigenvalues,
        node_scores,
        BitVector(in_core),
        labels,
        distance,
        components,
        binding_component,
        circuit_rank,
        graph.n_firms,
        graph.n_workers,
        length(graph.edge_u),
        seed,
        did_converge,
    )
end

nb_spectrum(problem::GMRFProblem; kwargs...) = nb_spectrum(problem.A_prior; kwargs...)
