#!/usr/bin/env julia
# ============================================================
# simulate_network_data.jl
#
# Path-driven Monte Carlo sampler.  Generates one bipartite graph
# from a Laplacian-normalized GMRF DGP and writes it as an edgelist so it
# can be consumed by the same estimators that run on real data.
#
# Usage:
#   julia --project=. src/create/simulate_network_data.jl <out_path> [flags]
#
# Arguments:
#   out_path    temp/samples/<chunk>/<topo>/edgelist.parquet
#               <chunk> must be "mc".
#               <topo>  selects topology + parameters (see below).
#
# Flags (all optional):
#   --nf=<int>          Number of firms (default: 150)
#   --nm=<int>          Number of managers (default: = nf)
#   --rho=<float>       Precision parameter rho (default: 0.5)
#   --sigma-a=<float>   Firm effect std dev (default: 1.5)
#   --sigma-z=<float>   Manager effect std dev (default: 1.0)
#   --sigma-eps=<float> Noise std dev (default: 1.0)
#   --seed=<int>        Random seed (default: 20260323)
#
# Topology labels (parsed from the parent directory name):
#   cycle                Bipartite cycle.
#   path                 Bipartite path (tree).
#   tree                 Random bipartite tree.
#   tree-<k>             k-ary BFS bipartite tree.
#   regular-<k>          k-regular bipartite circulant.
#   complete             Complete bipartite K_{nf,nm}.
#   erdos-renyi-0p<dd>   Random bipartite with edge prob 0.<dd>.
#
# Output:
#   <out_path>                                               (edgelist.parquet)
#   dirname(<out_path>)/metadata.txt                         (truth values)
# ============================================================

using SparseArrays
using LinearAlgebra
using Random
using DataFrames
using Graphs
using Parquet2
using Printf: @printf
using Statistics: mean

# ============================================================
# 1) Topology generators
# ============================================================

"Bipartite path: fᵢ–mᵢ chain. Requires n ≥ 2."
function bipartite_path(n::Int)
    n >= 2 || error("Path requires n >= 2, got $n")
    I_idx, J_idx = Int[], Int[]
    for i in 1:n
        push!(I_idx, i); push!(J_idx, i)
    end
    for i in 2:n
        push!(I_idx, i); push!(J_idx, i - 1)
    end
    return sparse(I_idx, J_idx, ones(length(I_idx)), n, n)
end

"Random bipartite tree via random attachment. Requires n ≥ 2."
function bipartite_random_tree(n::Int; rng::AbstractRNG)
    n >= 2 || error("Tree requires n >= 2, got $n")
    I_idx = [1]; J_idx = [1]
    firms_in = [1]; mgrs_in = [1]
    for i in 2:n
        m = mgrs_in[rand(rng, 1:length(mgrs_in))]
        push!(I_idx, i); push!(J_idx, m); push!(firms_in, i)
        f = firms_in[rand(rng, 1:length(firms_in))]
        push!(I_idx, f); push!(J_idx, i); push!(mgrs_in, i)
    end
    return sparse(I_idx, J_idx, ones(length(I_idx)), n, n)
end

"k-ary bipartite BFS tree. Requires n ≥ 2, k ≥ 1."
function bipartite_k_ary_tree(n::Int, k::Int)
    n >= 2 || error("Tree requires n >= 2, got $n")
    k >= 1 || error("Branching factor k must be >= 1, got $k")
    I_idx, J_idx = Int[], Int[]
    firm_queue = [1]; next_mgr = 1; next_firm = 2
    while !isempty(firm_queue)
        mgr_queue = Int[]
        for f in firm_queue, _ in 1:k
            next_mgr > n && break
            push!(I_idx, f); push!(J_idx, next_mgr); push!(mgr_queue, next_mgr)
            next_mgr += 1
        end
        isempty(mgr_queue) && break
        firm_queue = Int[]
        for m in mgr_queue, _ in 1:k
            next_firm > n && break
            push!(I_idx, next_firm); push!(J_idx, m); push!(firm_queue, next_firm)
            next_firm += 1
        end
    end
    return sparse(I_idx, J_idx, ones(length(I_idx)), n, n)
end

"Bipartite cycle. Requires n ≥ 3."
function bipartite_cycle(n::Int)
    n >= 3 || error("Cycle requires n >= 3, got $n")
    A = bipartite_path(n)
    A[1, n] = 1.0
    return A
end

"k-regular bipartite circulant."
function bipartite_regular(n::Int, k::Int)
    1 <= k <= n || error("Degree k must be in [1, $n], got $k")
    I_idx, J_idx = Int[], Int[]
    for i in 1:n, d in 0:k-1
        push!(I_idx, i); push!(J_idx, mod1(i + d, n))
    end
    return sparse(I_idx, J_idx, ones(length(I_idx)), n, n)
end

"Complete bipartite K_{nf, nm}."
function bipartite_complete(nf::Int, nm::Int)
    I_idx, J_idx = Int[], Int[]
    for i in 1:nf, j in 1:nm
        push!(I_idx, i); push!(J_idx, j)
    end
    return sparse(I_idx, J_idx, ones(length(I_idx)), nf, nm)
end

"Bipartite Erdős–Rényi with edge probability p; isolated nodes dropped."
function bipartite_erdos_renyi(nf::Int, nm::Int, p::Float64; rng::AbstractRNG)
    0 < p <= 1 || error("Edge probability must be in (0,1], got $p")
    I_idx, J_idx = Int[], Int[]
    for i in 1:nf, j in 1:nm
        rand(rng) < p && (push!(I_idx, i); push!(J_idx, j))
    end
    isempty(I_idx) && error("Erdős–Rényi graph has zero edges (p=$p too small for nf=$nf, nm=$nm)")
    firm_keep = sort(unique(I_idx))
    mgr_keep  = sort(unique(J_idx))
    firm_map  = Dict(v => i for (i, v) in enumerate(firm_keep))
    mgr_map   = Dict(v => i for (i, v) in enumerate(mgr_keep))
    I_new = [firm_map[i] for i in I_idx]
    J_new = [mgr_map[j] for j in J_idx]
    return sparse(I_new, J_new, ones(length(I_new)), length(firm_keep), length(mgr_keep))
end

# ============================================================
# 2) Laplacian-normalized GMRF precision and sampling
# ============================================================

"""
Build the degree-normalized (Laplacian) GMRF precision matrix Q:

    Ã          = D_F^{−½} · A · D_M^{−½}      (normalized adjacency)
    Q_ii_f     = 1 / σ²_a
    Q_ii_m     = 1 / σ²_z
    Q_ij edge  = −ρ / (σ_a · σ_z) · Ã[f,m]

Positive definite for |ρ| < 1 (singular values of Ã ≤ 1), independent of
graph density, so this works for all topologies including complete graphs.
"""
function build_precision_laplacian(A::SparseMatrixCSC, ρ::Float64, σ_a::Float64, σ_z::Float64)
    nf, nm = size(A)
    n = nf + nm
    d_f = vec(sum(A, dims=2))
    d_m = vec(sum(A, dims=1))
    any(d_f .== 0) && error("Firm(s) with zero degree — check topology")
    any(d_m .== 0) && error("Manager(s) with zero degree — check topology")

    A_norm = Diagonal(1.0 ./ sqrt.(d_f)) * A * Diagonal(1.0 ./ sqrt.(d_m))
    c = ρ / (σ_a * σ_z)
    Q = zeros(n, n)
    for i in 1:nf;  Q[i, i]         = 1.0 / σ_a^2; end
    for j in 1:nm;  Q[nf + j, nf + j] = 1.0 / σ_z^2; end
    for (i, j, v) in zip(findnz(A_norm)...)
        Q[i, nf + j] = -c * v
        Q[nf + j, i] = -c * v
    end
    return Symmetric(Q)
end

"Draw one (a, z) ~ N(0, Σ) via dense Cholesky."
function sample_effects(Σ::Symmetric, nf::Int; rng::AbstractRNG)
    C = cholesky(Σ).L
    x = C * randn(rng, size(Σ, 1))
    return x[1:nf], x[nf+1:end]
end

# ============================================================
# 3) Outcome generation
# ============================================================

"Generate spell-level outcomes y_{fm} = a_f + z_m + ε for all edges."
function generate_edgelist(A::SparseMatrixCSC, a::Vector{Float64}, z::Vector{Float64},
                           σ_ε::Float64; rng::AbstractRNG)
    rows, cols, _ = findnz(A)
    K = length(rows)
    y = Vector{Float64}(undef, K)
    for k in 1:K
        y[k] = a[rows[k]] + z[cols[k]] + σ_ε * randn(rng)
    end
    return DataFrame(frame_id_numeric = rows, person_id = cols, lnR = y)
end

# ============================================================
# 4) Topology label parser
# ============================================================

"""
Parse a topology label (the directory name under temp/samples/<chunk>/) into a
(name, params) tuple.  Returns a NamedTuple with a symbol `kind` and the
topology-specific parameters.
"""
function parse_topo_label(label::AbstractString)
    s = String(label)

    s == "cycle"    && return (kind=:cycle,)
    s == "path"     && return (kind=:path,)
    s == "tree"     && return (kind=:tree_random,)
    s == "complete" && return (kind=:complete,)

    m = match(r"^tree-(\d+)$", s)
    m !== nothing && return (kind=:tree_kary, k=parse(Int, m.captures[1]))

    m = match(r"^regular-(\d+)$", s)
    m !== nothing && return (kind=:regular, k=parse(Int, m.captures[1]))

    m = match(r"^erdos-renyi-0p(\d+)$", s)
    if m !== nothing
        p = parse(Float64, "0." * m.captures[1])
        return (kind=:erdos_renyi, p=p)
    end

    error("Unknown topology label: '$s'. " *
          "Supported: cycle, path, tree, tree-<k>, regular-<k>, complete, erdos-renyi-0p<dd>")
end

"""
Parse a simulate output path into (chunk, topo_label, outdir).
Expected: temp/samples/<chunk>/<topo_label>/edgelist.parquet
"""
function parse_out_path(out_path::AbstractString)
    parts = split(replace(out_path, "\\" => "/"), '/')
    parts[end] == "edgelist.parquet" ||
        error("Output file must be named 'edgelist.parquet', got: $(parts[end])")
    samples_idx = findfirst(==("samples"), parts)
    samples_idx === nothing && error("Path must contain 'samples' directory: $out_path")
    length(parts) >= samples_idx + 3 ||
        error("Path must have format temp/samples/<chunk>/<topo>/edgelist.parquet: $out_path")
    chunk = String(parts[samples_idx + 1])
    topo  = String(parts[samples_idx + 2])
    outdir = join(parts[1:end-1], "/")
    return (chunk, topo, outdir)
end

# ============================================================
# 5) CLI
# ============================================================

function parse_flag(args, name; default=nothing, type=String)
    prefix = "--$name="
    for arg in args
        if startswith(arg, prefix)
            val = arg[length(prefix)+1:end]
            return type == String ? val : parse(type, val)
        end
    end
    return default
end

function main()
    positional = filter(a -> !startswith(a, "--"), ARGS)
    flags      = filter(a ->  startswith(a, "--"), ARGS)

    if length(positional) < 1
        println(stderr, "Usage: julia src/create/simulate_network_data.jl <out_path> [flags]")
        println(stderr, "  out_path: temp/samples/<chunk>/<topo>/edgelist.parquet")
        exit(1)
    end

    out_path = positional[1]
    chunk, topo_label, outdir = parse_out_path(out_path)

    chunk == "mc" || error("Unsupported chunk '$chunk'. Only 'mc' is supported.")

    nf    = parse_flag(flags, "nf";        default=150,      type=Int)
    nm    = parse_flag(flags, "nm";        default=nf,       type=Int)
    ρ     = parse_flag(flags, "rho";       default=0.5,      type=Float64)
    σ_a   = parse_flag(flags, "sigma-a";   default=1.5,      type=Float64)
    σ_z   = parse_flag(flags, "sigma-z";   default=1.0,      type=Float64)
    σ_ε   = parse_flag(flags, "sigma-eps"; default=1.0,      type=Float64)
    seed  = parse_flag(flags, "seed";      default=20260323, type=Int)

    abs(ρ) < 1.0 || error("|rho| must be < 1, got $ρ")

    topo = parse_topo_label(topo_label)
    rng  = MersenneTwister(seed)

    @printf("=== Monte Carlo simulation ===\n")
    @printf("Output: %s\n", out_path)
    @printf("Topology: %s | N_F=%d, N_M=%d\n", topo_label, nf, nm)
    @printf("DGP: GMRF (Laplacian-normalized Q)\n")
    @printf("rho=%.3f, sigma_a=%.3f, sigma_z=%.3f, sigma_eps=%.3f, seed=%d\n",
            ρ, σ_a, σ_z, σ_ε, seed)

    if topo.kind in (:path, :tree_random, :tree_kary, :cycle, :regular) && nm != nf
        @printf("NOTE: topology %s requires nf == nm; --nm=%d ignored, using nf=%d for both.\n",
                topo_label, nm, nf)
    end

    A = if     topo.kind == :path;        bipartite_path(nf)
    elseif     topo.kind == :cycle;       bipartite_cycle(nf)
    elseif     topo.kind == :tree_random; bipartite_random_tree(nf; rng=rng)
    elseif     topo.kind == :tree_kary;   bipartite_k_ary_tree(nf, topo.k)
    elseif     topo.kind == :regular;     bipartite_regular(nf, topo.k)
    elseif     topo.kind == :complete;    bipartite_complete(nf, nm)
    elseif     topo.kind == :erdos_renyi; bipartite_erdos_renyi(nf, nm, topo.p; rng=rng)
    else       error("Unhandled topology kind: $(topo.kind)")
    end

    actual_nf, actual_nm = size(A)
    K = nnz(A)
    @printf("Graph: N_F=%d, N_M=%d, K=%d edges\n", actual_nf, actual_nm, K)

    # Bridge diagnostic (AKM/KSS-relevant)
    g = SimpleGraph(actual_nf + actual_nm)
    for (i, j, _) in zip(findnz(A)...)
        add_edge!(g, i, actual_nf + j)
    end
    n_bridges = length(bridges(g))
    @printf("Bridges: %d / %d edges (%.1f%%)\n", n_bridges, K, 100.0 * n_bridges / K)

    # Build Σ and sample (a, z)
    Q = build_precision_laplacian(A, ρ, σ_a, σ_z)
    Σ = Symmetric(inv(Matrix(Q)))
    a, z = sample_effects(Σ, actual_nf; rng=rng)

    # Spell-level AKM-target correlation (population value under this DGP)
    rows_nz, cols_nz, _ = findnz(A)
    cross_covs  = [Σ[rows_nz[k], actual_nf + cols_nz[k]]           for k in 1:K]
    var_a_spell = [Σ[rows_nz[k], rows_nz[k]]                        for k in 1:K]
    var_z_spell = [Σ[actual_nf + cols_nz[k], actual_nf + cols_nz[k]] for k in 1:K]
    rho_akm_target = mean(cross_covs) / sqrt(mean(var_a_spell) * mean(var_z_spell))
    @printf("Population AKM spell-level target rho_akm=%.4f\n", rho_akm_target)

    df = generate_edgelist(A, a, z, σ_ε; rng=rng)

    mkpath(outdir)
    Parquet2.writefile(out_path, df)
    @printf("Wrote edgelist: %s\n", out_path)

    meta_path = joinpath(outdir, "metadata.txt")
    open(meta_path, "w") do io
        @printf(io, "chunk=%s\n", chunk)
        @printf(io, "topology=%s\n", topo_label)
        @printf(io, "nf=%d\n", actual_nf)
        @printf(io, "nm=%d\n", actual_nm)
        @printf(io, "k_edges=%d\n", K)
        @printf(io, "n_bridges=%d\n", n_bridges)
        @printf(io, "rho=%.8f\n", ρ)
        @printf(io, "sigma_a=%.8f\n", σ_a)
        @printf(io, "sigma_z=%.8f\n", σ_z)
        @printf(io, "sigma_eps=%.8f\n", σ_ε)
        @printf(io, "seed=%d\n", seed)
        @printf(io, "dgp=gmrf_laplacian\n")
        @printf(io, "rho_akm_target=%.8f\n", rho_akm_target)
    end
    @printf("Wrote metadata: %s\n", meta_path)

    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
