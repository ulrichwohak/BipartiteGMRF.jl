#!/usr/bin/env julia

"""
sample.jl - Sample from chunked edgelist data.

Sampling is based on unique firm-person graph edges. The output preserves the
input observation level by filtering the original rows to the selected edges.

Usage:
    julia sample.jl <input.parquet> <output.parquet> [flags]

Arguments:
    input.parquet    Input edgelist file (e.g., temp/chunks/full/edgelist.parquet)
    output.parquet   Output sampled file. Sample spec and filters are parsed from
                     the output path directory structure:
                     - "full" -> keep whole chunk
                     - "bfshop-<k>-<d>" -> BFS d-hop from k seeds (e.g., bfshop-100-10)
                     - "<p>" (as directory name) -> edge sampling fraction
                     If the leaf directory is "no-founders", edges where
                     manager_category == 1 are dropped before sampling, and the
                     sample spec is taken from the parent directory.

Flags:
    --clean-1to1      For person-year inputs, keep only rows where each
                      person-year has one firm and each firm-year has one person.

Examples:
    julia sample.jl temp/chunks/full/edgelist.parquet temp/samples/full/full/edgelist.parquet
    julia sample.jl temp/chunks/full/edgelist.parquet temp/samples/full/bfshop-100-10/edgelist.parquet
    julia sample.jl temp/chunks/full/edgelist.parquet temp/samples/full/giant/no-founders/edgelist.parquet
    julia sample.jl temp/samples/full/giant/edgelist.parquet temp/samples/full/giant/tree/edgelist.parquet
"""

using Parquet2
using DataFrames
import DataFrames: DataFrame
using Logging
using Random
using Printf: @sprintf
using Graphs: AbstractGraph, induced_subgraph, SimpleGraph, add_edge!, add_vertex!, nv, ne, neighbors, edges, src, dst, connected_components, degree

const PERSON_COL = Symbol(get(ENV, "PERSON_COL", "person_id"))
const FIRM_COL   = Symbol(get(ENV, "FIRM_COL", "frame_id_numeric"))
const RANDOM_SEED = parse(Int, get(ENV, "RANDOM_SEED", "42"))

struct LabeledGraph
    graph::SimpleGraph
    node_labels_a::Dict{String,Int}      # source side labels (e.g., firms)
    node_labels_b::Dict{String,Int}      # target side labels (e.g., persons)
    edge_data::Dict{Tuple{String,String},Dict{Symbol,Any}}  # (a_label, b_label) -> attrs
    source_col::Symbol
    target_col::Symbol
end

struct Sampler
    method::Symbol
    nodes::Union{Int,Nothing}
    diameter::Union{Int,Nothing}
    edge_frac::Union{Float64,Nothing}
    min_edges::Union{Int,Nothing}
    min_mgrs::Union{Int,Nothing}
    min_firms::Union{Int,Nothing}
end

# Keyword constructor
Sampler(method::Symbol; nodes=nothing, diameter=nothing, edge_frac=nothing,
        min_edges=nothing, min_mgrs=nothing, min_firms=nothing) =
    Sampler(method, nodes, diameter, edge_frac, min_edges, min_mgrs, min_firms)

function LabeledGraph(df::DataFrame, source_col::Symbol, target_col::Symbol)::LabeledGraph
    g = SimpleGraph()
    node_labels_a = Dict{String,Int}()  # source side
    node_labels_b = Dict{String,Int}()  # target side
    edge_data = Dict{Tuple{String,String},Dict{Symbol,Any}}()

    for row in eachrow(df)
        src_label = string(row[source_col])
        dst_label = string(row[target_col])

        # Source side (A)
        src_id = get(node_labels_a, src_label) do
            add_vertex!(g)
            nv(g)
        end
        node_labels_a[src_label] = src_id

        # Target side (B)
        dst_id = get(node_labels_b, dst_label) do
            add_vertex!(g)
            nv(g)
        end
        node_labels_b[dst_label] = dst_id

        add_edge!(g, src_id, dst_id)

        edge_attrs = Dict{Symbol,Any}()
        for col in names(df)
            col == source_col && continue
            col == target_col && continue
            edge_attrs[Symbol(col)] = row[col]
        end

        edge_data[(src_label, dst_label)] = edge_attrs
    end

    LabeledGraph(g, node_labels_a, node_labels_b, edge_data, source_col, target_col)
end

function DataFrame(lg::LabeledGraph)::DataFrame
    # Build reverse mappings
    id_to_a = Dict(v => k for (k, v) in lg.node_labels_a)
    id_to_b = Dict(v => k for (k, v) in lg.node_labels_b)

    rows = Vector{Dict{Symbol,Any}}()
    for e in edges(lg.graph)
        src_id = src(e)
        dst_id = dst(e)
        
        # Look up which side each node belongs to
        a_label = get(id_to_a, src_id, nothing)
        b_label = get(id_to_b, dst_id, nothing)
        
        if a_label === nothing || b_label === nothing
            # Try swapping
            a_label = get(id_to_a, dst_id, nothing)
            b_label = get(id_to_b, src_id, nothing)
        end
        
        (a_label === nothing || b_label === nothing) && 
            error("Cannot map edge ($src_id, $dst_id) to labels")

        row = Dict{Symbol,Any}(
            lg.source_col => a_label,
            lg.target_col => b_label
        )

        if haskey(lg.edge_data, (a_label, b_label))
            merge!(row, lg.edge_data[(a_label, b_label)])
        end
        push!(rows, row)
    end

    isempty(rows) && return DataFrame()

    all_cols = union([keys(r) for r in rows]...)
    out = DataFrame()
    for col in all_cols
        out[!, col] = [get(r, col, missing) for r in rows]
    end
    return out
end

function filter_rows_to_edges(
    df::DataFrame,
    selected_edges::Set{Tuple{String,String}},
    source_col::Symbol,
    target_col::Symbol
)::DataFrame
    keep = Vector{Bool}(undef, nrow(df))
    for (i, row) in enumerate(eachrow(df))
        keep[i] = (string(row[source_col]), string(row[target_col])) in selected_edges
    end
    return df[keep, :]
end

function count_unique_nonmissing(x)
    vals = Set{Any}()
    for v in x
        ismissing(v) && continue
        push!(vals, v)
    end
    return length(vals)
end

function clean_one_to_one_personyear(
    df::DataFrame,
    source_col::Symbol,
    target_col::Symbol;
    year_col::Symbol=:year
)::DataFrame
    for c in (source_col, target_col, year_col)
        hasproperty(df, c) || error("--clean-1to1 requires column: $(c)")
    end

    n_before = nrow(df)

    source_year_counts = combine(
        groupby(df, [source_col, year_col]),
        target_col => count_unique_nonmissing => :n_targets_source_year
    )
    target_year_counts = combine(
        groupby(df, [target_col, year_col]),
        source_col => count_unique_nonmissing => :n_sources_target_year
    )

    d = leftjoin(df, source_year_counts, on=[source_col, year_col])
    d = leftjoin(d, target_year_counts, on=[target_col, year_col])
    filter!(row -> row.n_targets_source_year == 1 && row.n_sources_target_year == 1, d)
    select!(d, Not([:n_targets_source_year, :n_sources_target_year]))

    @info "Applied 1-to-1 person-year assignment cleaning" dropped=n_before - nrow(d) remaining=nrow(d)
    return d
end

function bfs_dhop(g::SimpleGraph, start::Integer, d::Integer)::Set{Int}
    d < 0 && throw(ArgumentError("diameter d must be non-negative"))
    visited = Set{Int}([start])
    d == 0 && return visited

    queue = Vector{Tuple{Int,Int}}([(start, 0)])
    front = 1

    while front <= length(queue)
        node, dist = queue[front]
        front += 1
        dist >= d && continue
        for nb in neighbors(g, node)
            nb ∈ visited && continue
            push!(visited, nb)
            push!(queue, (nb, dist + 1))
        end
    end
    return visited
end

function _sample_nodes(g::AbstractGraph, sampler::Sampler)::Vector{Int}
    Random.seed!(RANDOM_SEED)

    k = sampler.nodes === nothing ? 1 : sampler.nodes
    d = sampler.diameter === nothing ? 1 : sampler.diameter

    n = nv(g)
    k = min(k, n)

    seeds = randperm(n)[1:k]
    all_nodes = Set{Int}()

    for s in seeds
        union!(all_nodes, bfs_dhop(g, s, d))
    end

    return sort!(collect(all_nodes))
end

function sample(lg::LabeledGraph, sampler::Sampler)::LabeledGraph
    if sampler.method == :full
        return lg

    elseif sampler.method == :bfshop
        nodes = _sample_nodes(lg.graph, sampler)
        sampled_g, vmap = induced_subgraph(lg.graph, nodes)

        # Build reverse mappings
        id_to_a = Dict(v => k for (k, v) in lg.node_labels_a)
        id_to_b = Dict(v => k for (k, v) in lg.node_labels_b)

        new_node_labels_a = Dict{String,Int}()
        new_node_labels_b = Dict{String,Int}()
        
        for new_id in 1:nv(sampled_g)
            old_id = vmap[new_id]
            if haskey(id_to_a, old_id)
                new_node_labels_a[id_to_a[old_id]] = new_id
            elseif haskey(id_to_b, old_id)
                new_node_labels_b[id_to_b[old_id]] = new_id
            end
        end

        new_edge_data = Dict{Tuple{String,String},Dict{Symbol,Any}}()
        for (ek, attrs) in lg.edge_data
            a, b = ek
            (haskey(new_node_labels_a, a) && haskey(new_node_labels_b, b)) || continue
            new_edge_data[ek] = attrs
        end

        return LabeledGraph(sampled_g, new_node_labels_a, new_node_labels_b, new_edge_data, lg.source_col, lg.target_col)

    elseif sampler.method == :giant
        # Find connected components and keep only the largest
        comps = connected_components(lg.graph)
        largest_comp = comps[argmax(length.(comps))]
        @info "Giant component" n_components=length(comps) giant_nodes=length(largest_comp)

        sampled_g, vmap = induced_subgraph(lg.graph, sort(largest_comp))

        id_to_a = Dict(v => k for (k, v) in lg.node_labels_a)
        id_to_b = Dict(v => k for (k, v) in lg.node_labels_b)

        new_node_labels_a = Dict{String,Int}()
        new_node_labels_b = Dict{String,Int}()

        for new_id in 1:nv(sampled_g)
            old_id = vmap[new_id]
            if haskey(id_to_a, old_id)
                new_node_labels_a[id_to_a[old_id]] = new_id
            elseif haskey(id_to_b, old_id)
                new_node_labels_b[id_to_b[old_id]] = new_id
            end
        end

        new_edge_data = Dict{Tuple{String,String},Dict{Symbol,Any}}()
        for (ek, attrs) in lg.edge_data
            a, b = ek
            (haskey(new_node_labels_a, a) && haskey(new_node_labels_b, b)) || continue
            new_edge_data[ek] = attrs
        end

        return LabeledGraph(sampled_g, new_node_labels_a, new_node_labels_b, new_edge_data, lg.source_col, lg.target_col)

    elseif sampler.method == :minedge
        # Keep all components with at least min_edges edges
        k = sampler.min_edges
        comps = connected_components(lg.graph)

        # Label each node with its component index in one pass
        comp_label = zeros(Int, nv(lg.graph))
        for (i, comp) in enumerate(comps)
            for v in comp
                comp_label[v] = i
            end
        end

        # Count edges per component in one pass over the graph
        comp_edge_count = zeros(Int, length(comps))
        for e in edges(lg.graph)
            comp_edge_count[comp_label[src(e)]] += 1
        end

        keep = findall(comp_edge_count .>= k)
        kept_nodes = sort(vcat([comps[i] for i in keep]...))
        @info "minedge filter" min_edges=k total_comps=length(comps) kept_comps=length(keep) kept_nodes=length(kept_nodes)

        sampled_g, vmap = induced_subgraph(lg.graph, kept_nodes)

        id_to_a = Dict(v => k for (k, v) in lg.node_labels_a)
        id_to_b = Dict(v => k for (k, v) in lg.node_labels_b)

        new_node_labels_a = Dict{String,Int}()
        new_node_labels_b = Dict{String,Int}()

        for new_id in 1:nv(sampled_g)
            old_id = vmap[new_id]
            if haskey(id_to_a, old_id)
                new_node_labels_a[id_to_a[old_id]] = new_id
            elseif haskey(id_to_b, old_id)
                new_node_labels_b[id_to_b[old_id]] = new_id
            end
        end

        new_edge_data = Dict{Tuple{String,String},Dict{Symbol,Any}}()
        for (ek, attrs) in lg.edge_data
            a, b = ek
            (haskey(new_node_labels_a, a) && haskey(new_node_labels_b, b)) || continue
            new_edge_data[ek] = attrs
        end

        return LabeledGraph(sampled_g, new_node_labels_a, new_node_labels_b, new_edge_data, lg.source_col, lg.target_col)

    elseif sampler.method == :edgefrac
        p = sampler.edge_frac
        p === nothing && error("edge_frac sampler requires edge_frac")

        Random.seed!(RANDOM_SEED)

        kept_keys = Set{Tuple{String,String}}()
        for ek in keys(lg.edge_data)
            rand() <= p && push!(kept_keys, ek)
        end

        g2 = SimpleGraph()
        new_node_labels_a = Dict{String,Int}()
        new_node_labels_b = Dict{String,Int}()
        new_edge_data = Dict{Tuple{String,String},Dict{Symbol,Any}}()

        function get_or_add_a!(lab::String)
            get(new_node_labels_a, lab) do
                add_vertex!(g2)
                nv(g2)
            end
        end
        
        function get_or_add_b!(lab::String)
            get(new_node_labels_b, lab) do
                add_vertex!(g2)
                nv(g2)
            end
        end

        for (a, b) in kept_keys
            ia = get_or_add_a!(a)
            ib = get_or_add_b!(b)
            add_edge!(g2, ia, ib)
            new_edge_data[(a, b)] = lg.edge_data[(a, b)]
        end

        return LabeledGraph(g2, new_node_labels_a, new_node_labels_b, new_edge_data, lg.source_col, lg.target_col)

    elseif sampler.method == :tree
        # BFS spanning forest: one tree per connected component.
        # For each component, run BFS from the highest-degree node and collect
        # the tree edges (u→v when v is first discovered via u).
        id_to_a = Dict(v => k for (k, v) in lg.node_labels_a)
        id_to_b = Dict(v => k for (k, v) in lg.node_labels_b)

        n = nv(lg.graph)
        degrees = degree(lg.graph)
        visited = falses(n)
        tree_edge_keys = Vector{Tuple{String,String}}()

        for start_candidate in sortperm(degrees; rev=true)
            visited[start_candidate] && continue
            visited[start_candidate] = true
            queue = [start_candidate]
            while !isempty(queue)
                u = popfirst!(queue)
                for v in neighbors(lg.graph, u)
                    visited[v] && continue
                    visited[v] = true
                    push!(queue, v)
                    # Identify which side is firm and which is manager
                    if haskey(id_to_a, u) && haskey(id_to_b, v)
                        push!(tree_edge_keys, (id_to_a[u], id_to_b[v]))
                    elseif haskey(id_to_b, u) && haskey(id_to_a, v)
                        push!(tree_edge_keys, (id_to_a[v], id_to_b[u]))
                    end
                end
            end
        end

        new_edge_data = Dict{Tuple{String,String},Dict{Symbol,Any}}(
            k => lg.edge_data[k] for k in tree_edge_keys if haskey(lg.edge_data, k)
        )

        g2 = SimpleGraph()
        new_node_labels_a = Dict{String,Int}()
        new_node_labels_b = Dict{String,Int}()

        for (a, b) in tree_edge_keys
            if !haskey(new_node_labels_a, a)
                add_vertex!(g2)
                new_node_labels_a[a] = nv(g2)
            end
            if !haskey(new_node_labels_b, b)
                add_vertex!(g2)
                new_node_labels_b[b] = nv(g2)
            end
            add_edge!(g2, new_node_labels_a[a], new_node_labels_b[b])
        end

        @info "Spanning forest" input_edges=ne(lg.graph) tree_edges=length(tree_edge_keys)
        return LabeledGraph(g2, new_node_labels_a, new_node_labels_b, new_edge_data, lg.source_col, lg.target_col)

    elseif sampler.method == :mincomp
        # Keep all connected components that contain at least min_mgrs managers
        # (B-side nodes) AND at least min_firms firms (A-side nodes).
        # Unlike minedge (which filters by edge count), this enforces diversity
        # on both sides of the bipartite graph within each component.
        min_f = sampler.min_firms
        min_m = sampler.min_mgrs
        (min_f === nothing || min_m === nothing) &&
            error("mincomp sampler requires min_firms and min_mgrs")

        comps = connected_components(lg.graph)

        firm_ids_set = Set(values(lg.node_labels_a))
        mgr_ids_set  = Set(values(lg.node_labels_b))

        keep = [i for (i, comp) in enumerate(comps)
                if count(v -> v in firm_ids_set, comp) >= min_f &&
                   count(v -> v in mgr_ids_set,  comp) >= min_m]

        kept_nodes = sort(vcat([comps[i] for i in keep]...))
        @info "mincomp filter" min_mgrs=min_m min_firms=min_f total_comps=length(comps) kept_comps=length(keep) kept_nodes=length(kept_nodes)

        sampled_g, vmap = induced_subgraph(lg.graph, kept_nodes)

        id_to_a = Dict(v => k for (k, v) in lg.node_labels_a)
        id_to_b = Dict(v => k for (k, v) in lg.node_labels_b)

        new_node_labels_a = Dict{String,Int}()
        new_node_labels_b = Dict{String,Int}()

        for new_id in 1:nv(sampled_g)
            old_id = vmap[new_id]
            if haskey(id_to_a, old_id)
                new_node_labels_a[id_to_a[old_id]] = new_id
            elseif haskey(id_to_b, old_id)
                new_node_labels_b[id_to_b[old_id]] = new_id
            end
        end

        new_edge_data = Dict{Tuple{String,String},Dict{Symbol,Any}}()
        for (ek, attrs) in lg.edge_data
            a, b = ek
            (haskey(new_node_labels_a, a) && haskey(new_node_labels_b, b)) || continue
            new_edge_data[ek] = attrs
        end

        return LabeledGraph(sampled_g, new_node_labels_a, new_node_labels_b, new_edge_data, lg.source_col, lg.target_col)

    else
        error("Unknown sampler method: $(sampler.method)")
    end
end

function parse_sampler(spec::String)::Sampler
    s = lowercase(strip(spec))

    if s == "full"
        return Sampler(:full)

    elseif s == "tree"
        return Sampler(:tree)

    elseif s == "giant"
        return Sampler(:giant)

    elseif startswith(s, "minedge-")
        parts = split(s, '-')
        length(parts) == 2 || error("minedge spec must be minedge-<k>, got '$spec'")
        k = parse(Int, parts[2])
        k >= 1 || error("min_edges k must be >= 1, got $k")
        return Sampler(:minedge; min_edges=k)

    elseif startswith(s, "mincomp-")
        # mincomp-<m>-<f>: minimum m managers and f firms per connected component
        parts = split(s, '-')
        length(parts) == 3 || error("mincomp spec must be mincomp-<m>-<f>, got '$spec'")
        m = parse(Int, parts[2])
        f = parse(Int, parts[3])
        m >= 1 || error("min_mgrs must be >= 1, got $m")
        f >= 1 || error("min_firms must be >= 1, got $f")
        return Sampler(:mincomp; min_mgrs=m, min_firms=f)

    elseif startswith(s, "bfshop")
        # Handle both bfshop:<k>:<d> and bfshop-<k>-<d> formats
        if occursin(':', s)
            parts = split(s, ':')
            length(parts) == 3 || error("bfshop spec must be bfshop:<k>:<d>, got '$spec'")
            k = parse(Int, parts[2])
            d = parse(Int, parts[3])
        else
            parts = split(s, '-')
            length(parts) == 3 || error("bfshop spec must be bfshop-<k>-<d>, got '$spec'")
            k = parse(Int, parts[2])
            d = parse(Int, parts[3])
        end
        k > 0 || error("k must be >0")
        d >= 0 || error("d must be >=0")
        return Sampler(:bfshop; nodes=k, diameter=d)

    else
        p = try
            parse(Float64, s)
        catch
            error("Unknown sample spec '$spec'. Use: full | bfshop:<k>:<d> | <p>")
        end
        (0.0 < p <= 1.0) || error("Edge fraction p must be in (0,1], got $p")
        return Sampler(:edgefrac; edge_frac=p)
    end
end

"""
Parse sample spec and filter flags from output path.
Extracts the directory name between chunk and edgelist.parquet.
If the leaf directory is "no-founders", the sample spec is taken from
the parent directory and no_founders is set to true.

Examples:
    temp/samples/full/bfshop-100-10/edgelist.parquet             -> (spec="bfshop-100-10", no_founders=false)
    temp/samples/full/giant/no-founders/edgelist.parquet          -> (spec="giant", no_founders=true)
"""
function parse_spec_from_path(output_path::String)
    normalized = replace(output_path, "\\" => "/")
    parts = split(normalized, '/')

    isempty(parts) && error("Invalid output path: $output_path")
    filename = parts[end]
    filename == "edgelist.parquet" || error("Output file must be named 'edgelist.parquet', got: $filename")

    length(parts) < 2 && error("Output path must have parent directory for sample spec")
    sample_dir = parts[end-1]

    if sample_dir == "no-founders"
        length(parts) < 3 && error("no-founders requires a parent sample directory in the path")
        return (spec=String(parts[end-2]), no_founders=true)
    end

    return (spec=String(sample_dir), no_founders=false)
end

function main()
    if length(ARGS) < 2
        println(stderr, "Usage: julia sample.jl <input.parquet> <output.parquet> [flags]")
        println(stderr, "")
        println(stderr, "Output path determines sample strategy:")
        println(stderr, "  .../full/edgelist.parquet                    -> Keep entire chunk")
        println(stderr, "  .../tree/edgelist.parquet                    -> BFS spanning forest (one tree per component)")
        println(stderr, "  .../bfshop-<k>-<d>/edgelist.parquet          -> BFS d-hop from k seeds")
        println(stderr, "  .../<p>/edgelist.parquet                     -> Edge fraction sampling")
        println(stderr, "  .../<sample>/no-founders/edgelist.parquet    -> Filter founders, then sample")
        println(stderr, "")
        println(stderr, "Flags:")
        println(stderr, "  --clean-1to1                                 -> Keep only 1-to-1 person-year assignments")
        exit(1)
    end

    input_path = ARGS[1]
    output_path = ARGS[2]
    flags = ARGS[3:end]
    clean_1to1 = false
    for flag in flags
        if flag == "--clean-1to1"
            clean_1to1 = true
        else
            error("Unknown flag: $flag")
        end
    end

    # Parse sample spec and filter flags from output path
    parsed = parse_spec_from_path(output_path)
    sample_spec = parsed.spec
    no_founders = parsed.no_founders

    isfile(input_path) || error("Input file not found: $input_path")
    mkpath(dirname(output_path))

    @info "Reading input" input_path
    df = DataFrame(Parquet2.readfile(input_path))
    @info "Loaded" rows=nrow(df) cols=names(df)

    if no_founders
        if :manager_category in propertynames(df)
            n_before = nrow(df)
            filter!(row -> ismissing(row.manager_category) || row.manager_category != 1, df)
            @info "Dropped founders (manager_category == 1)" dropped=n_before - nrow(df) remaining=nrow(df)
        else
            @warn "no-founders path detected but manager_category column not found in input; skipping"
        end
    end

    if clean_1to1
        df = clean_one_to_one_personyear(df, FIRM_COL, PERSON_COL)
        nrow(df) > 0 || error("No rows remain after --clean-1to1.")
    end

    sampler = parse_sampler(sample_spec)
    @info "Parsed sampler" method=String(sampler.method) nodes=sampler.nodes diameter=sampler.diameter edge_frac=sampler.edge_frac min_edges=sampler.min_edges

    @info "Building labeled graph"
    lg = LabeledGraph(df, FIRM_COL, PERSON_COL)
    @info "Graph built" nv=nv(lg.graph) ne=ne(lg.graph)

    @info "Sampling"
    lg_s = sample(lg, sampler)
    @info "Sampled graph" nv=nv(lg_s.graph) ne=ne(lg_s.graph)

    selected_edges = Set(keys(lg_s.edge_data))
    df_out = filter_rows_to_edges(df, selected_edges, FIRM_COL, PERSON_COL)

    @info "Writing output" output_path rows=nrow(df_out) unique_edges=length(selected_edges)
    Parquet2.writefile(output_path, df_out)

    @info "Done"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
