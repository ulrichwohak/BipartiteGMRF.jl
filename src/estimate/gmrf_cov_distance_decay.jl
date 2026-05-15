#!/usr/bin/env julia

module GMRFCovDistanceDecay

const PROJECT_ROOT = realpath(normpath(joinpath(@__DIR__, "..", "..")))

using CSV
using JSON
using Parquet2
using DataFrames
using SparseArrays
using LinearAlgebra
using Statistics
using Printf: @printf

include("gmrf_extract_cov_submatrix.jl")

const CovTool = GMRFExtractCovSubmatrix

struct SelectionSource
    anchor_label::String
    anchor_side::Symbol
    anchor_id::String
    distance_from_anchor::Int
end

function parse_flags(args::Vector{String})
    positional = filter(arg -> !startswith(arg, "--"), args)
    flags_list = filter(arg -> startswith(arg, "--"), args)

    if isempty(positional)
        usage()
        return nothing
    end

    flags = Dict{String,String}()
    for flag in flags_list
        occursin("=", flag) || error("Unknown flag: $(flag). Flags must be --key=value.")
        key, value = split(flag, "=", limit=2)
        flags[key] = value
    end

    return (estimates_path=positional[1], flags=flags)
end

function parse_int_flag(flags::Dict{String,String}, key::String, default::Int; min_value::Int=0)
    value = parse(Int, get(flags, key, string(default)))
    value >= min_value || error("$(key) must be >= $(min_value); got $(value)")
    return value
end

function parse_bool_flag(flags::Dict{String,String}, key::String, default::Bool)
    raw = lowercase(strip(get(flags, key, string(default))))
    raw in ("true", "1", "yes", "y") && return true
    raw in ("false", "0", "no", "n") && return false
    error("$(key) must be true or false; got $(raw)")
end

function parse_sides(raw::String)::Vector{Symbol}
    sides = Symbol[]
    for token in CovTool.split_csv_arg(raw)
        s = lowercase(token)
        side = if s in ("firm", "firms")
            :firm
        elseif s in ("person", "persons", "people", "manager", "managers")
            :person
        else
            error("Unknown side in --sides=$(raw): $(token)")
        end
        side in sides || push!(sides, side)
    end
    isempty(sides) && error("--sides must contain firms and/or persons")
    return sides
end

function bipartite_neighbors(prep)::Vector{Vector{Int}}
    n = prep.N_F + prep.N_M
    neighbors = [Int[] for _ in 1:n]
    firm_idx, person_idx, _ = findnz(prep.A_fm)

    for k in eachindex(firm_idx)
        f = firm_idx[k]
        p = prep.N_F + person_idx[k]
        push!(neighbors[f], p)
        push!(neighbors[p], f)
    end

    for nb in neighbors
        sort!(nb)
    end
    return neighbors
end

function bfs_distances(neighbors::Vector{Vector{Int}}, start::Int; max_distance::Int=typemax(Int))::Vector{Int}
    n = length(neighbors)
    dist = fill(-1, n)
    queue = Vector{Int}(undef, n)
    head = 1
    tail = 1
    queue[tail] = start
    dist[start] = 0

    while head <= tail
        u = queue[head]
        head += 1
        dist[u] >= max_distance && continue

        for v in neighbors[u]
            if dist[v] == -1
                dist[v] = dist[u] + 1
                tail += 1
                queue[tail] = v
            end
        end
    end

    return dist
end

function side_range(prep, side::Symbol)
    if side == :firm
        return 1:prep.N_F
    elseif side == :person
        return (prep.N_F + 1):(prep.N_F + prep.N_M)
    else
        error("Unknown side: $(side)")
    end
end

function local_index(prep, side::Symbol, latent_index::Int)
    return side == :firm ? latent_index : latent_index - prep.N_F
end

function entity_id(prepared, side::Symbol, latent_index::Int)
    idx = local_index(prepared.prep, side, latent_index)
    return side == :firm ? prepared.firms[idx] : prepared.people[idx]
end

function entity_degree(prep, side::Symbol, latent_index::Int)::Int
    idx = local_index(prep, side, latent_index)
    d = side == :firm ? prep.d_f[idx] : prep.d_m[idx]
    return Int(round(d))
end

function entity_ref(prepared, side::Symbol, latent_index::Int)
    idx = local_index(prepared.prep, side, latent_index)
    id = side == :firm ? prepared.firms[idx] : prepared.people[idx]
    return CovTool.EntityRef(side, id, latent_index, idx)
end

function sorted_side_nodes(prepared, side::Symbol)
    prep = prepared.prep
    nodes = collect(side_range(prep, side))
    sort!(nodes; by = latent -> (
        -entity_degree(prep, side, latent),
        string(entity_id(prepared, side, latent)),
        latent,
    ))
    return nodes
end

function select_entities(prepared, neighbors::Vector{Vector{Int}};
                         sides::Vector{Symbol},
                         anchors_per_side::Int,
                         per_distance::Int,
                         max_distance::Int)
    prep = prepared.prep
    selected = Dict{Int,CovTool.EntityRef}()
    sources = Dict{Int,SelectionSource}()
    distance_grid = collect(0:2:max_distance)

    for side in sides
        candidates = sorted_side_nodes(prepared, side)
        anchors = first(candidates, min(anchors_per_side, length(candidates)))
        side_nodes = Set(side_range(prep, side))

        for anchor in anchors
            anchor_ref = entity_ref(prepared, side, anchor)
            anchor_label = CovTool.entity_label(anchor_ref)
            dist = bfs_distances(neighbors, anchor; max_distance=max_distance)

            for d in distance_grid
                at_distance = [node for node in candidates if node in side_nodes && dist[node] == d]
                chosen = first(at_distance, min(per_distance, length(at_distance)))

                for node in chosen
                    if !haskey(selected, node)
                        ref = entity_ref(prepared, side, node)
                        selected[node] = ref
                        sources[node] = SelectionSource(
                            anchor_label,
                            side,
                            string(anchor_ref.id),
                            d,
                        )
                    end
                end
            end
        end
    end

    refs = collect(values(selected))
    sort!(refs; by = ref -> (String(ref.side), string(ref.id), ref.latent_index))
    return refs, sources
end

function selected_entities_dataframe(refs::Vector{CovTool.EntityRef},
                                     sources::Dict{Int,SelectionSource},
                                     prepared)
    prep = prepared.prep
    return DataFrame(
        pos = collect(1:length(refs)),
        label = [CovTool.entity_label(ref) for ref in refs],
        side = [String(ref.side) for ref in refs],
        id = [string(ref.id) for ref in refs],
        local_index = [ref.local_index for ref in refs],
        latent_index = [ref.latent_index for ref in refs],
        degree = [entity_degree(prep, ref.side, ref.latent_index) for ref in refs],
        selected_by_anchor = [sources[ref.latent_index].anchor_label for ref in refs],
        selected_by_anchor_side = [String(sources[ref.latent_index].anchor_side) for ref in refs],
        selected_by_anchor_id = [sources[ref.latent_index].anchor_id for ref in refs],
        distance_from_anchor = [sources[ref.latent_index].distance_from_anchor for ref in refs],
    )
end

function covariance_to_correlation(cov::Matrix{Float64})::Matrix{Float64}
    n = size(cov, 1)
    size(cov, 2) == n || error("Correlation conversion requires a square covariance matrix.")
    corr = Matrix{Float64}(undef, n, n)
    variances = diag(cov)

    @inbounds for j in 1:n
        for i in 1:n
            denom = variances[i] > 0 && variances[j] > 0 ? sqrt(variances[i] * variances[j]) : NaN
            corr[i, j] = isfinite(denom) && denom > 0 ? cov[i, j] / denom : NaN
        end
    end

    return corr
end

function dense_labeled_dataframe(mat::Matrix{Float64}, refs::Vector{CovTool.EntityRef})
    labels = [CovTool.entity_label(ref) for ref in refs]
    df = DataFrame(
        entity = labels,
        side = [String(ref.side) for ref in refs],
        id = [string(ref.id) for ref in refs],
        local_index = [ref.local_index for ref in refs],
        latent_index = [ref.latent_index for ref in refs],
    )

    for j in 1:size(mat, 2)
        df[!, Symbol(labels[j])] = mat[:, j]
    end
    return df
end

function distance_matrix_for_refs(neighbors::Vector{Vector{Int}}, refs::Vector{CovTool.EntityRef})
    n = length(refs)
    distances = Matrix{Int}(undef, n, n)
    latent = [ref.latent_index for ref in refs]

    for i in 1:n
        dist = bfs_distances(neighbors, latent[i])
        for j in 1:n
            distances[i, j] = dist[latent[j]]
        end
    end

    return distances
end

function long_values_dataframe(matrix_name::Symbol,
                               cov::Matrix{Float64},
                               corr::Matrix{Float64},
                               refs::Vector{CovTool.EntityRef},
                               distances::Matrix{Int})
    n = length(refs)
    total = n * n
    matrix = fill(String(matrix_name), total)
    row_pos = Vector{Int}(undef, total)
    col_pos = Vector{Int}(undef, total)
    row_entity = Vector{String}(undef, total)
    col_entity = Vector{String}(undef, total)
    row_side = Vector{String}(undef, total)
    col_side = Vector{String}(undef, total)
    row_id = Vector{String}(undef, total)
    col_id = Vector{String}(undef, total)
    distance = Vector{Int}(undef, total)
    cov_value = Vector{Float64}(undef, total)
    corr_value = Vector{Float64}(undef, total)

    idx = 1
    @inbounds for j in 1:n
        for i in 1:n
            row_pos[idx] = i
            col_pos[idx] = j
            row_entity[idx] = CovTool.entity_label(refs[i])
            col_entity[idx] = CovTool.entity_label(refs[j])
            row_side[idx] = String(refs[i].side)
            col_side[idx] = String(refs[j].side)
            row_id[idx] = string(refs[i].id)
            col_id[idx] = string(refs[j].id)
            distance[idx] = distances[i, j]
            cov_value[idx] = cov[i, j]
            corr_value[idx] = corr[i, j]
            idx += 1
        end
    end

    return DataFrame(
        matrix = matrix,
        row_pos = row_pos,
        col_pos = col_pos,
        row_entity = row_entity,
        col_entity = col_entity,
        row_side = row_side,
        col_side = col_side,
        row_id = row_id,
        col_id = col_id,
        distance = distance,
        cov = cov_value,
        corr = corr_value,
    )
end

function pairwise_decay_dataframe(matrix_name::Symbol,
                                  cov::Matrix{Float64},
                                  corr::Matrix{Float64},
                                  refs::Vector{CovTool.EntityRef},
                                  distances::Matrix{Int})
    n = length(refs)
    matrix_col = String[]
    side_col = String[]
    row_entity = String[]
    col_entity = String[]
    row_id = String[]
    col_id = String[]
    distance_col = Int[]
    cov_col = Float64[]
    corr_col = Float64[]

    for i in 1:(n - 1)
        for j in (i + 1):n
            refs[i].side == refs[j].side || continue
            push!(matrix_col, String(matrix_name))
            push!(side_col, String(refs[i].side))
            push!(row_entity, CovTool.entity_label(refs[i]))
            push!(col_entity, CovTool.entity_label(refs[j]))
            push!(row_id, string(refs[i].id))
            push!(col_id, string(refs[j].id))
            push!(distance_col, distances[i, j])
            push!(cov_col, cov[i, j])
            push!(corr_col, corr[i, j])
        end
    end

    return DataFrame(
        matrix = matrix_col,
        side = side_col,
        row_entity = row_entity,
        col_entity = col_entity,
        row_id = row_id,
        col_id = col_id,
        distance = distance_col,
        cov = cov_col,
        corr = corr_col,
    )
end

function summarize_by_distance(pairwise::DataFrame)
    matrix_col = String[]
    side_col = String[]
    distance_col = Int[]
    n_pairs_col = Int[]
    mean_cov_col = Float64[]
    median_cov_col = Float64[]
    mean_abs_cov_col = Float64[]
    mean_corr_col = Float64[]
    median_corr_col = Float64[]
    mean_abs_corr_col = Float64[]
    min_corr_col = Float64[]
    max_corr_col = Float64[]

    if isempty(pairwise)
        return DataFrame(
            matrix = matrix_col,
            side = side_col,
            distance = distance_col,
            n_pairs = n_pairs_col,
            mean_cov = mean_cov_col,
            median_cov = median_cov_col,
            mean_abs_cov = mean_abs_cov_col,
            mean_corr = mean_corr_col,
            median_corr = median_corr_col,
            mean_abs_corr = mean_abs_corr_col,
            min_corr = min_corr_col,
            max_corr = max_corr_col,
        )
    end

    for g in groupby(pairwise, [:matrix, :side, :distance])
        cov_values = [Float64(x) for x in skipmissing(g.cov) if isfinite(Float64(x))]
        corr_values = [Float64(x) for x in skipmissing(g.corr) if isfinite(Float64(x))]
        isempty(cov_values) && continue
        isempty(corr_values) && continue

        push!(matrix_col, first(g.matrix))
        push!(side_col, first(g.side))
        push!(distance_col, first(g.distance))
        push!(n_pairs_col, nrow(g))
        push!(mean_cov_col, mean(cov_values))
        push!(median_cov_col, median(cov_values))
        push!(mean_abs_cov_col, mean(abs.(cov_values)))
        push!(mean_corr_col, mean(corr_values))
        push!(median_corr_col, median(corr_values))
        push!(mean_abs_corr_col, mean(abs.(corr_values)))
        push!(min_corr_col, minimum(corr_values))
        push!(max_corr_col, maximum(corr_values))
    end

    out = DataFrame(
        matrix = matrix_col,
        side = side_col,
        distance = distance_col,
        n_pairs = n_pairs_col,
        mean_cov = mean_cov_col,
        median_cov = median_cov_col,
        mean_abs_cov = mean_abs_cov_col,
        mean_corr = mean_corr_col,
        median_corr = median_corr_col,
        mean_abs_corr = mean_abs_corr_col,
        min_corr = min_corr_col,
        max_corr = max_corr_col,
    )
    sort!(out, [:matrix, :side, :distance])
    return out
end

function select_existing_columns(df::DataFrame, cols::Vector{Symbol})::DataFrame
    present = Symbol[]
    names = propertynames(df)
    for col in cols
        col in names && push!(present, col)
    end
    return select(df, present)
end

function covariance_summary_dataframe(summary::DataFrame)::DataFrame
    return select_existing_columns(summary, [
        :object,
        :matrix,
        :side,
        :endpoint_side,
        :tier,
        :endpoint_group,
        :distance,
        :base_distance,
        :n_pairs,
        :mean_cov,
        :median_cov,
        :mean_abs_cov,
        :mean_cov_ratio_to_d2,
        :mean_cov_ratio_to_base,
        :rho_power,
        :mean_rho_power,
        :mean_weighted_benchmark,
    ])
end

function correlation_summary_dataframe(summary::DataFrame)::DataFrame
    return select_existing_columns(summary, [
        :object,
        :matrix,
        :side,
        :endpoint_side,
        :tier,
        :endpoint_group,
        :distance,
        :base_distance,
        :n_pairs,
        :mean_corr,
        :median_corr,
        :mean_abs_corr,
        :min_corr,
        :max_corr,
        :mean_corr_ratio_to_d2,
        :mean_corr_ratio_to_base,
        :rho_power,
        :mean_rho_power,
        :mean_weighted_benchmark,
    ])
end

function matrix_modes(mode::Symbol)::Vector{Symbol}
    mode == :both && return [:prior, :posterior]
    mode in (:prior, :posterior) || error("--matrix must be prior, posterior, or both; got $(mode)")
    return [mode]
end

function extract_blocks(prep, meta, refs::Vector{CovTool.EntityRef};
                        matrix_mode::Symbol,
                        units::Symbol,
                        batch_size::Int)
    built = CovTool.build_exact_matrices(prep, meta)
    n = prep.N_F + prep.N_M
    idx = [ref.latent_index for ref in refs]
    scale = units == :original ? prep.y_std^2 : 1.0
    blocks = Dict{Symbol,Matrix{Float64}}()

    for mode in matrix_modes(matrix_mode)
        if mode == :prior
            @printf("Factoring prior precision Q...\n")
            F = cholesky(Symmetric(built.Q))
        else
            @printf("Factoring posterior precision M = Q + lambda * V'V...\n")
            F = cholesky(Symmetric(built.M))
        end

        @printf("Extracting %s covariance block...\n", String(mode))
        blocks[mode] = CovTool.extract_submatrix(F, n, idx, idx; batch_size=batch_size) .* scale
    end

    return blocks
end

function write_decay_outputs(out_dir::String,
                             meta,
                             refs::Vector{CovTool.EntityRef},
                             selected_df::DataFrame,
                             distances::Matrix{Int},
                             blocks::Dict{Symbol,Matrix{Float64}};
                             units::Symbol,
                             options::Dict{String,Any})
    mkpath(out_dir)

    selected_path = joinpath(out_dir, "selected_entities.csv")
    CSV.write(selected_path, selected_df)

    all_long = DataFrame[]
    all_pairwise = DataFrame[]
    output_files = Dict{String,Any}("selected_entities_csv" => selected_path)

    for mode in sort!(collect(keys(blocks)); by=String)
        cov = blocks[mode]
        corr = covariance_to_correlation(cov)
        base = String(mode)

        cov_dense = joinpath(out_dir, "$(base)_cov_dense.csv")
        cov_long = joinpath(out_dir, "$(base)_cov_long.csv")
        corr_dense = joinpath(out_dir, "$(base)_corr_dense.csv")
        corr_long = joinpath(out_dir, "$(base)_corr_long.csv")

        CSV.write(cov_dense, dense_labeled_dataframe(cov, refs))
        CSV.write(corr_dense, dense_labeled_dataframe(corr, refs))

        long_df = long_values_dataframe(mode, cov, corr, refs, distances)
        CSV.write(cov_long, select(long_df, Not(:corr)))
        CSV.write(corr_long, select(long_df, Not(:cov)))

        push!(all_long, long_df)
        push!(all_pairwise, pairwise_decay_dataframe(mode, cov, corr, refs, distances))

        output_files["$(base)_cov_dense_csv"] = cov_dense
        output_files["$(base)_cov_long_csv"] = cov_long
        output_files["$(base)_corr_dense_csv"] = corr_dense
        output_files["$(base)_corr_long_csv"] = corr_long
    end

    matrix_values = vcat(all_long...)
    pairwise = vcat(all_pairwise...)
    by_distance = summarize_by_distance(pairwise)

    matrix_values_path = joinpath(out_dir, "matrix_values_long.csv")
    pairwise_path = joinpath(out_dir, "pairwise_decay.csv")
    by_distance_path = joinpath(out_dir, "by_distance.csv")
    by_distance_cov_path = joinpath(out_dir, "by_distance_cov.csv")
    by_distance_corr_path = joinpath(out_dir, "by_distance_corr.csv")

    CSV.write(matrix_values_path, matrix_values)
    CSV.write(pairwise_path, pairwise)
    CSV.write(by_distance_path, by_distance)
    CSV.write(by_distance_cov_path, covariance_summary_dataframe(by_distance))
    CSV.write(by_distance_corr_path, correlation_summary_dataframe(by_distance))

    output_files["matrix_values_long_csv"] = matrix_values_path
    output_files["pairwise_decay_csv"] = pairwise_path
    output_files["by_distance_csv"] = by_distance_path
    output_files["by_distance_cov_csv"] = by_distance_cov_path
    output_files["by_distance_corr_csv"] = by_distance_corr_path

    metadata = Dict(
        "estimates_path" => meta.estimates_path,
        "input_path" => meta.input_path,
        "outcome" => String(meta.outcome),
        "adjacency_weighting" => String(meta.a_weighting),
        "prior_adjacency" => String(meta.prior_adjacency),
        "obs_weighting" => String(meta.obs_weighting),
        "decomp_target" => String(meta.decomp_target),
        "rho_eps" => meta.rho_eps,
        "maxdeg" => meta.maxdeg,
        "rho" => meta.rho,
        "sigma_a" => meta.sigma_a,
        "sigma_z" => meta.sigma_z,
        "sigma_eps" => meta.sigma_eps,
        "y_std" => meta.y_std,
        "N_F" => meta.N_F,
        "N_M" => meta.N_M,
        "K" => meta.K,
        "n" => meta.n,
        "units" => String(units),
        "selected_count" => length(refs),
        "pair_definition" => "distinct unordered pairs on the same side of the bipartite graph",
        "graph_distance" => "unweighted shortest-path distance over nonzero entries of the GMRF prior graph A_fm",
        "correlation_formula" => "corr_ij = cov_ij / sqrt(cov_ii * cov_jj), computed separately for each requested covariance matrix",
        "options" => options,
        "output_files" => output_files,
    )

    metadata_path = joinpath(out_dir, "metadata.json")
    open(metadata_path, "w") do io
        JSON.print(io, metadata, 2)
    end
    output_files["metadata_json"] = metadata_path

    return output_files
end

function usage()
    println(
        "Usage:\n" *
        "  julia --project=. src/estimate/gmrf_cov_distance_decay.jl <path/to/estimates.txt> [flags]\n" *
        "\n" *
        "Defaults are designed for `make cov-decay` on output/gmrfmle/full/full/estimates.txt.\n" *
        "\n" *
        "Flags:\n" *
        "  --matrix=prior|posterior|both      Covariance matrix to extract (default: prior)\n" *
        "  --sides=firms,persons             Sides to include (default: firms,persons)\n" *
        "  --anchors=N                       Top-degree anchors per side (default: 2)\n" *
        "  --per-distance=N                  Nodes per anchor-distance bin (default: 5)\n" *
        "  --max-distance=N                  Maximum same-side distance from anchors (default: 8)\n" *
        "  --units=original|scaled            Output covariance units (default: original)\n" *
        "  --batch-size=N                     RHS batch size for solves (default: 4)\n" *
        "  --name=<label>                     Output label (default: distance_decay)\n" *
        "  --output-dir=<path>                Override output directory\n" *
        "  --allow-sample-mismatch=true|false Continue if sample metadata differs from estimates (default: false)\n"
    )
end

function main(args::Vector{String}=ARGS)
    parsed = parse_flags(args)
    parsed === nothing && return 1

    estimates_path = CovTool.repo_path(parsed.estimates_path)
    flags = parsed.flags
    isfile(estimates_path) || error("Estimates file not found: $(estimates_path)")

    matrix_mode = Symbol(get(flags, "--matrix", "prior"))
    matrix_modes(matrix_mode)

    units = Symbol(get(flags, "--units", "original"))
    units in (:original, :scaled) || error("--units must be original or scaled; got $(units)")

    sides = parse_sides(get(flags, "--sides", "firms,persons"))
    anchors_per_side = parse_int_flag(flags, "--anchors", 2; min_value=1)
    per_distance = parse_int_flag(flags, "--per-distance", 5; min_value=1)
    max_distance = parse_int_flag(flags, "--max-distance", 8; min_value=0)
    batch_size = parse_int_flag(flags, "--batch-size", 4; min_value=1)
    allow_sample_mismatch = parse_bool_flag(flags, "--allow-sample-mismatch", false)
    name = get(flags, "--name", "distance_decay")
    out_dir = haskey(flags, "--output-dir") ?
        CovTool.repo_path(flags["--output-dir"]) :
        CovTool.default_output_dir(estimates_path, name)

    @printf("Reading estimates: %s\n", estimates_path)
    meta = CovTool.parse_estimates_file(estimates_path)

    input_path = CovTool.repo_path(meta.input_path)
    isfile(input_path) || error("Input parquet not found: $(input_path)")

    @printf("Reading sample parquet: %s\n", input_path)
    df = Parquet2.readfile(input_path) |> DataFrame
    @printf("Loaded rows: %d\n", nrow(df))

    if meta.maxdeg !== nothing
        df = CovTool.filter_maxdeg(df, meta.maxdeg; verbose=true)
        nrow(df) > 0 || error("No rows remain after maxdeg=$(meta.maxdeg)")
    end

    prepared = CovTool.prepare_data_with_ids(
        df;
        outcome=meta.outcome,
        a_weighting=meta.a_weighting,
        prior_adjacency=meta.prior_adjacency,
        obs_weighting=meta.obs_weighting,
        rho_eps=meta.rho_eps,
        verbose=true,
    )
    prep = prepared.prep
    CovTool.validate_reconstructed_sample(meta, prep;
                                          strict=!allow_sample_mismatch,
                                          firms=prepared.firms,
                                          people=prepared.people)

    neighbors = bipartite_neighbors(prep)
    refs, sources = select_entities(prepared, neighbors;
                                    sides=sides,
                                    anchors_per_side=anchors_per_side,
                                    per_distance=per_distance,
                                    max_distance=max_distance)
    isempty(refs) && error("No entities selected.")

    @printf("Selected entities: %d\n", length(refs))
    @printf("Output directory: %s\n", out_dir)

    selected_df = selected_entities_dataframe(refs, sources, prepared)
    distances = distance_matrix_for_refs(neighbors, refs)
    blocks = extract_blocks(prep, meta, refs;
                            matrix_mode=matrix_mode,
                            units=units,
                            batch_size=batch_size)

    options = Dict{String,Any}(
        "matrix" => String(matrix_mode),
        "sides" => String.(sides),
        "anchors_per_side" => anchors_per_side,
        "per_distance" => per_distance,
        "max_distance" => max_distance,
        "batch_size" => batch_size,
        "name" => name,
        "allow_sample_mismatch" => allow_sample_mismatch,
    )
    written = write_decay_outputs(out_dir, meta, refs, selected_df, distances, blocks;
                                  units=units, options=options)

    @printf("Wrote selected entities: %s\n", written["selected_entities_csv"])
    @printf("Wrote pairwise decay: %s\n", written["pairwise_decay_csv"])
    @printf("Wrote by-distance summary: %s\n", written["by_distance_csv"])
    @printf("Wrote metadata: %s\n", written["metadata_json"])
    return 0
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(GMRFCovDistanceDecay.main())
end
