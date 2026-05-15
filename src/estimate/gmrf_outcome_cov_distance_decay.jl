#!/usr/bin/env julia

module GMRFOutcomeCovDistanceDecay

using Pkg

const PROJECT_ROOT = realpath(normpath(joinpath(@__DIR__, "..", "..")))

function activate_outcome_decay_project_once!(project_root::AbstractString)
    active = Base.active_project()
    active_root = active === nothing ? nothing : (isfile(active) ? dirname(active) : active)
    if active_root === nothing || realpath(active_root) != project_root
        Pkg.activate(project_root)
    end
    return nothing
end

activate_outcome_decay_project_once!(PROJECT_ROOT)

using CSV
using JSON
using Parquet2
using DataFrames
using LinearAlgebra
using SparseArrays
using Statistics
using Printf: @printf

include("gmrf_cov_distance_decay.jl")

const Decay = GMRFCovDistanceDecay
const CovTool = GMRFCovDistanceDecay.CovTool

function parse_endpoint_sides(raw::String)::Vector{Symbol}
    s = lowercase(strip(raw))
    s == "both" && return [:firm, :person]
    return Decay.parse_sides(raw)
end

function node_side(prep, latent_index::Int)::Symbol
    return latent_index <= prep.N_F ? :firm : :person
end

function bfs_tree(neighbors::Vector{Vector{Int}}, start::Int; max_distance::Int=typemax(Int))
    n = length(neighbors)
    dist = fill(-1, n)
    parent = fill(0, n)
    queue = Vector{Int}(undef, n)
    head = 1
    tail = 1
    queue[tail] = start
    dist[start] = 0
    parent[start] = start

    while head <= tail
        u = queue[head]
        head += 1
        dist[u] >= max_distance && continue

        for v in neighbors[u]
            if dist[v] == -1
                dist[v] = dist[u] + 1
                parent[v] = u
                tail += 1
                queue[tail] = v
            end
        end
    end

    return dist, parent
end

function reconstruct_path(parent::Vector{Int}, start::Int, target::Int)::Vector{Int}
    parent[target] != 0 || error("Target $(target) is unreachable from $(start)")
    path = Int[target]
    while last(path) != start
        push!(path, parent[last(path)])
    end
    reverse!(path)
    return path
end

function build_edge_lookup(prep)
    f_rows, m_cols, weights = findnz(prep.A_fm)
    lookup = Dict{Tuple{Int,Int},NamedTuple{(:edge_id, :prior_weight),Tuple{Int,Float64}}}()
    for k in eachindex(f_rows)
        lookup[(f_rows[k], m_cols[k])] = (edge_id=k, prior_weight=Float64(weights[k]))
    end
    return lookup
end

function latent_label(prepared, latent_index::Int)::String
    side = node_side(prepared.prep, latent_index)
    return CovTool.entity_label(Decay.entity_ref(prepared, side, latent_index))
end

function outcome_path_from_latents(prepared, path::Vector{Int}, edge_lookup)
    prep = prepared.prep
    length(path) >= 3 || error("Outcome covariance paths must have distance at least 2.")
    iseven(length(path) - 1) || error("Outcome endpoints must be on the same side: $(path)")
    node_side(prep, first(path)) == node_side(prep, last(path)) ||
        error("Path endpoints must be on the same side: $(path)")

    endpoint_side = node_side(prep, first(path))
    if endpoint_side == :firm
        left_f_latent = path[1]
        left_p_latent = path[2]
        right_f_latent = path[end]
        right_p_latent = path[end - 1]
    else
        left_f_latent = path[2]
        left_p_latent = path[1]
        right_f_latent = path[end - 1]
        right_p_latent = path[end]
    end

    left_f_local = Decay.local_index(prep, :firm, left_f_latent)
    left_p_local = Decay.local_index(prep, :person, left_p_latent)
    right_f_local = Decay.local_index(prep, :firm, right_f_latent)
    right_p_local = Decay.local_index(prep, :person, right_p_latent)

    left_edge = get(edge_lookup, (left_f_local, left_p_local), nothing)
    right_edge = get(edge_lookup, (right_f_local, right_p_local), nothing)
    left_edge === nothing && error("Missing left observed edge ($(left_f_local), $(left_p_local))")
    right_edge === nothing && error("Missing right observed edge ($(right_f_local), $(right_p_local))")

    return (
        endpoint_side=endpoint_side,
        distance=length(path) - 1,
        left_firm_latent=left_f_latent,
        left_person_latent=left_p_latent,
        right_firm_latent=right_f_latent,
        right_person_latent=right_p_latent,
        left_firm_local=left_f_local,
        left_person_local=left_p_local,
        right_firm_local=right_f_local,
        right_person_local=right_p_local,
        left_firm_id=string(Decay.entity_id(prepared, :firm, left_f_latent)),
        left_person_id=string(Decay.entity_id(prepared, :person, left_p_latent)),
        right_firm_id=string(Decay.entity_id(prepared, :firm, right_f_latent)),
        right_person_id=string(Decay.entity_id(prepared, :person, right_p_latent)),
        left_edge_id=left_edge.edge_id,
        right_edge_id=right_edge.edge_id,
        left_edge_prior_weight=left_edge.prior_weight,
        right_edge_prior_weight=right_edge.prior_weight,
    )
end

function selected_path_record(prepared, path::Vector{Int}, edge_lookup, anchor::Int, target::Int)
    prep = prepared.prep
    endpoint_side = node_side(prep, anchor)
    anchor_ref = Decay.entity_ref(prepared, endpoint_side, anchor)
    target_ref = Decay.entity_ref(prepared, endpoint_side, target)
    outcome = outcome_path_from_latents(prepared, path, edge_lookup)

    return merge(outcome, (
        anchor_id=string(anchor_ref.id),
        target_id=string(target_ref.id),
        anchor_latent=anchor,
        target_latent=target,
        anchor_local=anchor_ref.local_index,
        target_local=target_ref.local_index,
        anchor_degree=Decay.entity_degree(prep, endpoint_side, anchor),
        target_degree=Decay.entity_degree(prep, endpoint_side, target),
        path_labels=join((latent_label(prepared, node) for node in path), " -> "),
        path_latent_indices=join(string.(path), " "),
    ))
end

function select_outcome_paths(prepared, neighbors::Vector{Vector{Int}};
                              endpoint_sides::Vector{Symbol},
                              anchors_per_side::Int,
                              per_distance::Int,
                              max_distance::Int)
    max_distance >= 2 || error("--max-distance must be at least 2 for outcome-pair decay.")
    prep = prepared.prep
    edge_lookup = build_edge_lookup(prep)
    paths = NamedTuple[]
    seen_pairs = Set{Tuple{Symbol,Int,Int}}()
    distance_grid = collect(2:2:max_distance)

    for side in endpoint_sides
        candidates = Decay.sorted_side_nodes(prepared, side)
        anchors = first(candidates, min(anchors_per_side, length(candidates)))

        for anchor in anchors
            dist, parent = bfs_tree(neighbors, anchor; max_distance=max_distance)

            for d in distance_grid
                selected_at_distance = 0
                for target in candidates
                    target == anchor && continue
                    dist[target] == d || continue

                    key = (side, min(anchor, target), max(anchor, target))
                    key in seen_pairs && continue

                    path = reconstruct_path(parent, anchor, target)
                    push!(paths, selected_path_record(prepared, path, edge_lookup, anchor, target))
                    push!(seen_pairs, key)
                    selected_at_distance += 1
                    selected_at_distance >= per_distance && break
                end
            end
        end
    end

    sort!(paths; by = p -> (String(p.endpoint_side), p.anchor_id, p.distance, p.target_id))
    return paths
end

function selected_paths_dataframe(paths::Vector{<:NamedTuple})
    return DataFrame(paths)
end

function latent_indices_for_paths(paths::Vector{<:NamedTuple})::Vector{Int}
    idx = Int[]
    for p in paths
        append!(idx, (p.left_firm_latent, p.left_person_latent, p.right_firm_latent, p.right_person_latent))
    end
    sort!(unique!(idx))
    return idx
end

function extract_sigma_blocks(prep, meta, latent_indices::Vector{Int};
                              matrix_mode::Symbol,
                              units::Symbol,
                              batch_size::Int)
    built = CovTool.build_exact_matrices(prep, meta)
    n = prep.N_F + prep.N_M
    scale = units == :original ? prep.y_std^2 : 1.0
    blocks = Dict{Symbol,Matrix{Float64}}()

    for mode in Decay.matrix_modes(matrix_mode)
        if mode == :prior
            @printf("Factoring prior precision Q...\n")
            F = cholesky(Symmetric(built.Q))
        else
            @printf("Factoring posterior precision M = Q + lambda * V'V...\n")
            F = cholesky(Symmetric(built.M))
        end

        @printf("Extracting %s Sigma block for %d latent nodes...\n", String(mode), length(latent_indices))
        blocks[mode] = CovTool.extract_submatrix(F, n, latent_indices, latent_indices; batch_size=batch_size) .* scale
    end

    return blocks
end

function outcome_pairwise_dataframe(matrix_name::Symbol,
                                    sigma_block::Matrix{Float64},
                                    latent_indices::Vector{Int},
                                    paths::Vector{<:NamedTuple})
    pos = Dict(latent => i for (i, latent) in enumerate(latent_indices))
    rows = NamedTuple[]

    cov_at(i::Int, j::Int) = sigma_block[pos[i], pos[j]]

    for p in paths
        cov_ff = cov_at(p.left_firm_latent, p.right_firm_latent)
        cov_fp = cov_at(p.left_firm_latent, p.right_person_latent)
        cov_pf = cov_at(p.left_person_latent, p.right_firm_latent)
        cov_pp = cov_at(p.left_person_latent, p.right_person_latent)
        cov = cov_ff + cov_fp + cov_pf + cov_pp

        var_left = cov_at(p.left_firm_latent, p.left_firm_latent) +
                   cov_at(p.left_firm_latent, p.left_person_latent) +
                   cov_at(p.left_person_latent, p.left_firm_latent) +
                   cov_at(p.left_person_latent, p.left_person_latent)
        var_right = cov_at(p.right_firm_latent, p.right_firm_latent) +
                    cov_at(p.right_firm_latent, p.right_person_latent) +
                    cov_at(p.right_person_latent, p.right_firm_latent) +
                    cov_at(p.right_person_latent, p.right_person_latent)
        denom = var_left > 0 && var_right > 0 ? sqrt(var_left * var_right) : NaN
        corr = isfinite(denom) && denom > 0 ? cov / denom : NaN

        push!(rows, (
            matrix=String(matrix_name),
            endpoint_side=String(p.endpoint_side),
            distance=p.distance,
            anchor_id=p.anchor_id,
            target_id=p.target_id,
            left_outcome="firm:$(p.left_firm_id)+person:$(p.left_person_id)",
            right_outcome="firm:$(p.right_firm_id)+person:$(p.right_person_id)",
            left_firm_id=p.left_firm_id,
            left_person_id=p.left_person_id,
            right_firm_id=p.right_firm_id,
            right_person_id=p.right_person_id,
            left_edge_id=p.left_edge_id,
            right_edge_id=p.right_edge_id,
            left_firm_latent=p.left_firm_latent,
            left_person_latent=p.left_person_latent,
            right_firm_latent=p.right_firm_latent,
            right_person_latent=p.right_person_latent,
            cov=cov,
            corr=corr,
            var_left=var_left,
            var_right=var_right,
            cov_leftfirm_rightfirm=cov_ff,
            cov_leftfirm_rightperson=cov_fp,
            cov_leftperson_rightfirm=cov_pf,
            cov_leftperson_rightperson=cov_pp,
        ))
    end

    return DataFrame(rows)
end

function summarize_by_distance(pairwise::DataFrame, rho::Float64)
    rows = NamedTuple[]
    if isempty(pairwise)
        return DataFrame(rows)
    end

    for g in groupby(pairwise, [:matrix, :endpoint_side, :distance])
        cov_values = [Float64(x) for x in skipmissing(g.cov) if isfinite(Float64(x))]
        corr_values = [Float64(x) for x in skipmissing(g.corr) if isfinite(Float64(x))]
        isempty(cov_values) && continue
        isempty(corr_values) && continue

        push!(rows, (
            matrix=first(g.matrix),
            endpoint_side=first(g.endpoint_side),
            distance=first(g.distance),
            n_pairs=nrow(g),
            mean_cov=mean(cov_values),
            median_cov=median(cov_values),
            mean_abs_cov=mean(abs.(cov_values)),
            mean_corr=mean(corr_values),
            median_corr=median(corr_values),
            mean_abs_corr=mean(abs.(corr_values)),
            min_corr=minimum(corr_values),
            max_corr=maximum(corr_values),
            rho_power=first(g.distance) >= 2 ? rho^(first(g.distance) - 2) : NaN,
        ))
    end

    out = DataFrame(rows)
    isempty(out) && return out
    sort!(out, [:matrix, :endpoint_side, :distance])

    base_cov = Dict{Tuple{String,String},Float64}()
    base_corr = Dict{Tuple{String,String},Float64}()
    for r in eachrow(out)
        if r.distance == 2
            key = (r.matrix, r.endpoint_side)
            base_cov[key] = r.mean_cov
            base_corr[key] = r.mean_corr
        end
    end

    mean_cov_ratio_to_d2 = Float64[]
    mean_corr_ratio_to_d2 = Float64[]
    for r in eachrow(out)
        key = (r.matrix, r.endpoint_side)
        bc = get(base_cov, key, NaN)
        br = get(base_corr, key, NaN)
        push!(mean_cov_ratio_to_d2, isfinite(bc) && bc != 0 ? r.mean_cov / bc : NaN)
        push!(mean_corr_ratio_to_d2, isfinite(br) && br != 0 ? r.mean_corr / br : NaN)
    end

    out[!, :mean_cov_ratio_to_d2] = mean_cov_ratio_to_d2
    out[!, :mean_corr_ratio_to_d2] = mean_corr_ratio_to_d2
    return out
end

function write_outputs(out_dir::String,
                       meta,
                       paths::Vector{<:NamedTuple},
                       latent_indices::Vector{Int},
                       blocks::Dict{Symbol,Matrix{Float64}};
                       units::Symbol,
                       options::Dict{String,Any})
    mkpath(out_dir)

    selected_df = selected_paths_dataframe(paths)
    selected_path = joinpath(out_dir, "selected_paths.csv")
    CSV.write(selected_path, selected_df)

    pairwise_parts = DataFrame[]
    for mode in sort!(collect(keys(blocks)); by=String)
        push!(pairwise_parts, outcome_pairwise_dataframe(mode, blocks[mode], latent_indices, paths))
    end
    pairwise = vcat(pairwise_parts...)
    by_distance = summarize_by_distance(pairwise, meta.rho)

    pairwise_path = joinpath(out_dir, "outcome_pairwise_decay.csv")
    by_distance_path = joinpath(out_dir, "by_distance.csv")
    by_distance_cov_path = joinpath(out_dir, "by_distance_cov.csv")
    by_distance_corr_path = joinpath(out_dir, "by_distance_corr.csv")
    CSV.write(pairwise_path, pairwise)
    CSV.write(by_distance_path, by_distance)
    CSV.write(by_distance_cov_path, Decay.covariance_summary_dataframe(by_distance))
    CSV.write(by_distance_corr_path, Decay.correlation_summary_dataframe(by_distance))

    output_files = Dict{String,Any}(
        "selected_paths_csv" => selected_path,
        "outcome_pairwise_decay_csv" => pairwise_path,
        "by_distance_csv" => by_distance_path,
        "by_distance_cov_csv" => by_distance_cov_path,
        "by_distance_corr_csv" => by_distance_corr_path,
    )

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
        "selected_path_count" => length(paths),
        "unique_latent_count" => length(latent_indices),
        "pair_definition" => "observed edge outcome pairs at same-side endpoint graph distances",
        "outcome_linear_form" => "Cov(a_f + z_m, a_g + z_n) = Sigma[f,g] + Sigma[f,n] + Sigma[m,g] + Sigma[m,n]",
        "residual_covariance" => "not added; sampled pairs are distinct observed matches",
        "outcome_variance_scope" => "var_left, var_right, and corr use the latent component a_f + z_m only; sigma_eps residual variance is not added to the denominator",
        "predictive_outcome_correlation_note" => "For raw predictive outcome correlations on distinct edges, residual covariance is zero but each variance includes residual noise, so correlations are lower in magnitude than the latent-component corr reported here.",
        "graph_distance" => "unweighted shortest-path distance over nonzero entries of the GMRF prior graph A_fm",
        "rho_benchmark" => "rho_power reports rho^(distance - 2), the paper's simple-path ratio benchmark relative to distance 2",
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
        "  julia --project=. src/estimate/gmrf_outcome_cov_distance_decay.jl <path/to/estimates.txt> [flags]\n" *
        "\n" *
        "Flags:\n" *
        "  --matrix=prior|posterior|both          Covariance matrix to extract (default: prior)\n" *
        "  --endpoint-side=firms|persons|both     Same-side endpoints to sample (default: both)\n" *
        "  --anchors=N                           Top-degree anchors per endpoint side (default: 3)\n" *
        "  --per-distance=N                      Target paths per anchor-distance bin (default: 8)\n" *
        "  --max-distance=N                      Maximum same-side endpoint distance (default: 8)\n" *
        "  --units=original|scaled                Output covariance units (default: original)\n" *
        "  --batch-size=N                         RHS batch size for solves (default: 8)\n" *
        "  --name=<label>                         Output label (default: outcome_sigma_decay)\n" *
        "  --output-dir=<path>                    Override output directory\n" *
        "  --allow-sample-mismatch=true|false     Continue if sample metadata differs from estimates (default: false)\n"
    )
end

function main(args::Vector{String}=ARGS)
    parsed = Decay.parse_flags(args)
    parsed === nothing && return 1

    estimates_path = CovTool.repo_path(parsed.estimates_path)
    flags = parsed.flags
    isfile(estimates_path) || error("Estimates file not found: $(estimates_path)")

    matrix_mode = Symbol(get(flags, "--matrix", "prior"))
    Decay.matrix_modes(matrix_mode)

    endpoint_sides = parse_endpoint_sides(get(flags, "--endpoint-side", "both"))
    units = Symbol(get(flags, "--units", "original"))
    units in (:original, :scaled) || error("--units must be original or scaled; got $(units)")

    anchors_per_side = Decay.parse_int_flag(flags, "--anchors", 3; min_value=1)
    per_distance = Decay.parse_int_flag(flags, "--per-distance", 8; min_value=1)
    max_distance = Decay.parse_int_flag(flags, "--max-distance", 8; min_value=2)
    batch_size = Decay.parse_int_flag(flags, "--batch-size", 8; min_value=1)
    allow_sample_mismatch = Decay.parse_bool_flag(flags, "--allow-sample-mismatch", false)
    name = get(flags, "--name", "outcome_sigma_decay")
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

    neighbors = Decay.bipartite_neighbors(prep)
    paths = select_outcome_paths(prepared, neighbors;
                                 endpoint_sides=endpoint_sides,
                                 anchors_per_side=anchors_per_side,
                                 per_distance=per_distance,
                                 max_distance=max_distance)
    isempty(paths) && error("No outcome paths selected.")

    latent_indices = latent_indices_for_paths(paths)
    @printf("Selected paths: %d\n", length(paths))
    @printf("Unique latent nodes in outcome forms: %d\n", length(latent_indices))
    @printf("Output directory: %s\n", out_dir)

    blocks = extract_sigma_blocks(prep, meta, latent_indices;
                                  matrix_mode=matrix_mode,
                                  units=units,
                                  batch_size=batch_size)

    options = Dict{String,Any}(
        "matrix" => String(matrix_mode),
        "endpoint_sides" => String.(endpoint_sides),
        "anchors_per_side" => anchors_per_side,
        "per_distance" => per_distance,
        "max_distance" => max_distance,
        "batch_size" => batch_size,
        "name" => name,
        "allow_sample_mismatch" => allow_sample_mismatch,
    )
    written = write_outputs(out_dir, meta, paths, latent_indices, blocks;
                            units=units, options=options)

    @printf("Wrote selected paths: %s\n", written["selected_paths_csv"])
    @printf("Wrote outcome pairwise decay: %s\n", written["outcome_pairwise_decay_csv"])
    @printf("Wrote by-distance summary: %s\n", written["by_distance_csv"])
    @printf("Wrote metadata: %s\n", written["metadata_json"])
    return 0
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(GMRFOutcomeCovDistanceDecay.main())
end
