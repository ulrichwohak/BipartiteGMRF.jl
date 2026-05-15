#!/usr/bin/env julia

module GMRFExtractCovSubmatrix

using Pkg

const PROJECT_ROOT = realpath(normpath(joinpath(@__DIR__, "..", "..")))

function activate_extract_project_once!(project_root::AbstractString)
    active = Base.active_project()
    active_root = active === nothing ? nothing : (isfile(active) ? dirname(active) : active)
    if active_root === nothing || realpath(active_root) != project_root
        Pkg.activate(project_root)
    end
    return nothing
end

activate_extract_project_once!(PROJECT_ROOT)

using CSV
using JSON
using Parquet2
using DataFrames
using SparseArrays
using LinearAlgebra
using Printf: @printf

include("gmrfmle.jl")

struct EntityRef
    side::Symbol
    id
    latent_index::Int
    local_index::Int
end

struct QuerySpec
    name::String
    mode::Symbol
    row_firms::Vector{String}
    row_people::Vector{String}
    col_firms::Vector{String}
    col_people::Vector{String}
end

function repo_path(path::AbstractString)
    return isabspath(path) ? normpath(path) : normpath(joinpath(PROJECT_ROOT, path))
end

function split_csv_arg(raw::Union{Nothing,String})::Vector{String}
    raw === nothing && return String[]
    stripped = strip(raw)
    isempty(stripped) && return String[]
    return [strip(tok) for tok in split(stripped, ',') if !isempty(strip(tok))]
end

function safe_slug(s::AbstractString)::String
    slug = replace(strip(s), r"[^A-Za-z0-9._-]+" => "_")
    return isempty(slug) ? "query" : slug
end

entity_label(ref::EntityRef) = string(ref.side == :firm ? "firm:" : "person:", ref.id)

function parse_typed_id(token::String, ::Type{T}) where {T}
    T0 = Base.nonmissingtype(T)
    if T0 <: AbstractString
        return convert(T, token)
    elseif T0 <: Integer
        return parse(T0, token)
    elseif T0 <: AbstractFloat
        return parse(T0, token)
    else
        try
            return Base.parse(T0, token)
        catch
            try
                return convert(T0, token)
            catch err
                error("Cannot parse ID '$token' as $(T): $(sprint(showerror, err))")
            end
        end
    end
end

function parse_estimates_file(path::String)
    lines = readlines(path)

    input_path      = nothing
    outcome         = nothing
    a_weighting     = nothing
    prior_adjacency = nothing
    obs_weighting   = nothing
    decomp_target   = nothing
    rho_eps         = nothing
    maxdeg          = nothing
    y_std           = nothing
    rho             = nothing
    sigma_a         = nothing
    sigma_z         = nothing
    sigma_eps       = nothing
    n_f             = nothing
    n_m             = nothing
    k_obs           = nothing
    n_latent        = nothing
    in_structural = false

    for line in lines
        s = strip(line)

        if startswith(s, "Input=")
            input_path = split(s, "=", limit=2)[2]
        elseif startswith(s, "Outcome=")
            outcome = Symbol(split(s, "=", limit=2)[2])
        elseif startswith(s, "AdjacencyWeighting=")
            a_weighting = Symbol(split(s, "=", limit=2)[2])
        elseif startswith(s, "PriorAdjacency=")
            prior_adjacency = Symbol(split(s, "=", limit=2)[2])
        elseif startswith(s, "ObsWeighting=")
            obs_weighting = Symbol(split(s, "=", limit=2)[2])
        elseif startswith(s, "DecompTarget=")
            decomp_target = Symbol(split(s, "=", limit=2)[2])
        elseif startswith(s, "RhoEps=")
            rho_eps = parse(Float64, split(s, "=", limit=2)[2])
        elseif startswith(s, "MaxDeg=")
            maxdeg = parse(Int, split(s, "=", limit=2)[2])
        elseif startswith(s, "y_std=")
            m = match(r"^y_std=([^ ]+)", s)
            m === nothing && error("Failed to parse y_std from '$s'")
            y_std = parse(Float64, m.captures[1])
        elseif (m = match(r"^N_F=(\d+), N_M=(\d+), K=(\d+), n=(\d+)$", s)) !== nothing
            n_f      = parse(Int, m.captures[1])
            n_m      = parse(Int, m.captures[2])
            k_obs    = parse(Int, m.captures[3])
            n_latent = parse(Int, m.captures[4])
        elseif s == "Estimates (structural units):"
            in_structural = true
        elseif in_structural
            if isempty(s) || occursin("Variance", s) || occursin("AKM-", s) ||
               occursin("Prior ", s) || occursin("Posterior ", s)
                in_structural = false
            elseif (m = match(r"^(rho|sigma_a|sigma_z|sigma_eps)\s*=\s*([+-]?[0-9.eE]+)$", s)) !== nothing
                key = m.captures[1]
                val = parse(Float64, m.captures[2])
                if key == "rho"
                    rho = val
                elseif key == "sigma_a"
                    sigma_a = val
                elseif key == "sigma_z"
                    sigma_z = val
                elseif key == "sigma_eps"
                    sigma_eps = val
                end
            end
        end
    end

    missing_fields = String[]
    input_path      === nothing && push!(missing_fields, "Input")
    outcome         === nothing && push!(missing_fields, "Outcome")
    a_weighting     === nothing && push!(missing_fields, "AdjacencyWeighting")
    prior_adjacency === nothing && push!(missing_fields, "PriorAdjacency")
    obs_weighting   === nothing && push!(missing_fields, "ObsWeighting")
    decomp_target   === nothing && push!(missing_fields, "DecompTarget")
    y_std           === nothing && push!(missing_fields, "y_std")
    rho             === nothing && push!(missing_fields, "rho")
    sigma_a         === nothing && push!(missing_fields, "sigma_a")
    sigma_z         === nothing && push!(missing_fields, "sigma_z")
    sigma_eps       === nothing && push!(missing_fields, "sigma_eps")
    n_f             === nothing && push!(missing_fields, "N_F")
    n_m             === nothing && push!(missing_fields, "N_M")
    k_obs           === nothing && push!(missing_fields, "K")
    n_latent        === nothing && push!(missing_fields, "n")
    isempty(missing_fields) || error("Missing fields in estimates file $(path): $(join(missing_fields, ", "))")

    a_weighting in (:degree, :spectral, :unweighted) ||
        error("Unsupported AdjacencyWeighting=$(a_weighting) in $(path)")
    prior_adjacency in (:binary, :counts) ||
        error("Unsupported PriorAdjacency=$(prior_adjacency) in $(path)")
    obs_weighting in (:raw, :edge, :effective) ||
        error("Unsupported ObsWeighting=$(obs_weighting) in $(path)")
    decomp_target = normalize_decomp_target(decomp_target)
    if obs_weighting == :effective && rho_eps === nothing
        error("ObsWeighting=effective requires RhoEps in estimates file $(path)")
    end

    return (
        estimates_path=path,
        input_path=String(input_path),
        outcome=outcome::Symbol,
        a_weighting=a_weighting::Symbol,
        prior_adjacency=prior_adjacency::Symbol,
        obs_weighting=obs_weighting::Symbol,
        decomp_target=decomp_target::Symbol,
        rho_eps=rho_eps === nothing ? nothing : Float64(rho_eps),
        maxdeg=maxdeg,
        y_std=Float64(y_std),
        rho=Float64(rho),
        sigma_a=Float64(sigma_a),
        sigma_z=Float64(sigma_z),
        sigma_eps=Float64(sigma_eps),
        N_F=Int(n_f),
        N_M=Int(n_m),
        K=Int(k_obs),
        n=Int(n_latent),
    )
end

function prepare_data_with_ids(
    df::DataFrame;
    outcome::Symbol,
    a_weighting::Symbol,
    prior_adjacency::Symbol,
    obs_weighting::Symbol,
    rho_eps::Union{Nothing,Float64}=nothing,
    verbose::Bool=true
)
    if obs_weighting == :effective && rho_eps === nothing
        error("prepare_data_with_ids requires rho_eps for effective observation weighting")
    end

    prep = prepare_data(
        df;
        outcome=outcome,
        a_weighting=a_weighting,
        prior_adjacency=prior_adjacency,
        obs_weighting=obs_weighting,
        rho_eps_fixed=obs_weighting == :effective ? rho_eps : nothing,
        rho_eps_estimate=false,
        verbose=verbose,
    )

    hasproperty(prep, :firms) || error("prepare_data must return firm IDs for covariance extraction")
    hasproperty(prep, :people) || error("prepare_data must return person IDs for covariance extraction")

    return (prep=prep, firms=copy(prep.firms), people=copy(prep.people))
end

function id_order_mismatch(side::String, expected, observed)
    if length(expected) != length(observed)
        return "$(side) ID count mismatch: prepare_data=$(length(expected)), supplied=$(length(observed))"
    end

    idx = findfirst(i -> !isequal(expected[i], observed[i]), eachindex(expected))
    idx === nothing && return nothing
    return "$(side) ID order mismatch at position $(idx): prepare_data=$(expected[idx]), supplied=$(observed[idx])"
end

function validate_reconstructed_sample(meta, prep; strict::Bool=true, firms=nothing, people=nothing)
    mismatches = String[]
    prep.N_F == meta.N_F || push!(mismatches, "N_F estimates=$(meta.N_F), rebuilt=$(prep.N_F)")
    prep.N_M == meta.N_M || push!(mismatches, "N_M estimates=$(meta.N_M), rebuilt=$(prep.N_M)")
    prep.K == meta.K || push!(mismatches, "K estimates=$(meta.K), rebuilt=$(prep.K)")
    prep.N_F + prep.N_M == meta.n || push!(mismatches, "n estimates=$(meta.n), rebuilt=$(prep.N_F + prep.N_M)")
    if firms !== nothing
        msg = id_order_mismatch("firm", prep.firms, firms)
        msg === nothing || push!(mismatches, msg)
    end
    if people !== nothing
        msg = id_order_mismatch("person", prep.people, people)
        msg === nothing || push!(mismatches, msg)
    end

    if !isempty(mismatches)
        msg = "Sample metadata mismatch: $(join(mismatches, "; "))"
        strict ? error(msg) : @warn msg
    end

    if !isapprox(prep.y_std, meta.y_std; rtol=1e-5, atol=1e-8)
        msg = "y_std mismatch: estimates file has $(meta.y_std), rebuilt sample has $(prep.y_std). The sample parquet or filters likely changed after estimation."
        strict ? error(msg) : @warn msg
    end
    return nothing
end

function build_exact_matrices(prep, meta)
    σa = meta.sigma_a / meta.y_std
    σz = meta.sigma_z / meta.y_std
    σe = meta.sigma_eps / meta.y_std

    (σa > 0 && σz > 0 && σe > 0) || error("Scaled sigmas must be positive.")

    inv_sa2 = 1.0 / (σa^2)
    inv_sz2 = 1.0 / (σz^2)
    cross   = meta.rho / (σa * σz)
    λ       = 1.0 / (σe^2)

    W = spdiagm(0 => prep.df_is) * prep.A_fm * spdiagm(0 => prep.dm_is)
    Wt = copy(transpose(W))

    Q = [spdiagm(0 => inv_sa2 .* prep.dw_f)  (-cross .* W);
         (-cross .* Wt)                      spdiagm(0 => inv_sz2 .* prep.dw_m)]
    M = Q + λ .* prep.VtV

    return (Q=Q, M=M, lambda=λ, sigma_a_scaled=σa, sigma_z_scaled=σz, sigma_eps_scaled=σe)
end

function resolve_entities(
    raw_ids::Vector{String},
    ids::Vector,
    side::Symbol,
    offset::Int
)::Vector{EntityRef}
    isempty(raw_ids) && return EntityRef[]

    T = eltype(ids)
    typed_ids = [parse_typed_id(tok, T) for tok in raw_ids]
    idx_map = Dict{T,Int}(id => i for (i, id) in enumerate(ids))

    refs = EntityRef[]
    missing = String[]
    for id in typed_ids
        if haskey(idx_map, id)
            local_idx = idx_map[id]
            push!(refs, EntityRef(side, id, offset + local_idx, local_idx))
        else
            push!(missing, string(id))
        end
    end

    isempty(missing) || error("Requested $(side) IDs not found in estimation sample: $(join(missing, ", "))")
    return refs
end

function build_query_spec(flags::Dict{String,String})
    firms      = split_csv_arg(get(flags, "--firms", nothing))
    people     = split_csv_arg(get(flags, "--persons", nothing))
    row_firms  = split_csv_arg(get(flags, "--row-firms", nothing))
    row_people = split_csv_arg(get(flags, "--row-persons", nothing))
    col_firms  = split_csv_arg(get(flags, "--col-firms", nothing))
    col_people = split_csv_arg(get(flags, "--col-persons", nothing))

    has_principal = !isempty(firms) || !isempty(people)
    has_rect = !isempty(row_firms) || !isempty(row_people) || !isempty(col_firms) || !isempty(col_people)

    if has_principal && has_rect
        error("Use either --firms/--persons for a principal block, or the explicit row/col flags, not both.")
    end

    mode = :rectangular
    if has_principal
        row_firms  = copy(firms)
        row_people = copy(people)
        col_firms  = copy(firms)
        col_people = copy(people)
        mode = :principal
    elseif !has_rect
        error("No block requested. Use --firms/--persons for a principal block or explicit row/col flags.")
    elseif isempty(col_firms) && isempty(col_people)
        col_firms  = copy(row_firms)
        col_people = copy(row_people)
        mode = :principal
    elseif isempty(row_firms) && isempty(row_people)
        row_firms  = copy(col_firms)
        row_people = copy(col_people)
        mode = :principal
    end

    nrf = length(row_firms)
    nrp = length(row_people)
    ncf = length(col_firms)
    ncp = length(col_people)
    default_name = mode == :principal ?
        "principal_f$(nrf)_p$(nrp)" :
        "rect_rf$(nrf)_rp$(nrp)_cf$(ncf)_cp$(ncp)"

    return QuerySpec(
        get(flags, "--name", default_name),
        mode,
        row_firms,
        row_people,
        col_firms,
        col_people,
    )
end

function build_row_col_refs(query::QuerySpec, firms::Vector, people::Vector, n_f::Int)
    row_refs = vcat(
        resolve_entities(query.row_firms, firms, :firm, 0),
        resolve_entities(query.row_people, people, :person, n_f),
    )
    col_refs = vcat(
        resolve_entities(query.col_firms, firms, :firm, 0),
        resolve_entities(query.col_people, people, :person, n_f),
    )

    isempty(row_refs) && error("Requested row set is empty after resolution.")
    isempty(col_refs) && error("Requested column set is empty after resolution.")

    return (row_refs=row_refs, col_refs=col_refs)
end

function extract_by_columns(F, n::Int, row_idx::Vector{Int}, col_idx::Vector{Int}; batch_size::Int=4)
    batch_size >= 1 || error("--batch-size must be >= 1")
    out = Matrix{Float64}(undef, length(row_idx), length(col_idx))
    rhs = zeros(Float64, n, min(batch_size, length(col_idx)))

    start = 1
    while start <= length(col_idx)
        stop = min(start + batch_size - 1, length(col_idx))
        width = stop - start + 1
        rhs_view = view(rhs, :, 1:width)
        fill!(rhs_view, 0.0)

        @inbounds for local_j in 1:width
            rhs_view[col_idx[start + local_j - 1], local_j] = 1.0
        end

        sol = F \ rhs_view
        out[:, start:stop] .= sol[row_idx, 1:width]
        start = stop + 1
    end

    return out
end

function extract_submatrix(F, n::Int, row_idx::Vector{Int}, col_idx::Vector{Int}; batch_size::Int=4)
    if length(col_idx) <= length(row_idx)
        return extract_by_columns(F, n, row_idx, col_idx; batch_size=batch_size)
    else
        return Matrix(transpose(extract_by_columns(F, n, col_idx, row_idx; batch_size=batch_size)))
    end
end

function refs_dataframe(refs::Vector{EntityRef}, pos_name::Symbol)
    return DataFrame(
        pos_name => collect(1:length(refs)),
        :label => [entity_label(ref) for ref in refs],
        :side => [String(ref.side) for ref in refs],
        :id => [string(ref.id) for ref in refs],
        :local_index => [ref.local_index for ref in refs],
        :latent_index => [ref.latent_index for ref in refs],
    )
end

function dense_dataframe(block::Matrix{Float64}, row_refs::Vector{EntityRef})
    df = DataFrame(
        row_pos = collect(1:length(row_refs)),
        row_label = [entity_label(ref) for ref in row_refs],
        row_side = [String(ref.side) for ref in row_refs],
        row_id = [string(ref.id) for ref in row_refs],
        row_local_index = [ref.local_index for ref in row_refs],
        row_latent_index = [ref.latent_index for ref in row_refs],
    )

    for j in 1:size(block, 2)
        df[!, Symbol("c$(j)")] = block[:, j]
    end
    return df
end

function long_dataframe(block::Matrix{Float64})
    nr, nc = size(block)
    total = nr * nc
    row_pos = Vector{Int}(undef, total)
    col_pos = Vector{Int}(undef, total)
    value   = Vector{Float64}(undef, total)
    idx = 1
    @inbounds for j in 1:nc
        for i in 1:nr
            row_pos[idx] = i
            col_pos[idx] = j
            value[idx] = block[i, j]
            idx += 1
        end
    end
    return DataFrame(row_pos=row_pos, col_pos=col_pos, value=value)
end

function default_output_dir(estimates_path::String, query_name::String)
    normalized = replace(normpath(estimates_path), "\\" => "/")
    parts = split(normalized, '/')
    out_idx = findfirst(==("output"), parts)
    label = safe_slug(query_name)
    estimate_stem = splitext(basename(estimates_path))[1]

    if out_idx !== nothing && out_idx < length(parts)
        rel_parts = parts[(out_idx + 1):(end - 1)]
        if estimate_stem == "estimates"
            return joinpath(PROJECT_ROOT, "output", "gmrf_cov", rel_parts..., label)
        else
            return joinpath(PROJECT_ROOT, "output", "gmrf_cov", rel_parts..., safe_slug(estimate_stem), label)
        end
    end

    return joinpath(dirname(estimates_path), "gmrf_cov", label)
end

function write_outputs(
    out_dir::String,
    meta,
    query::QuerySpec,
    row_refs::Vector{EntityRef},
    col_refs::Vector{EntityRef},
    blocks::Dict{Symbol,Matrix{Float64}};
    units::Symbol,
    write_long::Bool
)
    mkpath(out_dir)

    row_map_path = joinpath(out_dir, "rows.csv")
    col_map_path = joinpath(out_dir, "cols.csv")
    CSV.write(row_map_path, refs_dataframe(row_refs, :row_pos))
    CSV.write(col_map_path, refs_dataframe(col_refs, :col_pos))

    output_files = Dict{String,Any}(
        "rows_csv" => row_map_path,
        "cols_csv" => col_map_path,
    )

    for (kind, block) in sort!(collect(blocks); by=x -> String(x[1]))
        base = String(kind)
        dense_path = joinpath(out_dir, "$(base)_cov_dense.csv")
        CSV.write(dense_path, dense_dataframe(block, row_refs))
        output_files["$(base)_dense_csv"] = dense_path

        if write_long
            long_path = joinpath(out_dir, "$(base)_cov_long.parquet")
            Parquet2.writefile(long_path, long_dataframe(block))
            output_files["$(base)_long_parquet"] = long_path
        end
    end

    metadata = Dict(
        "query_name" => query.name,
        "query_mode" => String(query.mode),
        "units" => String(units),
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
        "row_count" => length(row_refs),
        "col_count" => length(col_refs),
        "matrices_written" => sort!(String.(collect(keys(blocks)))),
        "output_files" => output_files,
    )

    metadata_path = joinpath(out_dir, "metadata.json")
    open(metadata_path, "w") do io
        JSON.print(io, metadata, 2)
    end

    return merge(output_files, Dict("metadata_json" => metadata_path))
end

function usage()
    println(
        "Usage:\n" *
        "  julia --project=. src/estimate/gmrf_extract_cov_submatrix.jl <path/to/estimates.txt> [flags]\n" *
        "\n" *
        "Block selection:\n" *
        "  --firms=1,2,3           Principal block over these firms\n" *
        "  --persons=10,11         Principal block over these persons\n" *
        "  --row-firms=...         Explicit row firms\n" *
        "  --row-persons=...       Explicit row persons\n" *
        "  --col-firms=...         Explicit column firms\n" *
        "  --col-persons=...       Explicit column persons\n" *
        "\n" *
        "Other flags:\n" *
        "  --matrix=prior|posterior|both   Which covariance to extract (default: both)\n" *
        "  --units=original|scaled         Output covariance units (default: original)\n" *
        "  --batch-size=N                  RHS batch size for column solves (default: 4)\n" *
        "  --name=<label>                  Output label\n" *
        "  --output-dir=<path>             Override output directory\n" *
        "  --long                          Also write long-format Parquet matrices\n" *
        "\n" *
        "Examples:\n" *
        "  julia --project=. src/estimate/gmrf_extract_cov_submatrix.jl output/gmrfmle/kss-test-raw/kss/estimates.txt --firms=100,101 --persons=2001,2002\n" *
        "  julia --project=. src/estimate/gmrf_extract_cov_submatrix.jl output/gmrfmle/full/giant/estimates.txt --matrix=posterior --row-firms=1,2 --col-persons=10,11\n"
    )
end

function main()
    positional = filter(arg -> !startswith(arg, "--"), ARGS)
    flags_list = filter(arg -> startswith(arg, "--"), ARGS)

    if isempty(positional)
        usage()
        return 1
    end

    flags = Dict{String,String}()
    write_long = false
    for flag in flags_list
        if flag == "--long"
            write_long = true
        elseif occursin("=", flag)
            key, value = split(flag, "=", limit=2)
            flags[key] = value
        else
            error("Unknown flag: $(flag)")
        end
    end

    estimates_path = repo_path(positional[1])
    isfile(estimates_path) || error("Estimates file not found: $(estimates_path)")

    matrix_mode = Symbol(get(flags, "--matrix", "both"))
    matrix_mode in (:prior, :posterior, :both) ||
        error("--matrix must be prior, posterior, or both; got $(matrix_mode)")

    units = Symbol(get(flags, "--units", "original"))
    units in (:original, :scaled) || error("--units must be original or scaled; got $(units)")

    batch_size = parse(Int, get(flags, "--batch-size", "4"))
    batch_size >= 1 || error("--batch-size must be >= 1")

    query = build_query_spec(flags)
    output_dir = haskey(flags, "--output-dir") ?
        repo_path(flags["--output-dir"]) :
        default_output_dir(estimates_path, query.name)

    @printf("Reading estimates: %s\n", estimates_path)
    meta = parse_estimates_file(estimates_path)

    input_path = repo_path(meta.input_path)
    isfile(input_path) || error("Input parquet not found: $(input_path)")

    @printf("Reading sample parquet: %s\n", input_path)
    df = Parquet2.readfile(input_path) |> DataFrame
    @printf("Loaded rows: %d\n", nrow(df))

    if meta.maxdeg !== nothing
        df = filter_maxdeg(df, meta.maxdeg; verbose=true)
        nrow(df) > 0 || error("No rows remain after maxdeg=$(meta.maxdeg)")
    end

    prepared = prepare_data_with_ids(
        df;
        outcome=meta.outcome,
        a_weighting=meta.a_weighting,
        prior_adjacency=meta.prior_adjacency,
        obs_weighting=meta.obs_weighting,
        rho_eps=meta.rho_eps,
        verbose=true,
    )
    prep = prepared.prep
    validate_reconstructed_sample(meta, prep; firms=prepared.firms, people=prepared.people)

    refs = build_row_col_refs(query, prepared.firms, prepared.people, prep.N_F)
    row_refs = refs.row_refs
    col_refs = refs.col_refs
    row_idx = [ref.latent_index for ref in row_refs]
    col_idx = [ref.latent_index for ref in col_refs]

    @printf("Requested block: %d rows x %d cols\n", length(row_idx), length(col_idx))
    @printf("Output directory: %s\n", output_dir)

    built = build_exact_matrices(prep, meta)
    n = prep.N_F + prep.N_M
    scale = units == :original ? prep.y_std^2 : 1.0

    blocks = Dict{Symbol,Matrix{Float64}}()

    if matrix_mode in (:prior, :both)
        @printf("Factoring prior precision Q...\n")
        FQ = cholesky(Symmetric(built.Q))
        @printf("Extracting prior block...\n")
        prior_block = extract_submatrix(FQ, n, row_idx, col_idx; batch_size=batch_size)
        blocks[:prior] = prior_block .* scale
    end

    if matrix_mode in (:posterior, :both)
        @printf("Factoring posterior precision M = Q + lambda * V'V...\n")
        FM = cholesky(Symmetric(built.M))
        @printf("Extracting posterior block...\n")
        post_block = extract_submatrix(FM, n, row_idx, col_idx; batch_size=batch_size)
        blocks[:posterior] = post_block .* scale
    end

    written = write_outputs(output_dir, meta, query, row_refs, col_refs, blocks;
                            units=units, write_long=write_long)

    @printf("Wrote metadata: %s\n", written["metadata_json"])
    if haskey(written, "prior_dense_csv")
        @printf("Wrote prior block: %s\n", written["prior_dense_csv"])
    end
    if haskey(written, "posterior_dense_csv")
        @printf("Wrote posterior block: %s\n", written["posterior_dense_csv"])
    end

    return 0
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(GMRFExtractCovSubmatrix.main())
end
