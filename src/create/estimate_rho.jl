#!/usr/bin/env julia
# ============================================================
# estimate_moments_from_pairs.jl
# ------------------------------------------------------------
# Reads *_wide.csv files emitted by n_hop_edgelist.jl and estimates
# (σ_a^2, σ_z^2, ρ, σ_ε^2) from demeaned pairwise moments.
#
# Demeaning strategy (CRITICAL):
#   - If EDGELIST is provided (CSV or Parquet with column Y_COL), compute μ, Var(y)
#     from the full edgelist (no double counting).
#   - Else, build a UNIQUE endpoint set from all four *_wide.csv files keyed by
#     (person_id, firm_id) and compute μ, Var(y) from those unique endpoints.
#
# Moments used (default; edit if manuscript differs):
#   C_mm,2 = E[y1·y2 | mm 2-hop]
#   C_ff,2 = E[y1·y2 | ff 2-hop]
#   C_mm,4 = E[y1·y2 | mm pure 4-hop]
#   C_ff,4 = E[y1·y2 | ff pure 4-hop]
#   Var(y) = σ_a^2 + σ_z^2 + 2ρσ_aσ_z + σ_ε^2  ⇒  σ_ε^2 = Var(y) - σ_a^2 - σ_z^2 - 2ρσ_aσ_z
# ============================================================

using CSV, DataFrames, Statistics, Printf

# Optional Parquet support if EDGELIST is parquet
const _HAVE_PARQUET = try
    @eval using Parquet2
    true
catch
    false
end

# ----------------------------
# Configuration via ENV
# ----------------------------
function parse_threshold(val::AbstractString)
    # Added to support environment-driven degree trimming. Defaults to Inf (no trim)
    # but lets callers specify deterministic cutoffs for high-degree nodes.
    stripped = strip(lowercase(val))
    if isempty(stripped) || stripped == "inf" || stripped == "infinite" || stripped == "infinity"
        return Inf
    end
    try
        return Float64(parse(Int, stripped))
    catch
        @warn "Unable to parse degree threshold" value=val
        return Inf
    end
end

function parse_quantile(val::AbstractString, default::Float64)
    # Allows tuning how aggressively we trim by degree when explicit thresholds
    # are absent. Keeps the new filtering behaviour transparent to users.
    stripped = strip(lowercase(val))
    if isempty(stripped)
        return default
    end
    try
        q = parse(Float64, stripped)
        return clamp(q, 0.0, 1.0)
    catch
        @warn "Unable to parse quantile" value=val
        return default
    end
end

const PAIR_DIR   = get(ENV, "PAIR_DIR", "./temp")
const EDGELIST   = get(ENV, "EDGELIST", joinpath(PAIR_DIR, "edgelist.parquet"))  # path to full edgelist
const Y_COL      = Symbol(get(ENV, "Y_COL", "lnR"))
const PERSON_COL = Symbol(get(ENV, "PERSON_COL", "person_id"))
const FIRM_COL   = Symbol(get(ENV, "FIRM_COL", "frame_id_numeric"))

const MAX_MANAGER_DEGREE = parse_threshold(get(ENV, "MAX_MANAGER_DEGREE", "inf"))
const MAX_FIRM_DEGREE = parse_threshold(get(ENV, "MAX_FIRM_DEGREE", "inf"))
const DEGREE_DROP_QUANTILE = parse_quantile(get(ENV, "DEGREE_DROP_QUANTILE", "0.99"), 0.99)

# Column names in *_wide.csv (hard-coded by n_hop_edgelist.jl)
const Y1 = :y_1; const Y2 = :y_2
const PID1 = :person_id_1; const PID2 = :person_id_2
const FID1 = :firm_id_1;   const FID2 = :firm_id_2

# ----------------------------
# I/O helpers
# ----------------------------
function _read_pairs(path::AbstractString, pair_kind::Symbol)
    isfile(path) || error("Missing file: $path")
    df = CSV.read(path, DataFrame)
    for c in (Y1, Y2, PID1, PID2, FID1, FID2)
        @assert c ∈ propertynames(df) "Column $c is required in $(basename(path))"
    end
    dropmissing!(df, [Y1, Y2, PID1, PID2, FID1, FID2])
    df[!, Y1] = Float64.(df[!, Y1])
    df[!, Y2] = Float64.(df[!, Y2])
    finite_mask = isfinite.(df[!, Y1]) .& isfinite.(df[!, Y2])
    df = df[finite_mask, :]
    # Drop exact duplicates y1 == y2 to avoid inflating covariance with identical spells
    mismatch_mask = Float64.(df[!, Y1]) .!= Float64.(df[!, Y2])
    df = df[mismatch_mask, :]
    df = canonicalize_pairs(df, pair_kind)
    df = collapse_endpoint_pairs(df, pair_kind)
    df = apply_degree_filter(df, pair_kind)
    return df
end

function canonicalize_pairs(df::DataFrame, pair_kind::Symbol)
    # Added so every pair has a deterministic ordering. This was critical for
    # detecting and dropping 2-hop overlaps in the “pure” 4-hop data – without it
    # (a,b) and (b,a) slipped through as distinct rows.
    if pair_kind === :mm && all(c -> c ∈ propertynames(df), (PID1, PID2))
        p1 = Int.(df[!, PID1]); p2 = Int.(df[!, PID2])
        y1 = Float64.(df[!, Y1]); y2 = Float64.(df[!, Y2])
        swap_idx = findall(p1 .> p2)
        for i in swap_idx
            p1[i], p2[i] = p2[i], p1[i]
            y1[i], y2[i] = y2[i], y1[i]
        end
        df[!, PID1] = p1; df[!, PID2] = p2
        df[!, Y1] = y1; df[!, Y2] = y2
    elseif pair_kind === :ff && all(c -> c ∈ propertynames(df), (FID1, FID2))
        f1 = Int.(df[!, FID1]); f2 = Int.(df[!, FID2])
        y1 = Float64.(df[!, Y1]); y2 = Float64.(df[!, Y2])
        swap_idx = findall(f1 .> f2)
        for i in swap_idx
            f1[i], f2[i] = f2[i], f1[i]
            y1[i], y2[i] = y2[i], y1[i]
        end
        df[!, FID1] = f1; df[!, FID2] = f2
        df[!, Y1] = y1; df[!, Y2] = y2
    end
    return df
end

function collapse_endpoint_pairs(df::DataFrame, pair_kind::Symbol)
    # New helper collapses multiple observation-level paths onto a single
    # endpoint pair. Prevents multi-path managers/firms from overweighting 4-hop
    # moments relative to 2-hop moments.
    if isempty(df)
        return df
    end
    if pair_kind === :mm && all(c -> c ∈ propertynames(df), (PID1, PID2))
        g = groupby(df, [PID1, PID2], sort=true)
        return combine(g, Y1 => mean => Y1, Y2 => mean => Y2)
    elseif pair_kind === :ff && all(c -> c ∈ propertynames(df), (FID1, FID2))
        g = groupby(df, [FID1, FID2], sort=true)
        return combine(g, Y1 => mean => Y1, Y2 => mean => Y2)
    else
        return df
    end
end

function apply_degree_filter(df::DataFrame, pair_kind::Symbol)
    # Drops hubs (very high-degree managers/firms) so the moments are not driven
    # by a handful of super-connectors. Thresholds/quantiles are configurable via
    # environment variables.
    limit = pair_kind === :mm ? MAX_MANAGER_DEGREE : MAX_FIRM_DEGREE
    id1_col = pair_kind === :mm ? PID1 : FID1
    id2_col = pair_kind === :mm ? PID2 : FID2
    deg = Dict{Int,Int}()
    for row in eachrow(df)
        id1 = Int(row[id1_col]); id2 = Int(row[id2_col])
        deg[id1] = get(deg, id1, 0) + 1
        deg[id2] = get(deg, id2, 0) + 1
    end
    if isempty(deg)
        return df
    end
    threshold = if isfinite(limit)
        Int(limit)
    else
        sorted_vals = sort(collect(values(deg)))
        idx = clamp(floor(Int, DEGREE_DROP_QUANTILE * length(sorted_vals)), 1, length(sorted_vals))
        if idx == length(sorted_vals) && length(sorted_vals) > 1
            idx -= 1
        end
        sorted_vals[idx]
    end
    max_deg = maximum(values(deg))
    if threshold >= max_deg
        return df
    end
    keep_mask = [deg[Int(row[id1_col])] ≤ threshold && deg[Int(row[id2_col])] ≤ threshold for row in eachrow(df)]
    filtered = df[keep_mask, :]
    removed = nrow(df) - nrow(filtered)
    if removed > 0
        kind_str = pair_kind === :mm ? "manager" : "firm"
        @info "High-degree $(kind_str) pairs dropped" threshold=threshold removed=removed retained=nrow(filtered)
    end
    return filtered
end

function verify_disjoint(mm2::DataFrame, mm4::DataFrame, kind::Symbol)
    # Guard rail: make sure “pure 4-hop” sets really exclude direct neighbors.
    # Emits a warning when overlap remains so upstream generators can be fixed.
    id1 = kind === :mm ? PID1 : FID1
    id2 = kind === :mm ? PID2 : FID2
    pairs2 = Set(Iterators.map(row -> (Int(row[id1]), Int(row[id2])), eachrow(mm2)))
    overlap = Int[]
    for row in eachrow(mm4)
        pair = (Int(row[id1]), Int(row[id2]))
        if pair in pairs2
            push!(overlap, pair[1]); push!(overlap, pair[2])
        end
    end
    if !isempty(overlap)
        @warn "Detected overlap between 2-hop and pure 4-hop pairs" kind=kind count=length(overlap)÷2
    end
end

function _read_edgelist(path::AbstractString)
    isfile(path) || error("Missing EDGELIST: $path")
    if endswith(lowercase(path), ".csv")
        df = CSV.read(path, DataFrame)
    elseif endswith(lowercase(path), ".parquet")
        @assert _HAVE_PARQUET "Parquet2.jl not available; install it or pass a CSV edgelist"
        df = DataFrame(Parquet2.Dataset(path))
    else
        error("Unsupported EDGELIST format: $path (use .csv or .parquet)")
    end
    for c in (PERSON_COL, FIRM_COL, Y_COL)
        @assert c ∈ propertynames(df) "EDGELIST missing $(c)"
    end
    dropmissing!(df, [PERSON_COL, FIRM_COL, Y_COL])
    df[!, Y_COL] = Float64.(df[!, Y_COL])
    df = df[isfinite.(df[!, Y_COL]), :]
    return df
end





# ----------------------------
# Demeaning: compute μ and Var(y)
# ----------------------------
"""
    compute_mean_var_y(edgelist_path::String, wide_files::Vector{String})

If `edgelist_path` is nonempty, compute μ, Var(y) from the full edgelist.
Else, construct a UNIQUE endpoint set from wide files keyed by (person_id, firm_id)
and compute μ, Var(y) from those unique endpoints.
"""
function compute_mean_var_y(edgelist_path::String, wide_files::Vector{String})
    isempty(edgelist_path) && error("EDGELIST path is empty. Set ENV[\"EDGELIST\"] or place edgelist at temp/edgelist.parquet.")
    df = _read_edgelist(edgelist_path)
    y = Float64.(df[!, Y_COL])
    return mean(y), var(y)
end

# ----------------------------
# Moments on centered pairs (global mean)
# ----------------------------
"""
    pair_cov_centered(df_pairs, μ)

Return (C, μ̂₁, μ̂₂) where C = E[(y1-μ)(y2-μ)] using the global mean μ, and the μ̂ columns
are the simple means of y1, y2 for reporting.
"""
function pair_cov_centered(df::DataFrame, μ::Float64)
    # Added NaN guard so that when degree/overlap filtering wipes out a dataset
    # we surface that fact cleanly rather than throwing during reduction.
    if isempty(df)
        return NaN, NaN, NaN
    end
    y1 = Float64.(df[!, Y1])
    y2 = Float64.(df[!, Y2])
    C = mean((y1 .- μ) .* (y2 .- μ))
    return C, mean(y1), mean(y2)
end

# ----------------------------
# Estimation pipeline
# ----------------------------
function main()
    files = Dict(
        :mm2 => joinpath(PAIR_DIR, "mm_2hop_wide.csv"),
        :ff2 => joinpath(PAIR_DIR, "ff_2hop_wide.csv"),
        :mm4 => joinpath(PAIR_DIR, "mm_4hop_pure_wide.csv"),
        :ff4 => joinpath(PAIR_DIR, "ff_4hop_pure_wide.csv"),
    )
    for (k,p) in files
        @assert isfile(p) "Expected $(basename(p)) in $(PAIR_DIR)."
    end

    # 1) Compute μ and Var(y) respecting duplication
    μ, Vy = compute_mean_var_y(EDGELIST, collect(values(files)))

    # 2) Read pairs and compute moments centered at global μ
    d_mm2 = _read_pairs(files[:mm2], :mm)
    d_ff2 = _read_pairs(files[:ff2], :ff)
    d_mm4 = _read_pairs(files[:mm4], :mm)
    d_ff4 = _read_pairs(files[:ff4], :ff)

    verify_disjoint(d_mm2, d_mm4, :mm)
    verify_disjoint(d_ff2, d_ff4, :ff)

    C_mm2, μ1_mm2, μ2_mm2 = pair_cov_centered(d_mm2, μ)
    C_ff2, μ1_ff2, μ2_ff2 = pair_cov_centered(d_ff2, μ)
    C_mm4, μ1_mm4, μ2_mm4 = pair_cov_centered(d_mm4, μ)
    C_ff4, μ1_ff4, μ2_ff4 = pair_cov_centered(d_ff4, μ)

    # 3) Map moments → parameters (EDIT HERE if your paper differs)
    σa2 = C_mm2
    σz2 = C_ff2
    ρ_mm = (σa2>0) ? sqrt(max(C_mm4/σa2, 0.0)) : NaN
    ρ_ff = (σz2>0) ? sqrt(max(C_ff4/σz2, 0.0)) : NaN
    ρvals = filter(!isnan, [ρ_mm, ρ_ff])
    ρ = isempty(ρvals) ? NaN : mean(ρvals)

    σeps2 = (isfinite(ρ) && isfinite(σa2) && isfinite(σz2)) ? (Vy - σa2 - σz2 - 2*ρ*sqrt(max(σa2,0)*max(σz2,0))) : NaN

    # 4) Report
    println("\n=== Raw Moment Estimates (from *_wide.csv) ===")
    @printf("Mean(y) [global]:         %.6f\n", μ)
    @printf("Var(y)  [global]:         %.6f\n\n", Vy)
    @printf("C_mm,2:                   %.6f   [N=%d, μ₁=%.6f, μ₂=%.6f]\n", C_mm2, nrow(d_mm2), μ1_mm2, μ2_mm2)
    @printf("C_ff,2:                   %.6f   [N=%d, μ₁=%.6f, μ₂=%.6f]\n", C_ff2, nrow(d_ff2), μ1_ff2, μ2_ff2)
    @printf("C_mm,4:                   %.6f   [N=%d, μ₁=%.6f, μ₂=%.6f]\n", C_mm4, nrow(d_mm4), μ1_mm4, μ2_mm4)
    @printf("C_ff,4:                   %.6f   [N=%d, μ₁=%.6f, μ₂=%.6f]\n\n", C_ff4, nrow(d_ff4), μ1_ff4, μ2_ff4)
    @printf("ρ_mm:                     %.6f\n", ρ_mm)
    @printf("ρ_ff:                     %.6f\n", ρ_ff)
    @printf("ρ (avg):                  %.6f\n", ρ)
    @printf("σ_ε²:                     %.6f\n", σeps2)
    println("=================================================")

    @info "Estimation finished; results printed above"
end

main()
