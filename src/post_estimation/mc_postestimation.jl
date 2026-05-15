#!/usr/bin/env julia
# ============================================================
# src/post_estimation/mc_postestimation.jl
#
# Summarize Monte Carlo estimates produced by the standard
# estimator pattern rules.  Scans each estimator's output tree
# under output/<estimator>/mc/<topo>/estimates.txt, joins with
# the truth values in temp/samples/mc/<topo>/metadata.txt, and
# prints a comparison table.
#
# Estimators scanned:
#   output/gmrfmle/mc/<topo>/estimates.txt              (degree Q)
#   output/gmrfmle-unweighted/mc/<topo>/estimates.txt   (unweighted Q)
#   output/gmrfmle-vs/mc/<topo>/estimates.txt           (variance-stable Q)
#   output/akm_pytwoway/mc/<topo>/estimates.txt         (AKM/KSS)
#
# Output:
#   output/monte_carlo/summary.csv
#   stdout table
#
# Usage:
#   julia --project=. src/post_estimation/mc_postestimation.jl
# ============================================================

using Pkg
Pkg.activate(".")

using DataFrames
using CSV
using Printf: @printf, @sprintf

const ESTIMATORS = [
    (name="gmrfmle",             dir="output/gmrfmle"),
    (name="gmrfmle-unweighted",  dir="output/gmrfmle-unweighted"),
    (name="gmrfmle-vs",          dir="output/gmrfmle-vs"),
    (name="akm_pytwoway",        dir="output/akm_pytwoway"),
]

const SAMPLES_ROOT = joinpath("temp", "samples", "mc")

# ──────────────────────────────────────────────────────────────────
# 1) Parsers
# ──────────────────────────────────────────────────────────────────

"""
Read a simple key=value metadata file into a Dict{String,String}.
Skips blank lines and lines that do not contain '='.
"""
function read_kv(path::AbstractString)::Dict{String,String}
    kv = Dict{String,String}()
    isfile(path) || return kv
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || !occursin('=', line)) && continue
        k, v = split(line, '=', limit=2)
        kv[strip(k)] = strip(v)
    end
    return kv
end

"""
Parse an estimates.txt file from current gmrfmle outputs or AKM/KSS fallback.
Returns a NamedTuple with fields rho, sigma_a, sigma_z, sigma_eps (Float64 or NaN).

For GMRF MLE outputs, parse only the structural estimates block so posterior
summary values do not overwrite the MLE parameters. Non-GMRF files retain the
legacy whole-file scan used for AKM/KSS outputs.

Recognized line prefixes inside the selected block (leading whitespace ignored):
    rho        = <value>
    sigma_a    = <value>
    sigma_z    = <value>
    sigma_eps  = <value>
"""
function _parse_estimate_lines(lines)
    rho = sigma_a = sigma_z = sigma_eps = NaN
    for line in lines
        m = match(r"^\s*rho\s*=\s*([-+.\deE]+)", line)
        m !== nothing && (rho       = parse(Float64, m.captures[1]); continue)
        m = match(r"^\s*sigma_a\s*=\s*([-+.\deE]+)", line)
        m !== nothing && (sigma_a   = parse(Float64, m.captures[1]); continue)
        m = match(r"^\s*sigma_z\s*=\s*([-+.\deE]+)", line)
        m !== nothing && (sigma_z   = parse(Float64, m.captures[1]); continue)
        m = match(r"^\s*sigma_eps\s*=\s*([-+.\deE]+)", line)
        m !== nothing && (sigma_eps = parse(Float64, m.captures[1]); continue)
    end
    return (; rho, sigma_a, sigma_z, sigma_eps)
end

function parse_estimates(path::AbstractString)
    isfile(path) || return (; rho=NaN, sigma_a=NaN, sigma_z=NaN, sigma_eps=NaN)
    lines = readlines(path)

    if !isempty(lines) && startswith(strip(first(lines)), "GMRF MLE Estimation")
        structural_start = findfirst(i -> occursin("Estimates (structural units)", lines[i]), eachindex(lines))
        structural_start === nothing && return (; rho=NaN, sigma_a=NaN, sigma_z=NaN, sigma_eps=NaN)
        block_end = nothing
        for i in (structural_start + 1):lastindex(lines)
            s = strip(lines[i])
            if startswith(s, "Prior Variance Decomposition") ||
               startswith(s, "Posterior Variance Decomposition") ||
               startswith(s, "AKM-Comparable Summary") ||
               startswith(s, "Target Posterior Summary")
                block_end = i
                break
            end
        end
        stop_idx = block_end === nothing ? lastindex(lines) : block_end - 1
        return _parse_estimate_lines(lines[(structural_start + 1):stop_idx])
    end

    return _parse_estimate_lines(lines)
end

# ──────────────────────────────────────────────────────────────────
# 2) Topology ordering for display
# ──────────────────────────────────────────────────────────────────

const TOPO_ORDER = [
    "path", "tree", "tree-2", "tree-3",
    "cycle",
    "regular-3", "regular-10", "regular-20", "regular-50",
    "erdos-renyi-0p05", "erdos-renyi-0p1", "erdos-renyi-0p2",
    "erdos-renyi-0p3", "erdos-renyi-0p5",
    "complete",
]

function topo_rank(label::AbstractString)
    idx = findfirst(==(label), TOPO_ORDER)
    return idx === nothing ? length(TOPO_ORDER) + 1 : idx
end

# ──────────────────────────────────────────────────────────────────
# 3) Assemble the summary DataFrame
# ──────────────────────────────────────────────────────────────────

function discover_topologies()::Vector{String}
    isdir(SAMPLES_ROOT) || return String[]
    topos = String[]
    for d in readdir(SAMPLES_ROOT)
        isdir(joinpath(SAMPLES_ROOT, d)) || continue
        isfile(joinpath(SAMPLES_ROOT, d, "metadata.txt")) || continue
        push!(topos, d)
    end
    sort!(topos; by=topo_rank)
    return topos
end

function build_summary()::DataFrame
    topos = discover_topologies()
    isempty(topos) && error("No MC topologies found in $SAMPLES_ROOT/. Run `make mc-simulate` first.")

    rows = NamedTuple[]
    for topo in topos
        meta = read_kv(joinpath(SAMPLES_ROOT, topo, "metadata.txt"))
        rho_true        = parse(Float64, get(meta, "rho",            "NaN"))
        sigma_a_true    = parse(Float64, get(meta, "sigma_a",        "NaN"))
        sigma_z_true    = parse(Float64, get(meta, "sigma_z",        "NaN"))
        sigma_eps_true  = parse(Float64, get(meta, "sigma_eps",      "NaN"))
        rho_akm_target  = parse(Float64, get(meta, "rho_akm_target", "NaN"))
        nf              = parse(Int,     get(meta, "nf",             "0"))
        nm              = parse(Int,     get(meta, "nm",             "0"))
        k_edges         = parse(Int,     get(meta, "k_edges",        "0"))
        n_bridges       = parse(Int,     get(meta, "n_bridges",      "0"))

        for est in ESTIMATORS
            est_path = joinpath(est.dir, "mc", topo, "estimates.txt")
            e = parse_estimates(est_path)
            push!(rows, (
                topology       = topo,
                estimator      = est.name,
                nf             = nf,
                nm             = nm,
                k_edges        = k_edges,
                n_bridges      = n_bridges,
                rho_true       = rho_true,
                sigma_a_true   = sigma_a_true,
                sigma_z_true   = sigma_z_true,
                sigma_eps_true = sigma_eps_true,
                rho_akm_target = rho_akm_target,
                rho            = e.rho,
                sigma_a        = e.sigma_a,
                sigma_z        = e.sigma_z,
                sigma_eps      = e.sigma_eps,
                found          = isfile(est_path),
            ))
        end
    end
    return DataFrame(rows)
end

# ──────────────────────────────────────────────────────────────────
# 4) Pretty-print
# ──────────────────────────────────────────────────────────────────

fmt(x::Real) = isnan(x) ? "     .  " : @sprintf("%8.4f", x)
fmt(::Missing) = "     .  "

function print_table(df::DataFrame)
    @printf("\n=== Monte Carlo Summary (Laplacian-normalized GMRF DGP) ===\n")
    @printf("Samples: %d topologies × %d estimators = %d estimates\n",
            length(unique(df.topology)), length(ESTIMATORS), nrow(df))

    for topo in unique(df.topology)
        sub = filter(:topology => ==(topo), df)
        first_row = first(sub)
        @printf("\n── %s  (N_F=%d, N_M=%d, K=%d, bridges=%d) ──\n",
                topo, first_row.nf, first_row.nm, first_row.k_edges, first_row.n_bridges)
        @printf("%-22s  %8s  %8s  %8s  %8s\n",
                "param", "truth", "rho", "sigma_a", "sigma_z")
        @printf("  truth                       %s  %s  %s\n",
                fmt(first_row.rho_true),
                fmt(first_row.sigma_a_true),
                fmt(first_row.sigma_z_true))
        @printf("  (rho_akm_target = %s  sigma_eps_true = %s)\n",
                fmt(first_row.rho_akm_target), fmt(first_row.sigma_eps_true))
        for r in eachrow(sub)
            mark = r.found ? " " : "×"
            @printf("  %-22s  %s  %s  %s  %s  sigma_eps=%s %s\n",
                    r.estimator, " ", fmt(r.rho), fmt(r.sigma_a),
                    fmt(r.sigma_z), fmt(r.sigma_eps), mark)
        end
    end
end

# ──────────────────────────────────────────────────────────────────
# 5) Main
# ──────────────────────────────────────────────────────────────────

function main()
    df = build_summary()

    outdir = joinpath("output", "monte_carlo")
    mkpath(outdir)
    outpath = joinpath(outdir, "summary.csv")
    CSV.write(outpath, df)
    @printf("Wrote: %s (%d rows)\n", outpath, nrow(df))

    print_table(df)

    n_missing = count(.!df.found)
    if n_missing > 0
        @printf("\n[NOTE] %d/%d estimator outputs missing (marked ×). Run `make mc-estimate`.\n",
                n_missing, nrow(df))
    end

    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
