#!/usr/bin/env julia
# ============================================================
# kss_connected_set.jl
#
# Compute the KSS leave-one-out (match-level) connected set using
# VarianceComponentsHDFE.jl — the same pruning the KSS estimator uses.
#
# Requires Julia 1.8.x and the vchdfe project environment:
#   julia +1.8.5 --project=<vchdfe-env> src/create/kss_connected_set.jl <input.parquet> <output.parquet>
# ============================================================

function get_flag_value(args::Vector{String}, prefix::String)
    for arg in args
        startswith(arg, prefix) && return split(arg, '=', limit=2)[2]
    end
    return nothing
end

const VCHDFE_PROJECT = get_flag_value(ARGS, "--vchdfe-project=")
if VCHDFE_PROJECT !== nothing
    pushfirst!(LOAD_PATH, abspath(VCHDFE_PROJECT))
end

using Parquet2
using DataFrames
using Printf: @printf
using VarianceComponentsHDFE: VCHDFESettings, JLAAlgorithm, get_leave_one_out_set

function main()
    positional = filter(a -> !startswith(a, "--"), ARGS)
    length(positional) >= 2 || error(
        "Usage: julia +1.8.5 --project=<vchdfe-env> src/create/kss_connected_set.jl <input.parquet> <output.parquet>"
    )
    in_path, out_path = positional[1], positional[2]
    isfile(in_path) || error("Input not found: $in_path")

    @printf("Reading: %s\n", in_path)
    df = Parquet2.readfile(in_path) |> DataFrame
    @printf("Loaded: %d rows\n", nrow(df))

    hasproperty(df, :person_id)        || error("Missing column: person_id")
    hasproperty(df, :frame_id_numeric) || error("Missing column: frame_id_numeric")
    hasproperty(df, :lnR)              || error("Missing column: lnR")

    y_raw = [x isa Missing ? NaN : Float64(x) for x in df.lnR]
    keep  = isfinite.(y_raw)
    d     = df[keep, :]
    y     = Float64.(y_raw[keep])
    @printf("Non-missing lnR: %d rows\n", length(y))

    # Dense re-index (raw IDs can be up to 76M, causing infeasible graph allocations)
    person_levels = sort(unique(d.person_id))
    firm_levels   = sort(unique(d.frame_id_numeric))
    person_dense  = Int.(indexin(d.person_id, person_levels))
    firm_dense    = Int.(indexin(d.frame_id_numeric, firm_levels))

    settings = VCHDFESettings(
        leverage_algorithm=JLAAlgorithm(num_simulations=1),
        leave_out_level="match",
        print_level=1,
        first_id_display="Worker",
        first_id_display_small="worker",
        second_id_display="Firm",
        second_id_display_small="firm",
        outcome_id_display="lnR",
        outcome_id_display_small="lnR",
    )

    @printf("Computing KSS leave-one-out (match-level) connected set...\n")
    loo = get_leave_one_out_set(y, person_dense, firm_dense, settings, nothing)
    n_loo = length(loo.obs)
    @printf("LOO set: %d rows (dropped %d of %d)\n", n_loo, length(y) - n_loo, length(y))

    df_out = d[loo.obs, :]
    mkpath(dirname(out_path))
    Parquet2.writefile(out_path, df_out)
    @printf("Wrote: %s\n", out_path)
    return 0
end

exit(main())
