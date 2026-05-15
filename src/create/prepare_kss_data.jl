#!/usr/bin/env julia
# ============================================================
# prepare_kss_data.jl
#
# Convert KSS replication package CSV data to the Parquet
# edgelist format expected by the GMRF estimators.
#
# Usage:
#   julia --project=. src/create/prepare_kss_data.jl
#
# Input:
#   data/test.csv    — 71K person-year obs (no header)
#                      col1=id, col2=firmid, col3=year, col4=y, col5-10=controls
#   data/lincom.csv  — 128K person-year obs (no header)
#                      col1=id, col2=firmid, col3=year, col4=region, col5=y, col6-9=controls
#
# Output:
#   temp/chunks/kss-test/edgelist.parquet
#   temp/chunks/kss-lincom/edgelist.parquet
# ============================================================

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Parquet2
using Statistics

function convert_test_csv(input_path::String, output_dir::String; collapse::Bool=true)
    println("Reading $input_path ...")
    df = CSV.read(input_path, DataFrame; header=false)
    println("  Loaded $(nrow(df)) rows, $(ncol(df)) columns")

    # Map columns: col1=worker_id, col2=firm_id, col3=year, col4=outcome
    rename!(df, :Column1 => :person_id,
                :Column2 => :frame_id_numeric,
                :Column3 => :year,
                :Column4 => :lnR)

    if collapse
        # Collapse to unique (person_id, frame_id_numeric) edges
        out = combine(
            groupby(df, [:frame_id_numeric, :person_id]),
            :year => length => :T,
            :lnR  => mean   => :lnR
        )
        println("  Collapsed to $(nrow(out)) unique edges")
    else
        # Keep raw person-year observations (matching KSS MATLAB input)
        out = select(df, :person_id, :frame_id_numeric, :year, :lnR)
        println("  Keeping all $(nrow(out)) person-year observations (no collapse)")
    end

    n_firms   = length(unique(out.frame_id_numeric))
    n_workers = length(unique(out.person_id))
    println("  Firms: $n_firms, Workers: $n_workers")

    mkpath(output_dir)
    out_path = joinpath(output_dir, "edgelist.parquet")
    Parquet2.writefile(out_path, out)
    println("  Wrote: $out_path")
    return out_path
end

function convert_lincom_csv(input_path::String, output_dir::String)
    println("Reading $input_path ...")
    df = CSV.read(input_path, DataFrame; header=false)
    println("  Loaded $(nrow(df)) rows, $(ncol(df)) columns")

    # Map columns: col1=worker_id, col2=firm_id, col3=year, col4=region, col5=outcome
    rename!(df, :Column1 => :person_id,
                :Column2 => :frame_id_numeric,
                :Column3 => :year,
                :Column4 => :region,
                :Column5 => :lnR)

    # Collapse to unique (person_id, frame_id_numeric) edges
    edges = combine(
        groupby(df, [:frame_id_numeric, :person_id]),
        :year => length => :T,
        :lnR  => mean   => :lnR
    )
    println("  Collapsed to $(nrow(edges)) unique edges")

    n_firms   = length(unique(edges.frame_id_numeric))
    n_workers = length(unique(edges.person_id))
    println("  Firms: $n_firms, Workers: $n_workers")

    mkpath(output_dir)
    out_path = joinpath(output_dir, "edgelist.parquet")
    Parquet2.writefile(out_path, edges)
    println("  Wrote: $out_path")
    return out_path
end

function main()
    test_csv   = "data/test.csv"
    lincom_csv = "data/lincom.csv"

    if isfile(test_csv)
        # Collapsed edges (for GMRF)
        convert_test_csv(test_csv, "temp/chunks/kss-test"; collapse=true)
        # Raw person-year observations (matches KSS MATLAB input)
        convert_test_csv(test_csv, "temp/chunks/kss-test-raw"; collapse=false)
    else
        println("WARNING: $test_csv not found, skipping")
    end

    if isfile(lincom_csv)
        convert_lincom_csv(lincom_csv, "temp/chunks/kss-lincom")
    else
        println("WARNING: $lincom_csv not found, skipping")
    end

    println("\nDone. Next steps:")
    println("  julia --project=. src/estimate/gmrfmle_exact.jl temp/chunks/kss-test/edgelist.parquet lnR --decompose")
    println("  julia --project=. src/estimate/gmrfmle_exact.jl temp/chunks/kss-lincom/edgelist.parquet lnR --decompose")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
