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

using CSV
using DataFrames
using Logging
using Parquet2
using Statistics

function convert_test_csv(input_path::String, output_dir::String; collapse::Bool=true)
    @info "Reading input" input_path
    df = CSV.read(input_path, DataFrame; header=false)
    @info "Loaded input" rows=nrow(df) columns=ncol(df)

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
        @info "Collapsed to unique edges" rows=nrow(out)
    else
        # Keep raw person-year observations (matching KSS MATLAB input)
        out = select(df, :person_id, :frame_id_numeric, :year, :lnR)
        @info "Keeping person-year observations without collapse" rows=nrow(out)
    end

    n_firms   = length(unique(out.frame_id_numeric))
    n_workers = length(unique(out.person_id))
    @info "Prepared edge data" firms=n_firms workers=n_workers

    mkpath(output_dir)
    out_path = joinpath(output_dir, "edgelist.parquet")
    Parquet2.writefile(out_path, out)
    @info "Wrote output" out_path
    return out_path
end

function convert_lincom_csv(input_path::String, output_dir::String)
    @info "Reading input" input_path
    df = CSV.read(input_path, DataFrame; header=false)
    @info "Loaded input" rows=nrow(df) columns=ncol(df)

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
    @info "Collapsed to unique edges" rows=nrow(edges)

    n_firms   = length(unique(edges.frame_id_numeric))
    n_workers = length(unique(edges.person_id))
    @info "Prepared edge data" firms=n_firms workers=n_workers

    mkpath(output_dir)
    out_path = joinpath(output_dir, "edgelist.parquet")
    Parquet2.writefile(out_path, edges)
    @info "Wrote output" out_path
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
        @warn "Input file not found; skipping" input_path=test_csv
    end

    if isfile(lincom_csv)
        convert_lincom_csv(lincom_csv, "temp/chunks/kss-lincom")
    else
        @warn "Input file not found; skipping" input_path=lincom_csv
    end

    next_step_1 = "julia --project=. src/estimate/gmrfmle_exact.jl temp/chunks/kss-test/edgelist.parquet lnR --decompose"
    next_step_2 = "julia --project=. src/estimate/gmrfmle_exact.jl temp/chunks/kss-lincom/edgelist.parquet lnR --decompose"
    @info "Done" next_step_1 next_step_2
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
