#!/usr/bin/env julia

"""
chunk.jl - Chunk edgelist data by time windows using Kezdi.jl.

Usage:
    julia chunk.jl [windows] [--obs-level=edge|personyear]

Arguments:
    windows     Window specification (default: "1990-1991,1992-1993,...")
                Options: "full" (all years), "1990-1995" (custom), or omit for defaults
    obs-level   "edge" collapses to unique firm-person edges (default).
                "personyear" keeps repeated person-year observations.

Input:
    temp/merged-panel.parquet

Output:
    temp/chunks/<obs-level>/<window_id>/edgelist.parquet when --obs-level is set.
    temp/chunks/<window_id>/edgelist.parquet when --obs-level is omitted.
        edge:       frame_id_numeric, person_id, manager_category, T, lnR, lnY, lnL
        personyear: frame_id_numeric, person_id, manager_category, year, lnR, lnY, lnL
    temp/chunks[/<obs-level>]/manifest.txt (list of all chunks)
"""

using Kezdi
using DataFrames
using Parquet2
using Logging

const DEFAULT_WINDOWS = [(1990, 1991), (1992, 1993), (1994, 1995), (1996, 1997), 
                         (1998, 1999), (2000, 2001), (2002, 2003), (2004, 2005),
                         (2006, 2007), (2008, 2009), (2010, 2011), (2012, 2013),
                         (2014, 2015), (2016, 2017), (2018, 2018)]

parse_windows(s::AbstractString) = begin
    s = strip(s)
    s == "full" && return [(:full,)]
    
    parts = split(s, ',')
    result = []
    for p in parts
        p = strip(p)
        isempty(p) && continue
        ab = split(p, '-')
        length(ab) == 2 || (@error "Bad window: '$p' (expected 'start-end')"; exit(1))
        a = parse(Int, strip(ab[1]))
        b = parse(Int, strip(ab[2]))
        a <= b || (@error "Window start > end: '$p'"; exit(1))
        push!(result, (a, b))
    end
    result
end

function parse_args(args::Vector{String})
    windows_arg = nothing
    obs_level = nothing

    for arg in args
        if startswith(arg, "--obs-level=")
            obs_level_str = lowercase(strip(split(arg, '=', limit=2)[2]))
            obs_level_str in ("edge", "personyear") ||
                (@error "Bad obs-level: '$obs_level_str' (expected edge or personyear)"; exit(1))
            obs_level = Symbol(obs_level_str)
        elseif startswith(arg, "--")
            @error "Unknown flag: $arg"
            exit(1)
        else
            windows_arg === nothing || (@error "Only one windows argument is supported"; exit(1))
            windows_arg = arg
        end
    end

    windows = windows_arg === nothing ? DEFAULT_WINDOWS : parse_windows(windows_arg)
    outroot = obs_level === nothing ? "temp/chunks" : joinpath("temp", "chunks", String(obs_level))
    obs_level = obs_level === nothing ? :edge : obs_level
    return (windows=windows, obs_level=obs_level, outroot=outroot)
end

preprocess_panel(df::DataFrame)::DataFrame = begin
    @info "Preprocessing panel..."
    @with df begin

        # Year-demean the outcomes (Kezdi handles missings by default)
        @egen lnR_yrmean = mean(lnR), by(year)
        @egen lnY_yrmean = mean(lnY), by(year)
        @egen lnL_yrmean = mean(lnL), by(year)
        
        @replace lnR = lnR - lnR_yrmean
        @replace lnY = lnY - lnY_yrmean
        @replace lnL = lnL - lnL_yrmean
        
        @drop lnR_yrmean lnY_yrmean lnL_yrmean
    end
end

collapse_edges(df::DataFrame)::DataFrame = begin
    @info "Collapsing to unique edges..."
    @with df begin
        @collapse T = rowcount(year) lnR = mean(lnR) lnY = mean(lnY) lnL = mean(lnL), by(frame_id_numeric, person_id, manager_category)
    end
end

remove_degree_mean(df::DataFrame)::DataFrame = begin
    @info "Removing degree mean..."
    @with df begin
        @egen degree_firm = rowcount(distinct(person_id)), by(frame_id_numeric)
        @egen degree_manager = rowcount(distinct(frame_id_numeric)), by(person_id)
        @replace degree_firm = 5 @if degree_firm > 5
        @replace degree_manager = 5 @if degree_manager > 5
        @egen edge_mean_lnR = mean(lnR), by(degree_firm, degree_manager)
        @replace lnR = lnR - edge_mean_lnR
        @drop edge_mean_lnR
    end
end

personyear_rows(df::DataFrame)::DataFrame = begin
    @info "Keeping person-year observations..."
    preferred = [:frame_id_numeric, :person_id, :manager_category, :year, :lnR, :lnY, :lnL]
    keep = [c for c in preferred if c in propertynames(df)]
    select(df, keep)
end

finish_chunk(df::DataFrame, obs_level::Symbol)::DataFrame =
    obs_level == :edge ? collapse_edges(df) |> remove_degree_mean :
    obs_level == :personyear ? personyear_rows(df) :
    error("Unsupported obs_level: $obs_level")

make_window_chunk(df_pre::DataFrame, ys::Int, ye::Int, obs_level::Symbol)::DataFrame = begin
    # Kezdi.jl cannot handle varnames inside closures yet
    ysf() = ys
    yef() = ye
    filtered = @with df_pre begin
        @keep @if year >= ysf()
        @keep @if year <= yef()
    end
    finish_chunk(filtered, obs_level)
end

make_full_chunk(df_pre::DataFrame, obs_level::Symbol)::DataFrame =
    finish_chunk(df_pre, obs_level)

write_chunk(outroot::String, window_id::String, chunk::DataFrame)::String = begin
    outdir = joinpath(outroot, window_id)
    mkpath(outdir)
    outpath = joinpath(outdir, "edgelist.parquet")
    Parquet2.writefile(outpath, chunk)
    outpath
end

function main()
    parsed = parse_args(ARGS)
    windows = parsed.windows
    obs_level = parsed.obs_level
    outroot = parsed.outroot

    input_path = "temp/merged-panel.parquet"

    isfile(input_path) || (@error "Input not found" input_path; exit(1))
    mkpath(outroot)
    
    @info "Reading panel" input_path obs_level=obs_level outroot=outroot
    df = DataFrame(Parquet2.readfile(input_path))
    @info "Loaded" rows=nrow(df)
    
    df_pre = preprocess_panel(df)
    
    # Write manifest
    manifest_path = joinpath(outroot, "manifest.txt")
    open(manifest_path, "w") do io
        println(io, "window_id\tpath\trows\tobs_level")
    end
    
    for w in windows
        if w[1] == :full
            @info "Building FULL chunk..."
            chunk = make_full_chunk(df_pre, obs_level)
            outpath = write_chunk(outroot, "full", chunk)
            @info "Wrote" outpath rows=nrow(chunk) obs_level=obs_level
            open(manifest_path, "a") do io
                println(io, "full\t$outpath\t$(nrow(chunk))\t$(obs_level)")
            end
        else
            ys, ye = w
            window_id = "$(ys)-$(ye)"
            @info "Building chunk" window=window_id
            chunk = make_window_chunk(df_pre, ys, ye, obs_level)
            outpath = write_chunk(outroot, window_id, chunk)
            @info "Wrote" outpath rows=nrow(chunk) obs_level=obs_level
            open(manifest_path, "a") do io
                println(io, "$window_id\t$outpath\t$(nrow(chunk))\t$(obs_level)")
            end
        end
    end
    
    @info "Done" manifest=manifest_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
