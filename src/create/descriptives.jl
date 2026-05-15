#!/usr/bin/env julia
# ============================================================
# descriptives.jl
# ============================================================

# --- Headless plotting (important on servers) ---
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using DataFrames
using Parquet2
using SparseArrays
using Graphs
using Statistics
using Plots

# ----------------------------
# Configuration via ENV
# ----------------------------
const MERGED_PANEL_PATH = get(ENV, "MERGED_PANEL_PATH", "temp/merged-panel.parquet")
const Y_COL = Symbol(get(ENV, "Y_COL", "lnR"))
const PERSON_COL = Symbol(get(ENV, "PERSON_COL", "person_id"))
const FIRM_COL = Symbol(get(ENV, "FIRM_COL", "frame_id_numeric"))

const WINDOW_SIZE_YEARS = parse(Int, get(ENV, "WINDOW_SIZE_YEARS", "3"))
const WINDOW_STEP_YEARS = parse(Int, get(ENV, "WINDOW_STEP_YEARS", "1"))
const START_YEAR = get(ENV, "START_YEAR", "2004")
const END_YEAR = get(ENV, "END_YEAR", "2020")

# Output directories (requested defaults)
const FIG_DIR  = get(ENV, "FIG_DIR", "output/fig")
const DESC_DIR = get(ENV, "DESC_DIR", "output/descriptives")

# ----------------------------
# Helpers
# ----------------------------
function safe_var(x)
    n = length(x)
    return n < 2 ? missing : var(skipmissing(x))
end

function build_windows(years::Vector{Int})
    y_min = isempty(START_YEAR) ? minimum(years) : max(minimum(years), parse(Int, START_YEAR))
    y_max = isempty(END_YEAR) ? maximum(years) : min(maximum(years), parse(Int, END_YEAR))
    starts = collect(y_min:WINDOW_STEP_YEARS:(y_max - WINDOW_SIZE_YEARS + 1))
    return [(s, min(s + WINDOW_SIZE_YEARS - 1, y_max)) for s in starts if s + WINDOW_SIZE_YEARS - 1 <= y_max]
end

function build_incidence_matrix(df::DataFrame; person_col::Symbol, firm_col::Symbol)
    person_ids = unique!(Int.(copy(df[!, person_col])))
    firm_ids = unique!(Int.(copy(df[!, firm_col])))
    pid_to_row = Dict{Int,Int}(p => i for (i, p) in enumerate(person_ids))
    fid_to_col = Dict{Int,Int}(f => j for (j, f) in enumerate(firm_ids))

    rows = Vector{Int}(undef, nrow(df))
    cols = Vector{Int}(undef, nrow(df))
    @inbounds for (k, r) in enumerate(eachrow(df))
        rows[k] = pid_to_row[Int(r[person_col])]
        cols[k] = fid_to_col[Int(r[firm_col])]
    end
    vals = ones(Int, length(rows))
    B = sparse(rows, cols, vals, length(person_ids), length(firm_ids))
    B .= sign.(B)
    return B, person_ids, firm_ids
end

function projection_adjacencies(B::SparseMatrixCSC{Int,Int})
    A_mm = (B * B') .> 0
    A_ff = (B' * B) .> 0
    for i in 1:size(A_mm, 1)
        A_mm[i, i] = false
    end
    for j in 1:size(A_ff, 1)
        A_ff[j, j] = false
    end
    return A_mm, A_ff
end

function component_summary(A::SparseMatrixCSC{Bool,Int})
    G = SimpleGraph(A)
    comps = connected_components(G)
    num_comps = length(comps)
    lcc = isempty(comps) ? 0 : maximum(length.(comps))
    deg = degree.(Ref(G), 1:nv(G))
    isolates = count(==(0), deg)
    return num_comps, lcc, isolates
end

function degree_map(ids::Vector{Int}, deg_vec)::Dict{Int,Int}
    d = Dict{Int,Int}()
    @inbounds for (i, id) in enumerate(ids)
        d[id] = Int(deg_vec[i])
    end
    return d
end

function variance_by_degree(df::DataFrame, id_col::Symbol, y_col::Symbol, deg_map::Dict{Int,Int};
                            window_start::Int, window_end::Int, degree_type::String)
    df_local = df[:, [id_col, y_col]]
    df_local[!, :degree] = get.(Ref(deg_map), Int.(df_local[!, id_col]), 0)
    g = groupby(df_local, :degree, sort=true)
    out = combine(g,
                  :degree => first => :degree,
                  y_col => length => :n_obs,
                  y_col => safe_var => :var_y)
    out[!, :window_start] .= window_start
    out[!, :window_end] .= window_end
    out[!, :degree_type] .= degree_type
    return out[:, [:window_start, :window_end, :degree_type, :degree, :n_obs, :var_y]]
end

function ensure_dir(path::String)
    if !isdir(path)
        mkpath(path)
    end
end

# ============================================================
# CCDF utilities + snapshot plotting (added)
# ============================================================

# Returns x-values and CCDF(x)=P(X >= x), evaluated at unique observed x (positive only by default).
function ccdf_xy(v::AbstractVector; drop_nonpositive::Bool=true)
    w = collect(skipmissing(v))
    w = Float64.(w)
    w = filter(isfinite, w)
    if drop_nonpositive
        w = w[w .> 0]
    end
    isempty(w) && return (Float64[], Float64[])

    s = sort(w)
    n = length(s)

    xs = Float64[]
    counts = Float64[]
    i = 1
    while i <= n
        x = s[i]
        j = i
        while j <= n && s[j] == x
            j += 1
        end
        push!(xs, x)
        push!(counts, j - i)
        i = j
    end

    surv = reverse(cumsum(reverse(counts))) ./ n
    return xs, surv
end

function write_ccdf_parquet(vals::AbstractVector; outpath::String, xname::Symbol)
    x, y = ccdf_xy(vals; drop_nonpositive=true)
    isempty(x) && return false
    df = DataFrame(xname => x, :ccdf => y)
    ensure_dir(dirname(outpath))
    Parquet2.writefile(outpath, df)
    return true
end

function save_ccdf_plot(vals::AbstractVector;
                        fig_path::String,
                        table_path::String,
                        title::String,
                        xlabel::String,
                        xname::Symbol)
    x, y = ccdf_xy(vals; drop_nonpositive=true)
    if isempty(x)
        println("No positive values for CCDF plot: ", fig_path, " (skipping).")
        return
    end

    # numeric output
    write_ccdf_parquet(vals; outpath=table_path, xname=xname)

    plt = plot(xlabel=xlabel,
               ylabel="CCDF = P(X ≥ x)",
               xscale=:log10,
               yscale=:log10,
               legend=false,
               size=(900, 600),
               title=title)
    plot!(plt, x, y; seriestype=:scatter, markersize=3)
    plot!(plt, x, y; seriestype=:line, alpha=0.7)

    ensure_dir(dirname(fig_path))
    savefig(plt, fig_path)
    println("Wrote CCDF plot ", fig_path)
end

function component_sizes(A::SparseMatrixCSC{Bool,Int})
    G = SimpleGraph(A)
    comps = connected_components(G)
    return Int.(length.(comps))
end

function run_snapshot_plots(df_snap::DataFrame; label::String)
    if nrow(df_snap) == 0
        println("Snapshot $(label): empty, skipping.")
        return
    end

    pairs = unique(df_snap[:, [PERSON_COL, FIRM_COL]])
    if nrow(pairs) == 0
        println("Snapshot $(label): no unique pairs, skipping.")
        return
    end

    B, _, _ = build_incidence_matrix(pairs; person_col=PERSON_COL, firm_col=FIRM_COL)
    manager_deg_bi = vec(sum(B, dims=2))
    firm_deg_bi    = vec(sum(B, dims=1))

    A_mm, A_ff = projection_adjacencies(B)
    manager_deg_proj = vec(sum(A_mm, dims=2))
    firm_deg_proj    = vec(sum(A_ff, dims=1))

    mm_sizes = component_sizes(A_mm)
    ff_sizes = component_sizes(A_ff)

    fig_subdir = joinpath(FIG_DIR, "snapshots_ccdf")
    tab_subdir = joinpath(DESC_DIR, "ccdf", label)
    ensure_dir(fig_subdir)
    ensure_dir(tab_subdir)

    # Degree CCDFs
    save_ccdf_plot(manager_deg_bi,
        fig_path=joinpath(fig_subdir, "ccdf_degree_manager_bipartite_$(label).png"),
        table_path=joinpath(tab_subdir, "ccdf_degree_manager_bipartite.parquet"),
        title="Degree CCDF (log–log) — manager bipartite — $(label)",
        xlabel="Degree",
        xname=:degree
    )
    save_ccdf_plot(firm_deg_bi,
        fig_path=joinpath(fig_subdir, "ccdf_degree_firm_bipartite_$(label).png"),
        table_path=joinpath(tab_subdir, "ccdf_degree_firm_bipartite.parquet"),
        title="Degree CCDF (log–log) — firm bipartite — $(label)",
        xlabel="Degree",
        xname=:degree
    )
    save_ccdf_plot(manager_deg_proj,
        fig_path=joinpath(fig_subdir, "ccdf_degree_manager_projected_$(label).png"),
        table_path=joinpath(tab_subdir, "ccdf_degree_manager_projected.parquet"),
        title="Degree CCDF (log–log) — manager projected — $(label)",
        xlabel="Degree",
        xname=:degree
    )
    save_ccdf_plot(firm_deg_proj,
        fig_path=joinpath(fig_subdir, "ccdf_degree_firm_projected_$(label).png"),
        table_path=joinpath(tab_subdir, "ccdf_degree_firm_projected.parquet"),
        title="Degree CCDF (log–log) — firm projected — $(label)",
        xlabel="Degree",
        xname=:degree
    )

    # Component-size CCDFs (projected graphs)
    save_ccdf_plot(mm_sizes,
        fig_path=joinpath(fig_subdir, "ccdf_component_sizes_manager_projection_$(label).png"),
        table_path=joinpath(tab_subdir, "ccdf_component_sizes_manager_projection.parquet"),
        title="Component-size CCDF (log–log) — manager projection — $(label)",
        xlabel="Component size",
        xname=:component_size
    )
    save_ccdf_plot(ff_sizes,
        fig_path=joinpath(fig_subdir, "ccdf_component_sizes_firm_projection_$(label).png"),
        table_path=joinpath(tab_subdir, "ccdf_component_sizes_firm_projection.parquet"),
        title="Component-size CCDF (log–log) — firm projection — $(label)",
        xlabel="Component size",
        xname=:component_size
    )
end

# ============================================================
# Existing variance plotter (extended)
# ============================================================

function save_multiwindow_plot(df::DataFrame, degree_type::String;
                               outdir::String,
                               deg_filter::Function = d -> true,
                               suffix::String = "all_windows",
                               add_slope::Bool = false,
                               xlog::Bool = false)

    df_sub = filter(row -> row.degree_type == degree_type &&
                           !ismissing(row.var_y) &&
                           row.var_y > 0 &&
                           deg_filter(row.degree) &&
                           (!xlog || row.degree > 0), df)
    if nrow(df_sub) == 0
        println("No positive variance values for $(degree_type) across all windows; skipping combined plot.")
        return
    end

    ensure_dir(outdir)
    png_path = joinpath(outdir, "variance_by_degree_$(degree_type)_$(suffix).png")
    pal = palette(:tab10)

    plt = plot(xlabel = "Degree",
               ylabel = "Var($(String(Y_COL)) demeaned)",
               xscale = xlog ? :log10 : :identity,
               yscale = :log10,
               legend = :topright,
               size = (900, 600),
               title = "$(degree_type): $(suffix)")

    for (idx, grp) in enumerate(groupby(df_sub, [:window_start, :window_end], sort=true))
        color = pal[mod1(idx, length(pal))]
        lbl = "$(first(grp.window_start))-$(first(grp.window_end))"
        plot!(plt, grp.degree, grp.var_y; seriestype=:scatter, markersize=3, color=color, label=lbl)
        plot!(plt, grp.degree, grp.var_y; seriestype=:line, color=color, alpha=0.6, label="")
    end

    if add_slope
        degs = Float64.(df_sub.degree)
        vars = Float64.(df_sub.var_y)
        vx = var(degs)
        if vx > 0
            slope = cov(degs, vars) / vx
            intercept = mean(vars) - slope * mean(degs)
            x_min, x_max = extrema(degs)
            xs = range(x_min, x_max; length=50)
            ys = intercept .+ slope .* xs
            plot!(plt, xs, ys; color=:black, linestyle=:dash, label="OLS slope = $(round(slope, sigdigits=3))")
        end
    end

    savefig(plt, png_path)
    println("Wrote variance plot ", png_path)
end

# ============================================================
# Main
# ============================================================

function run_descriptives()
    ensure_dir(FIG_DIR)
    ensure_dir(DESC_DIR)

    df_raw = Parquet2.readfile(MERGED_PANEL_PATH) |> DataFrame

    # Materialize all columns; convert string-like to String
    df = DataFrame()
    for name in names(df_raw)
        col = collect(df_raw[!, name])
        if any(x -> x isa AbstractString, col)
            df[!, name] = map(x -> ismissing(x) ? missing : String(x), col)
        else
            df[!, name] = col
        end
    end

    @assert Y_COL ∈ propertynames(df) "Column $(Y_COL) not found in merged panel."
    @assert PERSON_COL ∈ propertynames(df) "Column $(PERSON_COL) not found in merged panel."
    @assert FIRM_COL ∈ propertynames(df) "Column $(FIRM_COL) not found in merged panel."
    @assert :year ∈ propertynames(df) "Column :year not found in merged panel."

    dropmissing!(df, [Y_COL, PERSON_COL, FIRM_COL, :year])
    df[!, Y_COL] = Float64.(df[!, Y_COL])
    df[!, PERSON_COL] = Int.(df[!, PERSON_COL])
    df[!, FIRM_COL] = Int.(df[!, FIRM_COL])
    df[!, :year] = Int.(df[!, :year])

    years = Int.(df.year)
    windows = build_windows(years)
    isempty(windows) && error("No windows generated; check START_YEAR/END_YEAR and data coverage.")

    components_rows = DataFrame()
    mgr_var_rows = DataFrame()
    firm_var_rows = DataFrame()

    for (w_start, w_end) in windows
        df_w = df[(df.year .>= w_start) .& (df.year .<= w_end), :]
        if nrow(df_w) == 0
            println("Skipping window $w_start-$w_end (no data).")
            continue
        end

        # Demean outcome by calendar year within the window
        yr_means = Dict(row.year => row.mean_y for row in eachrow(combine(groupby(df_w, :year), Y_COL => mean => :mean_y)))
        df_w[!, :y_demeaned] = Float64.(df_w[!, Y_COL]) .- get.(Ref(yr_means), Int.(df_w[!, :year]), NaN)

        pairs = unique(df_w[:, [PERSON_COL, FIRM_COL]])
        B, person_ids, firm_ids = build_incidence_matrix(pairs; person_col=PERSON_COL, firm_col=FIRM_COL)

        manager_deg_bi = vec(sum(B, dims=2))
        firm_deg_bi    = vec(sum(B, dims=1))

        A_mm, A_ff = projection_adjacencies(B)
        manager_deg_proj = vec(sum(A_mm, dims=2))
        firm_deg_proj    = vec(sum(A_ff, dims=1))

        mm_components, mm_lcc, mm_iso = component_summary(A_mm)
        ff_components, ff_lcc, ff_iso = component_summary(A_ff)

        comps_row = DataFrame(
            window_start = w_start,
            window_end = w_end,
            n_managers = length(person_ids),
            n_firms = length(firm_ids),
            n_edges = nrow(pairs),
            manager_degree_mean_bipartite = mean(manager_deg_bi),
            firm_degree_mean_bipartite = mean(firm_deg_bi),
            manager_degree_mean_projected = mean(manager_deg_proj),
            firm_degree_mean_projected = mean(firm_deg_proj),
            manager_components = mm_components,
            manager_lcc_size = mm_lcc,
            manager_lcc_share = length(person_ids) == 0 ? 0.0 : mm_lcc / length(person_ids),
            manager_isolates = mm_iso,
            firm_components = ff_components,
            firm_lcc_size = ff_lcc,
            firm_lcc_share = length(firm_ids) == 0 ? 0.0 : ff_lcc / length(firm_ids),
            firm_isolates = ff_iso
        )
        components_rows = isempty(components_rows) ? comps_row : vcat(components_rows, comps_row)

        mgr_bi_map    = degree_map(person_ids, manager_deg_bi)
        mgr_proj_map  = degree_map(person_ids, manager_deg_proj)
        firm_bi_map   = degree_map(firm_ids, firm_deg_bi)
        firm_proj_map = degree_map(firm_ids, firm_deg_proj)

        mgr_bi = variance_by_degree(df_w, PERSON_COL, :y_demeaned, mgr_bi_map;
                                    window_start=w_start, window_end=w_end, degree_type="manager_bipartite")
        mgr_proj = variance_by_degree(df_w, PERSON_COL, :y_demeaned, mgr_proj_map;
                                      window_start=w_start, window_end=w_end, degree_type="manager_projected")
        firm_bi = variance_by_degree(df_w, FIRM_COL, :y_demeaned, firm_bi_map;
                                     window_start=w_start, window_end=w_end, degree_type="firm_bipartite")
        firm_proj = variance_by_degree(df_w, FIRM_COL, :y_demeaned, firm_proj_map;
                                       window_start=w_start, window_end=w_end, degree_type="firm_projected")

        # FIXED: Julia requires spacing around ternary operator `? :`
        mgr_var_rows = isempty(mgr_var_rows) ? vcat(mgr_bi, mgr_proj) : vcat(mgr_var_rows, mgr_bi, mgr_proj)
        firm_var_rows = isempty(firm_var_rows) ? vcat(firm_bi, firm_proj) : vcat(firm_var_rows, firm_bi, firm_proj)
    end

    # Export parquet tables
    Parquet2.writefile(joinpath(DESC_DIR, "descriptives_components.parquet"), components_rows)
    Parquet2.writefile(joinpath(DESC_DIR, "descriptives_manager_var_by_degree.parquet"), mgr_var_rows)
    Parquet2.writefile(joinpath(DESC_DIR, "descriptives_firm_var_by_degree.parquet"), firm_var_rows)
    println("Wrote parquet tables to ", DESC_DIR)

    # Variance plots (existing + log–log versions)
    var_fig_dir = joinpath(FIG_DIR, "variance_by_degree")
    ensure_dir(var_fig_dir)

    for dt in ("manager_bipartite", "firm_bipartite", "manager_projected", "firm_projected")
        base_df = dt in ("manager_bipartite", "manager_projected") ? mgr_var_rows : firm_var_rows

        save_multiwindow_plot(base_df, dt; outdir=var_fig_dir, add_slope=false)
        save_multiwindow_plot(base_df, dt; outdir=var_fig_dir, deg_filter=d -> d <= 5, suffix="deg_le_5", add_slope=true)

        # log–log (not CCDF)
        save_multiwindow_plot(base_df, dt; outdir=var_fig_dir, suffix="all_windows_loglog", add_slope=false, xlog=true)
        save_multiwindow_plot(base_df, dt; outdir=var_fig_dir, deg_filter=d -> d <= 5, suffix="deg_le_5_loglog", add_slope=true, xlog=true)
    end

    # Snapshot CCDF plots + numeric CCDF tables
    run_snapshot_plots(df; label="full")
    run_snapshot_plots(df[(df.year .>= 1990) .& (df.year .<= 1998), :]; label="1990_1998")
    run_snapshot_plots(df[(df.year .>= 2010) .& (df.year .<= 2018), :]; label="2010_2018")

    println("All figures written under ", FIG_DIR)
end

run_descriptives()