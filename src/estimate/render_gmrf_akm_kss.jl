#!/usr/bin/env julia

using Printf: @sprintf

function parse_number(str::AbstractString)
    s = replace(strip(str), "," => "")
    isempty(s) && error("Cannot parse empty numeric field.")
    m = match(r"^[ \t]*([+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?|[+-]?Inf|NaN)", s)
    m === nothing && error("Cannot parse numeric field from: $str")
    return parse(Float64, m.captures[1])
end

function parse_bool(str::AbstractString)
    s = lowercase(strip(str))
    s == "true" && return true
    s == "false" && return false
    error("Cannot parse Bool from: $str")
end

function parse_pipe_kv(line::AbstractString)
    out = Dict{String,String}()
    for part in split(line, '|')
        if occursin('=', part)
            key, value = split(part, '=', limit=2)
            out[strip(key)] = strip(value)
        end
    end
    return out
end

function parse_simple_kv(line::AbstractString)
    key, value = split(line, '=', limit=2)
    return strip(key) => strip(value)
end

function maybe_parse_number(meta::Dict{String,Any}, key::String)
    return haskey(meta, key) ? parse_number(String(meta[key])) : NaN
end

function maybe_parse_int(meta::Dict{String,Any}, key::String)
    return haskey(meta, key) ? parse(Int, String(meta[key])) : missing
end

function parse_gmrf_values(lines, start_idx::Int; stop_at=String[])
    vals = Dict{String,Float64}()
    keys = ("rho", "sigma_a", "sigma_z", "sigma_eps", "rho_eps",
            "V_firm", "V_manager", "V_cross", "V_eps", "V_total")

    for line in lines[(start_idx + 1):end]
        s = strip(line)
        isempty(s) && continue
        any(marker -> startswith(s, marker), stop_at) && break
        startswith(s, "Variance decomposition") && continue
        for key in keys
            if occursin(Regex("^" * key * "\\s*="), s)
                _, rhs = split(s, '=', limit=2)
                vals[key] = parse_number(rhs)
            end
        end
    end

    return vals
end

const GMRF_POSTERIOR_SUMMARY_MARKERS = (
    "AKM-Comparable Summary",
    "Target Posterior Summary",
)

function find_gmrf_section(lines, markers)
    return findfirst(i -> any(marker -> startswith(strip(lines[i]), marker), markers), eachindex(lines))
end

function make_gmrf_decomp(vals::Dict{String,Float64})
    V_firm = vals["V_firm"]
    V_person = vals["V_manager"]
    V_cross = vals["V_cross"]
    V_eps = vals["V_eps"]
    V_total = vals["V_total"]
    cov_fp = V_cross / 2.0
    rho = (V_firm > 0 && V_person > 0) ? cov_fp / sqrt(V_firm * V_person) : NaN

    return (
        V_firm=V_firm,
        V_person=V_person,
        Cov_firm_person=cov_fp,
        V_cross=V_cross,
        V_eps=V_eps,
        V_total=V_total,
        rho=rho,
        share_firm=V_total != 0 ? V_firm / V_total : NaN,
        share_person=V_total != 0 ? V_person / V_total : NaN,
        share_cross=V_total != 0 ? V_cross / V_total : NaN,
        share_eps=V_total != 0 ? V_eps / V_total : NaN,
        share_explained=V_total != 0 ? (V_firm + V_person + V_cross) / V_total : NaN,
    )
end

function parse_akm(path::String)
    isfile(path) || error("AKM/KSS output not found: $path")
    lines = readlines(path)

    meta = Dict{String,Any}()
    rows = Dict{String,Dict{String,Any}}()
    current = nothing

    for line in lines
        s = strip(line)
        isempty(s) && continue

        if startswith(s, "Estimator=")
            hdr = parse_pipe_kv(s)
            key = hdr["Estimator"]
            available = haskey(hdr, "Available") ? parse_bool(hdr["Available"]) : true
            rows[key] = Dict{String,Any}("Estimator" => key, "Label" => hdr["Label"],
                                         "Available" => available)
            current = available ? rows[key] : nothing
        elseif startswith(s, "Chunk=") || startswith(s, "Outcome=") || startswith(s, "InputRows=")
            merge!(meta, Dict{String,Any}(parse_pipe_kv(s)))
        elseif startswith(s, "Input=") || startswith(s, "Simulations=") ||
               startswith(s, "OutcomeVariance=") || startswith(s, "HomoskedasticSigmaCommon=") ||
               startswith(s, "HomoskedasticSigmaMethod=") || startswith(s, "KSSUnavailableReason=")
            k, v = parse_simple_kv(s)
            meta[k] = v
        elseif current !== nothing && occursin('=', s)
            k, v = parse_simple_kv(s)
            current[k] = v
        end
    end

    # Support both old format (PackageRows/SameSample) and new (KSSRows/KSSAvailable)
    kss_rows = haskey(meta, "KSSRows") ? parse(Int, String(meta["KSSRows"])) :
               haskey(meta, "PackageRows") ? parse(Int, String(meta["PackageRows"])) : 0
    kss_available = haskey(meta, "KSSAvailable") ? parse_bool(String(meta["KSSAvailable"])) :
                    haskey(meta, "SameSample") ? parse_bool(String(meta["SameSample"])) : true

    return (
        path=path,
        input=String(meta["Input"]),
        chunk=String(meta["Chunk"]),
        sample=String(meta["Sample"]),
        algorithm=String(meta["Algorithm"]),
        leave_out_level=String(meta["LeaveOutLevel"]),
        simulations=haskey(meta, "Simulations") ? parse(Int, String(meta["Simulations"])) : missing,
        input_rows=parse(Int, String(meta["InputRows"])),
        outcome_rows=parse(Int, String(meta["OutcomeNonmissingRows"])),
        kss_rows=kss_rows,
        kss_available=kss_available,
        kss_unavailable_reason=get(meta, "KSSUnavailableReason", nothing),
        outcome_variance=parse_number(String(meta["OutcomeVariance"])),
        sigma_common=haskey(meta, "HomoskedasticSigmaCommon") ?
                     parse_number(String(meta["HomoskedasticSigmaCommon"])) : NaN,
        sigma_method=haskey(meta, "HomoskedasticSigmaMethod") ?
                     String(meta["HomoskedasticSigmaMethod"]) : "N/A",
        rows=rows,
    )
end

function parse_gmrf(path::String)
    isfile(path) || error("GMRF output not found: $path")
    lines = readlines(path)

    meta = Dict{String,Any}()
    k_line = nothing

    for line in lines
        s = strip(line)
        startswith(s, "Chunk=") && merge!(meta, Dict{String,Any}(parse_pipe_kv(s)))
        (startswith(s, "Outcome=") || startswith(s, "AdjacencyWeighting=") ||
         startswith(s, "PriorAdjacency=") || startswith(s, "ObsWeighting=") ||
         startswith(s, "DecompTarget=") || startswith(s, "RhoEpsMode=") ||
         startswith(s, "RhoEps=") || startswith(s, "PersonYearRowsFinite=") ||
         startswith(s, "EffectiveWeight=") || startswith(s, "EffectiveWeightOverTSum=") ||
         startswith(s, "WithinDF=")) &&
            merge!(meta, Dict{String,Any}(parse_pipe_kv(s)))
        startswith(s, "Input=") && (meta[first(parse_simple_kv(s))] = last(parse_simple_kv(s)))
        startswith(s, "N_F=") && (k_line = s)
    end

    posterior_summary_start = find_gmrf_section(lines, GMRF_POSTERIOR_SUMMARY_MARKERS)
    posterior_summary_start === nothing &&
        error("Could not find a GMRF posterior summary block in $path. This report requires current gmrfmle_exact.jl output generated with decomposition enabled.")
    prior_start = findfirst(i -> occursin("Prior Variance Decomposition", lines[i]), eachindex(lines))
    prior_start === nothing && error("Could not find prior variance decomposition block in $path")
    structural_start = findfirst(i -> occursin("Estimates (structural units)", lines[i]), eachindex(lines))
    structural_start === nothing && error("Could not find structural estimates block in $path")

    vals = parse_gmrf_values(lines, posterior_summary_start; stop_at=["Posterior Variance Decomposition"])
    prior_vals = parse_gmrf_values(lines, prior_start; stop_at=[
        "AKM-Comparable Summary",
        "Target Posterior Summary",
        "Posterior Variance Decomposition",
    ])
    structural_vals = parse_gmrf_values(lines, structural_start; stop_at=[
        "Prior Variance Decomposition",
        "AKM-Comparable Summary",
        "Target Posterior Summary",
        "Posterior Variance Decomposition",
    ])

    K = let m = match(r"K=(\d+)", something(k_line, ""))
        m === nothing ? missing : parse(Int, m.captures[1])
    end

    posterior = make_gmrf_decomp(vals)
    prior = make_gmrf_decomp(prior_vals)
    rho_eps_value = haskey(meta, "RhoEps") ? maybe_parse_number(meta, "RhoEps") :
                    get(structural_vals, "rho_eps", NaN)
    return (
        path=path,
        input=String(meta["Input"]),
        chunk=String(meta["Chunk"]),
        sample=String(meta["Sample"]),
        weighting=String(meta["AdjacencyWeighting"]),
        obs_weighting=String(get(meta, "ObsWeighting", "raw")),
        decomp_target=String(get(meta, "DecompTarget", "estimation")),
        rho_eps_mode=String(get(meta, "RhoEpsMode", "none")),
        rho_eps=rho_eps_value,
        effective_summary=(
            formula=String(get(meta, "EffectiveWeight", "")),
            personyear_rows_finite=maybe_parse_int(meta, "PersonYearRowsFinite"),
            effective_weight_sum=maybe_parse_number(meta, "EffectiveWeightSum"),
            mean_effective_weight=maybe_parse_number(meta, "MeanEffectiveWeight"),
            max_effective_weight=maybe_parse_number(meta, "MaxEffectiveWeight"),
            effective_weight_over_t_sum=maybe_parse_number(meta, "EffectiveWeightOverTSum"),
            within_df=maybe_parse_int(meta, "WithinDF"),
            within_ss_scaled=maybe_parse_number(meta, "WithinSSScaled"),
            log_weight_sum=maybe_parse_number(meta, "LogWeightSum"),
        ),
        K=K,
        prior=prior,
        structural_rho=structural_vals["rho"],
        structural_sigma_a=structural_vals["sigma_a"],
        structural_sigma_z=structural_vals["sigma_z"],
        structural_sigma_eps=structural_vals["sigma_eps"],
        V_firm=posterior.V_firm,
        V_person=posterior.V_person,
        Cov_firm_person=posterior.Cov_firm_person,
        V_cross=posterior.V_cross,
        V_eps=posterior.V_eps,
        V_total=posterior.V_total,
        sigma_firm=vals["sigma_a"],
        sigma_person=vals["sigma_z"],
        sigma_eps=vals["sigma_eps"],
        rho=vals["rho"],
        share_firm=posterior.share_firm,
        share_person=posterior.share_person,
        share_cross=posterior.share_cross,
        share_eps=posterior.share_eps,
        share_explained=posterior.share_explained,
    )
end

fmt(x::Real)         = @sprintf("%.6f", Float64(x))
fmt_int(x::Integer)  = string(x)
fmt_pct(x::Real)     = @sprintf("%+.1f%%", 100 * Float64(x))
fmt_bool(x::Bool)    = x ? "`true`" : "`false`"
fmt_or_na(x::Real)     = isfinite(Float64(x)) ? fmt(x) : "—"
fmt_pct_or_na(x::Real) = isfinite(Float64(x)) ? fmt_pct(x) : "—"

function row(cells)
    return "| " * join(cells, " | ") * " |\n"
end

function _na_akm_row(label::String)
    return (available=false, label=label,
            V_firm=NaN, V_person=NaN, Cov_firm_person=NaN,
            V_cross=NaN, V_eps=NaN, V_total=NaN,
            sigma_firm=NaN, sigma_person=NaN, rho=NaN,
            share_firm=NaN, share_person=NaN, share_cross=NaN,
            share_eps=NaN, share_explained=NaN)
end

function get_akm_row(akm, key::String)
    !haskey(akm.rows, key) && return _na_akm_row(key)
    src = akm.rows[key]
    avail = src["Available"]
    avail isa String && (avail = parse_bool(avail))
    !avail && return _na_akm_row(String(src["Label"]))
    return (
        available=true,
        label=String(src["Label"]),
        V_firm=parse_number(String(src["V_firm"])),
        V_person=parse_number(String(src["V_person"])),
        Cov_firm_person=parse_number(String(src["Cov_firm_person"])),
        V_cross=parse_number(String(src["V_cross"])),
        V_eps=parse_number(String(src["V_eps"])),
        V_total=parse_number(String(src["V_total"])),
        sigma_firm=parse_number(String(src["sigma_firm"])),
        sigma_person=parse_number(String(src["sigma_person"])),
        rho=parse_number(String(src["rho"])),
        share_firm=parse_number(String(src["share_firm"])),
        share_person=parse_number(String(src["share_person"])),
        share_cross=parse_number(String(src["share_cross"])),
        share_eps=parse_number(String(src["share_eps"])),
        share_explained=parse_number(String(src["share_explained"])),
    )
end

function gmrf_q_label(gmrf)
    gmrf.weighting == "degree" && return "Weighted Q"
    gmrf.weighting == "unweighted" && return "Unweighted Q"
    return "$(gmrf.weighting) Q"
end

function gmrf_obs_label(gmrf)
    gmrf.obs_weighting == "raw" && return "raw"
    gmrf.obs_weighting == "effective" && return "effective rho_eps"
    return gmrf.obs_weighting
end

function gmrf_spec_label(gmrf)
    return "$(gmrf_q_label(gmrf)) / $(gmrf_obs_label(gmrf))"
end

function gmrf_target_label(gmrf)
    gmrf.decomp_target == "personyear" && return "person-year"
    gmrf.decomp_target in ("estimation", "likelihood", "likelihood_mean") && return "estimation-observation"
    return replace(gmrf.decomp_target, "_" => " ")
end

gmrf_posterior_label(gmrf) = "Posterior, $(gmrf_target_label(gmrf)) target"
gmrf_prior_label(gmrf) = "Prior, Object 2, $(gmrf_target_label(gmrf)) target"

function render_markdown(akm, gmrfs)
    raw = get_akm_row(akm, "raw_akm")
    ho  = get_akm_row(akm, "kss_homoskedastic")
    he  = get_akm_row(akm, "kss_heteroskedastic")
    show_rho_eps = any(g -> isfinite(Float64(g.rho_eps)), gmrfs)

    io = IOBuffer()
    write(io, "# GMRF vs AKM/KSS\n\n")
    write(io, "> [!info] Common Sample\n")
    write(io, "> Input parquet: `$(akm.input)`\n")
    write(io, "> Nonmissing person-year observations: `$(fmt_int(akm.outcome_rows))`\n")
    if akm.kss_available
        write(io, "> KSS LOO set: `$(fmt_int(akm.kss_rows))` rows (KSS available)\n")
    else
        reason = akm.kss_unavailable_reason !== nothing ? ": $(akm.kss_unavailable_reason)" : ""
        write(io, "> KSS LOO set: `$(fmt_int(akm.kss_rows))` rows — **KSS not available**$(reason)\n")
    end
    k_specs = ["$(gmrf_spec_label(g))=$(fmt_int(g.K))" for g in gmrfs if !ismissing(g.K)]
    if length(k_specs) == 1
        write(io, "> GMRF runs use the same parquet with `K = $(split(k_specs[1], '=')[2])`\n")
    elseif !isempty(k_specs)
        write(io, "> GMRF runs use the same parquet; K by specification: `$(join(k_specs, "; "))`\n")
    end
    write(io, "> Sample variance of the outcome: `$(fmt(akm.outcome_variance))`\n\n")

    write(io, "## Variance Components\n")
    write(io, "| Estimator | Specification | Object | V_firm | V_person | Cov(firm, person) | 2Cov | V_eps | V_total | Corr(firm, person) |\n")
    write(io, "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    write(io, row(["AKM", raw.label, "Plug-in", fmt_or_na(raw.V_firm), fmt_or_na(raw.V_person), fmt_or_na(raw.Cov_firm_person), fmt_or_na(raw.V_cross), fmt_or_na(raw.V_eps), fmt_or_na(raw.V_total), fmt_or_na(raw.rho)]))
    write(io, row(["AKM + KSS", "Homoskedastic", "Bias-corrected", fmt_or_na(ho.V_firm), fmt_or_na(ho.V_person), fmt_or_na(ho.Cov_firm_person), fmt_or_na(ho.V_cross), fmt_or_na(ho.V_eps), fmt_or_na(ho.V_total), fmt_or_na(ho.rho)]))
    write(io, row(["AKM + KSS", "Heteroskedastic", "Bias-corrected", fmt_or_na(he.V_firm), fmt_or_na(he.V_person), fmt_or_na(he.Cov_firm_person), fmt_or_na(he.V_cross), fmt_or_na(he.V_eps), fmt_or_na(he.V_total), fmt_or_na(he.rho)]))
    for gmrf in gmrfs
        write(io, row(["GMRF", gmrf_spec_label(gmrf), gmrf_posterior_label(gmrf), fmt(gmrf.V_firm), fmt(gmrf.V_person), fmt(gmrf.Cov_firm_person), fmt(gmrf.V_cross), fmt(gmrf.V_eps), fmt(gmrf.V_total), fmt(gmrf.rho)]))
        write(io, row(["GMRF", gmrf_spec_label(gmrf), gmrf_prior_label(gmrf), fmt(gmrf.prior.V_firm), fmt(gmrf.prior.V_person), fmt(gmrf.prior.Cov_firm_person), fmt(gmrf.prior.V_cross), fmt(gmrf.prior.V_eps), fmt(gmrf.prior.V_total), fmt(gmrf.prior.rho)]))
    end

    write(io, "\n## Variance Shares\n")
    write(io, "| Estimator | Specification | Object | Firm | Person | 2Cov | Residual | Explained |\n")
    write(io, "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
    write(io, row(["AKM", raw.label, "Plug-in", fmt_pct_or_na(raw.share_firm), fmt_pct_or_na(raw.share_person), fmt_pct_or_na(raw.share_cross), fmt_pct_or_na(raw.share_eps), fmt_pct_or_na(raw.share_explained)]))
    write(io, row(["AKM + KSS", "Homoskedastic", "Bias-corrected", fmt_pct_or_na(ho.share_firm), fmt_pct_or_na(ho.share_person), fmt_pct_or_na(ho.share_cross), fmt_pct_or_na(ho.share_eps), fmt_pct_or_na(ho.share_explained)]))
    write(io, row(["AKM + KSS", "Heteroskedastic", "Bias-corrected", fmt_pct_or_na(he.share_firm), fmt_pct_or_na(he.share_person), fmt_pct_or_na(he.share_cross), fmt_pct_or_na(he.share_eps), fmt_pct_or_na(he.share_explained)]))
    for gmrf in gmrfs
        write(io, row(["GMRF", gmrf_spec_label(gmrf), gmrf_posterior_label(gmrf), fmt_pct(gmrf.share_firm), fmt_pct(gmrf.share_person), fmt_pct(gmrf.share_cross), fmt_pct(gmrf.share_eps), fmt_pct(gmrf.share_explained)]))
        write(io, row(["GMRF", gmrf_spec_label(gmrf), gmrf_prior_label(gmrf), fmt_pct(gmrf.prior.share_firm), fmt_pct(gmrf.prior.share_person), fmt_pct(gmrf.prior.share_cross), fmt_pct(gmrf.prior.share_eps), fmt_pct(gmrf.prior.share_explained)]))
    end

    write(io, "\n## GMRF Structural MLE Parameters\n")
    if show_rho_eps
        write(io, "| Specification | rho | sigma_a | sigma_z | sigma_eps | rho_eps |\n")
        write(io, "| --- | ---: | ---: | ---: | ---: | ---: |\n")
    else
        write(io, "| Specification | rho | sigma_a | sigma_z | sigma_eps |\n")
        write(io, "| --- | ---: | ---: | ---: | ---: |\n")
    end
    for gmrf in gmrfs
        cells = [gmrf_spec_label(gmrf), fmt(gmrf.structural_rho), fmt(gmrf.structural_sigma_a),
                 fmt(gmrf.structural_sigma_z), fmt(gmrf.structural_sigma_eps)]
        show_rho_eps && push!(cells, fmt_or_na(gmrf.rho_eps))
        write(io, row(cells))
    end

    write(io, "\n## Notes\n")
    write(io, "- `2Cov` is the sorting term in the variance decomposition. `Cov(firm, person)` is shown separately because the reported correlation is defined from the covariance, not from `2Cov`.\n")
    write(io, "- The Raw AKM row is computed on the full sample via sparse direct two-way FE (within-firm Laplacian, sparse Cholesky).\n")
    if akm.kss_available
        write(io, "- The KSS rows use the leave-one-worker-out connected set (`$(fmt_int(akm.kss_rows))` rows). The homoskedastic row uses a common residual variance `sigma = $(fmt(akm.sigma_common))` computed as `$(akm.sigma_method)`.\n")
    else
        reason = akm.kss_unavailable_reason !== nothing ? " ($(akm.kss_unavailable_reason))" : ""
        write(io, "- KSS bias correction is **not available**: the leave-one-worker-out connected set collapsed to `$(fmt_int(akm.kss_rows))` rows$(reason). The Hungarian CEO network is too sparse — every mover is an articulation point. KSS cells show `—`.\n")
    end
    write(io, "- The GMRF posterior rows come from the Object 3 posterior decomposition that `gmrfmle_exact.jl` computes when `--decompose` is passed; these are the moments intended to be directly comparable to AKM/KSS on the same sample.\n")
    write(io, "- The GMRF prior rows report the fitted Object 2 prior decomposition; they are shown for comparison but are not the AKM/KSS-comparable posterior moments.\n")
    write(io, "- The `Corr(firm, person)` column is the implied correlation from the variance decomposition. It is not the same object as the structural GMRF `rho`; the structural MLE parameters are reported separately.\n")
    show_rho_eps && write(io, "- Effective-weight GMRF rows use `rho_eps` from their estimate files; raw observation-weighting rows show `—` in the structural-parameter table.\n")
    write(io, "- In the GMRF output files the person component is stored as `V_manager` / `sigma_z`; this report renames it to `V_person` / `sigma_person`.\n")
    source_files = join(vcat(["`$(akm.path)`"], ["`$(g.path)`" for g in gmrfs]), ", ")
    write(io, "- Source files: $(source_files).\n")

    return String(take!(io))
end

function main()
    default_akm_path = "output/akm_vchdfe/kss-test-raw/kss/estimates.txt"
    default_gmrf_paths = [
        "output/gmrfmle/kss-test-raw/kss/estimates_decomp-personyear.txt",
        "output/gmrfmle/kss-test-raw/kss/estimates_obs-effective_rhoeps-est_decomp-personyear.txt",
        "output/gmrfmle/kss-test-raw/kss/estimates_unweighted_decomp-personyear.txt",
        "output/gmrfmle/kss-test-raw/kss/estimates_unweighted_obs-effective_rhoeps-est_decomp-personyear.txt",
    ]
    if !all(isfile, default_gmrf_paths)
        default_gmrf_paths = [
            "output/gmrfmle/kss-test-raw/kss/estimates.txt",
            "output/gmrfmle/kss-test-raw/kss/estimates_unweighted.txt",
        ]
    end
    args = isempty(ARGS) ? vcat([default_akm_path], default_gmrf_paths, ["output/gmrf_akm_kss.md"]) : ARGS

    length(args) >= 3 || error("Usage: julia --project=. src/estimate/render_gmrf_akm_kss.jl <akm_estimates.txt> <gmrf_estimates.txt> [<gmrf_estimates.txt>...] <out.md>")

    akm_path = first(args)
    gmrf_paths = args[2:end-1]
    out_path = last(args)
    akm = parse_akm(akm_path)
    gmrfs = parse_gmrf.(gmrf_paths)

    all(g -> akm.input == g.input, gmrfs) ||
        error("Input parquet paths differ across outputs.")

    mkpath(dirname(out_path))
    write(out_path, render_markdown(akm, gmrfs))
    println("Wrote: $out_path")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
