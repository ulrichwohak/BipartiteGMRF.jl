#!/usr/bin/env julia
# ============================================================
# akm_vchdfe.jl
#
# Thin wrapper around VarianceComponentsHDFE.jl for AKM/KSS
# estimation on the exact same parquet sample used by GMRF.
#
# Usage:
#   julia +1.8.5 --project=<env-with-Parquet2> src/estimate/akm_vchdfe.jl <path/to/edgelist.parquet> [outcome] [flags]
#
# Flags:
#   --algorithm=<jla|exact>         Leverage algorithm (default: jla)
#   --simulations=<N>               JLA simulations (default: 200)
#   --leave-out-level=<match|obs>   KSS leave-out level (default: match)
#   --print-level=<N>               Package verbosity (default: 1)
#   --allow-sample-change           Allow the package to drop/reorder rows
#   --vchdfe-project=<path>         Julia project/env where VarianceComponentsHDFE.jl is installed
#
# Output:
#   output/akm_vchdfe/<chunk>/<sample>/estimates*.txt
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

v"1.8.0" <= VERSION < v"1.9.0" ||
    @warn "VarianceComponentsHDFE.jl is tested under Julia 1.8.x. Current version: $VERSION. Results may differ or the package may fail to load."

using Parquet2
using DataFrames
using LinearAlgebra
using SparseArrays
using Statistics
using Printf: @printf
using VarianceComponentsHDFE: VCHDFESettings, JLAAlgorithm, ExactAlgorithm,
    get_leave_one_out_set, leave_out_estimation, compute_matchid,
    kss_quadratic_form

to_float_nan(x) = x isa Missing ? NaN : Float64(x)

function parse_input_path(in_path::String)::Tuple{String,String,String}
    normalized = replace(in_path, "\\" => "/")
    parts = split(normalized, '/')

    samples_idx = findfirst(==("samples"), parts)
    if samples_idx !== nothing && length(parts) >= samples_idx + 3
        chunk = parts[samples_idx + 1]
        sample_parts = parts[samples_idx + 2 : end - 1]
        sample = join(sample_parts, "/")
        out_path = joinpath("output", "akm_vchdfe", chunk, sample_parts..., "estimates.txt")
        return (chunk, sample, out_path)
    end

    chunks_idx = findfirst(==("chunks"), parts)
    if chunks_idx !== nothing && length(parts) >= chunks_idx + 2
        chunk_parts = parts[chunks_idx + 1 : end - 1]
        chunk = join(chunk_parts, "/")
        out_path = joinpath("output", "akm_vchdfe", chunk_parts..., "exact", "estimates.txt")
        return (chunk, "exact", out_path)
    end

    error("Cannot parse input path. Expected temp/chunks/<chunk>/... or temp/samples/<chunk>/<sample>/...: $in_path")
end

function make_settings(
    algorithm::String,
    simulations::Int,
    leave_out_level::String,
    outcome::Symbol,
    print_level::Int
)
    lev =
        algorithm == "exact" ? ExactAlgorithm() :
        algorithm == "jla"   ? JLAAlgorithm(num_simulations=simulations) :
        error("--algorithm must be jla or exact; got $algorithm")

    return VCHDFESettings(
        leverage_algorithm=lev,
        leave_out_level=leave_out_level,
        print_level=print_level,
        first_id_display="Worker",
        first_id_display_small="worker",
        second_id_display="Firm",
        second_id_display_small="firm",
        outcome_id_display=String(outcome),
        outcome_id_display_small=String(outcome),
    )
end

function summarize_components(V_firm::Real, V_person::Real, cov_fp::Real, V_total::Real)
    V_firm   = Float64(V_firm)
    V_person = Float64(V_person)
    cov_fp   = Float64(cov_fp)
    V_total  = Float64(V_total)
    V_cross  = 2.0 * cov_fp
    V_eps    = V_total - V_firm - V_person - V_cross
    sigma_f  = V_firm > 0 ? sqrt(V_firm) : NaN
    sigma_p  = V_person > 0 ? sqrt(V_person) : NaN
    rho = (isfinite(sigma_f) && isfinite(sigma_p) && sigma_f > 0 && sigma_p > 0) ?
          cov_fp / (sigma_f * sigma_p) : NaN
    share_firm   = V_total != 0 ? V_firm   / V_total : NaN
    share_person = V_total != 0 ? V_person / V_total : NaN
    share_cross  = V_total != 0 ? V_cross  / V_total : NaN
    share_eps    = V_total != 0 ? V_eps    / V_total : NaN
    return (
        V_firm=V_firm,
        V_person=V_person,
        Cov_firm_person=cov_fp,
        V_cross=V_cross,
        V_eps=V_eps,
        V_total=V_total,
        sigma_firm=sigma_f,
        sigma_person=sigma_p,
        rho=rho,
        share_firm=share_firm,
        share_person=share_person,
        share_cross=share_cross,
        share_eps=share_eps,
        share_explained=share_firm + share_person + share_cross,
    )
end

function build_var_design(first_id::Vector{Int}, second_id::Vector{Int}, leave_out_level::String)
    K_py = length(first_id)
    N = maximum(first_id)
    J = maximum(second_id)
    S = sparse(1.0I, J - 1, J - 1)
    S = vcat(S, sparse(-zeros(1, J - 1)))

    if leave_out_level == "match"
        match_id = compute_matchid(first_id, second_id)
        M = maximum(match_id)
        weight = zeros(Int, M)
        first_match = zeros(Int, M)
        second_match = zeros(Int, M)
        for k in eachindex(match_id)
            m = match_id[k]
            weight[m] += 1
            first_match[m] = first_id[k]
            second_match[m] = second_id[k]
        end
        first_id_weighted = vcat(fill.(first_match, weight)...)
        second_id_weighted = vcat(fill.(second_match, weight)...)
        Dvar = hcat(sparse(collect(1:K_py), first_id_weighted, 1.0, K_py, N), spzeros(K_py, J - 1))
        Fvar = hcat(spzeros(K_py, N), -1.0 * sparse(collect(1:K_py), second_id_weighted, 1.0, K_py, J) * S)
    else
        D = sparse(collect(1:K_py), first_id, 1.0, K_py, N)
        F = sparse(collect(1:K_py), second_id, 1.0, K_py, J)
        Dvar = hcat(D, spzeros(K_py, J - 1))
        Fvar = hcat(spzeros(K_py, N), -1.0 * F * S)
    end
    return Dvar, Fvar
end

function compute_akm_rows(y::Vector{Float64}, first_id::Vector{Int}, second_id::Vector{Int}, settings)
    est = leave_out_estimation(y, first_id, second_id, nothing, settings)
    Dvar, Fvar = build_var_design(first_id, second_id, settings.leave_out_level)
    zero_sigma = zeros(Float64, length(est.sigma_i))

    raw_firm   = kss_quadratic_form(zero_sigma, Fvar, Fvar, est.β, est.Bii_second)
    raw_person = kss_quadratic_form(zero_sigma, Dvar, Dvar, est.β, est.Bii_first)
    raw_cov    = kss_quadratic_form(zero_sigma, Fvar, Dvar, est.β, est.Bii_cov)

    eta = est.y - est.X * est.β
    sigma_dof = length(est.y) - size(est.X, 2)
    sigma_dof > 0 || error("Cannot form homoskedastic variance estimate: residual degrees of freedom must be positive.")
    sigma_common = sum(abs2, eta) / sigma_dof
    ho_sigma = fill(sigma_common, length(est.sigma_i))
    ho_firm   = kss_quadratic_form(ho_sigma, Fvar, Fvar, est.β, est.Bii_second)
    ho_person = kss_quadratic_form(ho_sigma, Dvar, Dvar, est.β, est.Bii_first)
    ho_cov    = kss_quadratic_form(ho_sigma, Fvar, Dvar, est.β, est.Bii_cov)

    V_total = var(y)
    return (
        sigma_common=sigma_common,
        sigma_method="SSR/(n-p)",
        raw=summarize_components(raw_firm.theta, raw_person.theta, raw_cov.theta, V_total),
        ho=summarize_components(ho_firm.theta_KSS, ho_person.theta_KSS, ho_cov.theta_KSS, V_total),
        he=summarize_components(est.θ_second, est.θ_first, est.θCOV, V_total),
    )
end

function write_row(io, key::String, label::String, row)
    @printf(io, "Estimator=%s | Label=%s\n", key, label)
    @printf(io, "  V_firm           = %+.6f\n", row.V_firm)
    @printf(io, "  V_person         = %+.6f\n", row.V_person)
    @printf(io, "  Cov_firm_person  = %+.6f\n", row.Cov_firm_person)
    @printf(io, "  V_cross          = %+.6f\n", row.V_cross)
    @printf(io, "  V_eps            = %+.6f\n", row.V_eps)
    @printf(io, "  V_total          = %+.6f\n", row.V_total)
    @printf(io, "  sigma_firm       = %.6f\n", row.sigma_firm)
    @printf(io, "  sigma_person     = %.6f\n", row.sigma_person)
    @printf(io, "  rho              = %.6f\n", row.rho)
    @printf(io, "  share_firm       = %+.6f\n", row.share_firm)
    @printf(io, "  share_person     = %+.6f\n", row.share_person)
    @printf(io, "  share_cross      = %+.6f\n", row.share_cross)
    @printf(io, "  share_eps        = %+.6f\n", row.share_eps)
    @printf(io, "  share_explained  = %+.6f\n", row.share_explained)
end

function write_row_na(io, key::String, label::String)
    @printf(io, "Estimator=%s | Label=%s | Available=false\n", key, label)
end

# Sparse direct two-way FE: y = alpha_i + psi_j + eps.
# Solves via the within-firm Laplacian L_F = diag(M_j) - X'*diag(1/M_i)*X.
# Returns (person_effects, firm_effects), both indexed by dense ID.
function compute_twoway_fe(y::Vector{Float64}, first_id::Vector{Int}, second_id::Vector{Int})
    N = maximum(first_id)
    J = maximum(second_id)
    n = length(y)

    person_count = zeros(Float64, N)
    person_sum   = zeros(Float64, N)
    for k in eachindex(first_id)
        person_count[first_id[k]] += 1.0
        person_sum[first_id[k]]   += y[k]
    end
    person_mean = person_sum ./ person_count

    y_tilde = y .- person_mean[first_id]

    rhs = zeros(Float64, J)
    for k in eachindex(first_id)
        rhs[second_id[k]] += y_tilde[k]
    end

    # X[i,j] = number of observations from person i to firm j (duplicates are summed)
    X    = sparse(first_id, second_id, ones(Float64, n), N, J)
    Dinv = spdiagm(0 => 1.0 ./ person_count)
    firm_count = vec(sum(X; dims=1))
    L_F  = spdiagm(0 => firm_count) - X' * Dinv * X

    # L_F is symmetric PSD (null space = constant per connected component).
    # ε*I breaks normalization degeneracy; variance components are invariant to it.
    psi = (L_F + 1e-8 * I) \ rhs

    alpha = zeros(Float64, N)
    for k in eachindex(first_id)
        alpha[first_id[k]] += y[k] - psi[second_id[k]]
    end
    alpha ./= person_count

    return alpha, psi
end

# Plug-in variance components from two-way FE on the full sample.
const MIN_KSS_ROWS = 500

function compute_full_raw_akm(y::Vector{Float64}, first_id::Vector{Int}, second_id::Vector{Int})
    alpha, psi = compute_twoway_fe(y, first_id, second_id)
    fe_firm   = psi[second_id]
    fe_person = alpha[first_id]
    return summarize_components(var(fe_firm), var(fe_person), cov(fe_firm, fe_person), var(y))
end

function main()
    positional = filter(a -> !startswith(a, "--"), ARGS)
    flags      = filter(a ->  startswith(a, "--"), ARGS)

    if length(positional) < 1
        println(stderr,
            "Usage:\n" *
            "  julia +1.8.5 --project=<env-with-Parquet2> src/estimate/akm_vchdfe.jl <path/to/edgelist.parquet> [outcome] [flags]\n" *
            "\n" *
            "Flags:\n" *
            "  --algorithm=<jla|exact>         Leverage algorithm (default: jla)\n" *
            "  --simulations=<N>               JLA simulations (default: 200)\n" *
            "  --leave-out-level=<match|obs>   KSS leave-out level (default: match)\n" *
            "  --print-level=<N>               Package verbosity (default: 1)\n" *
            "  --allow-sample-change           Allow the package to change the row set\n" *
            "  --vchdfe-project=<path>         Julia project/env where VarianceComponentsHDFE.jl is installed\n"
        )
        return 1
    end

    in_path = positional[1]
    outcome = (length(positional) >= 2) ? Symbol(positional[2]) : :lnR

    algorithm       = "jla"
    simulations     = 200
    leave_out_level = "match"
    print_level     = 1

    for flag in flags
        if startswith(flag, "--algorithm=")
            algorithm = lowercase(split(flag, '=', limit=2)[2])
        elseif startswith(flag, "--simulations=")
            simulations = parse(Int, split(flag, '=', limit=2)[2])
        elseif startswith(flag, "--leave-out-level=")
            leave_out_level = lowercase(split(flag, '=', limit=2)[2])
            leave_out_level in ("match", "obs") ||
                error("--leave-out-level must be match or obs; got $leave_out_level")
        elseif startswith(flag, "--print-level=")
            print_level = parse(Int, split(flag, '=', limit=2)[2])
        elseif flag == "--allow-sample-change"
            nothing  # accepted for backward compat, no longer needed
        elseif startswith(flag, "--vchdfe-project=")
            nothing
        else
            error("Unknown flag: $flag")
        end
    end

    simulations >= 0 || error("--simulations must be >= 0")
    print_level >= 0 || error("--print-level must be >= 0")
    isfile(in_path) || error("Input file not found: $in_path")

    chunk, sample_id, out_path = parse_input_path(in_path)
    suffixes = String[]
    algorithm != "jla" && push!(suffixes, algorithm)
    leave_out_level != "match" && push!(suffixes, leave_out_level)
    algorithm == "jla" && simulations != 200 && push!(suffixes, "jla$(simulations)")
    if !isempty(suffixes)
        out_path = replace(out_path, "estimates.txt" => "estimates_$(join(suffixes, "_")).txt")
    end
    mkpath(dirname(out_path))

    @printf("Reading: %s\n", in_path)
    df = Parquet2.readfile(in_path) |> DataFrame
    @printf("Loaded rows: %d\n", nrow(df))

    hasproperty(df, :frame_id_numeric) || error("Missing required column: frame_id_numeric")
    hasproperty(df, :person_id) || error("Missing required column: person_id")
    hasproperty(df, outcome) || error("Missing required column: $(outcome)")

    y_all = map(to_float_nan, df[!, outcome])
    keep = map(isfinite, y_all)
    d = df[keep, :]
    y = Float64.(y_all[keep])

    @printf("After non-missing %s filter: %d rows\n", String(outcome), length(y))
    length(y) > 0 || error("No usable observations after filtering non-missing $(outcome).")

    # Re-index IDs to dense 1..N so the package can build finite-size graph structures
    # regardless of the original ID range (e.g. frame_id_numeric up to 76M in real data).
    person_levels = sort(unique(d.person_id))
    firm_levels   = sort(unique(d.frame_id_numeric))
    person_dense  = Int.(indexin(d.person_id,  person_levels))
    firm_dense    = Int.(indexin(d.frame_id_numeric, firm_levels))
    @printf("Dense IDs: %d workers (max %d), %d firms (max %d)\n",
            length(person_levels), maximum(person_dense),
            length(firm_levels),   maximum(firm_dense))

    # --- Pass 1: Raw AKM on full sample via alternating-projections two-way FE ---
    @printf("Computing raw AKM (full data, %d obs)...\n", length(y))
    raw_row = compute_full_raw_akm(y, person_dense, firm_dense)

    # --- Pass 2: Attempt KSS via leave-one-out connected set ---
    settings = make_settings(algorithm, simulations, leave_out_level, outcome, print_level)
    @printf("Computing LOO connected set for KSS...\n")
    loo = get_leave_one_out_set(y, person_dense, firm_dense, settings, nothing)
    loo_rows = length(loo.obs)
    kss_available = loo_rows >= MIN_KSS_ROWS

    kss_sigma_common = NaN
    kss_sigma_method = "N/A"
    kss_ho = nothing
    kss_he = nothing
    if kss_available
        @printf("KSS LOO set: %d rows — computing bias-corrected estimates...\n", loo_rows)
        first_loo  = Int.(loo.first_id)
        second_loo = Int.(loo.second_id)
        y_loo      = Float64.(loo.y)
        kss = compute_akm_rows(y_loo, first_loo, second_loo, settings)
        kss_sigma_common = kss.sigma_common
        kss_sigma_method = kss.sigma_method
        kss_ho = kss.ho
        kss_he = kss.he
    else
        @printf("KSS LOO set collapsed to %d rows (< %d threshold) — KSS not available.\n",
                loo_rows, MIN_KSS_ROWS)
    end

    open(out_path, "w") do io
        @printf(io, "AKM/KSS Estimation (VarianceComponentsHDFE.jl)\n")
        @printf(io, "Chunk=%s | Sample=%s\n", chunk, sample_id)
        @printf(io, "Input=%s\n", in_path)
        @printf(io, "Outcome=%s | Algorithm=%s | LeaveOutLevel=%s\n",
                String(outcome), algorithm, leave_out_level)
        if algorithm == "jla"
            @printf(io, "Simulations=%d\n", simulations)
        end
        @printf(io, "InputRows=%d | OutcomeNonmissingRows=%d | KSSRows=%d | KSSAvailable=%s\n",
                nrow(df), nrow(d), loo_rows, string(kss_available))
        if !kss_available
            @printf(io, "KSSUnavailableReason=LOO set collapsed (%d of %d rows)\n",
                    loo_rows, nrow(d))
        end
        @printf(io, "OutcomeVariance=%+.6f\n", var(y))
        if kss_available
            @printf(io, "HomoskedasticSigmaCommon=%+.6f\n", kss_sigma_common)
            @printf(io, "HomoskedasticSigmaMethod=%s\n", kss_sigma_method)
        end
        write_row(io, "raw_akm", "Raw AKM", raw_row)
        if kss_available
            write_row(io, "kss_homoskedastic", "AKM + KSS (homoskedastic)", kss_ho)
            write_row(io, "kss_heteroskedastic", "AKM + KSS (heteroskedastic)", kss_he)
        else
            write_row_na(io, "kss_homoskedastic", "AKM + KSS (homoskedastic)")
            write_row_na(io, "kss_heteroskedastic", "AKM + KSS (heteroskedastic)")
        end
    end

    @printf("Wrote: %s\n", out_path)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
