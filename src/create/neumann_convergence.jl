#!/usr/bin/env julia
# ============================================================
# neumann_convergence.jl
# ------------------------------------------------------------
# Computes the Neumann-series convergence bound for the bipartite
# GMRF under the correlation parametrization:
#   Q(ρ) = [ σ_a⁻² D_a    -(ρ/(σ_aσ_z)) W
#           -(ρ/(σ_aσ_z)) W'          σ_z⁻² D_z ]
# The Neumann operator is M = D⁻¹C, whose spectral radius is
#   ρ(M) = |ρ| * sqrt(λ_max(K)),   K = D_a⁻¹ W D_z⁻¹ W'
# The series converges iff ρ(M) < 1 ⇔ |ρ| < 1 / sqrt(λ_max(K)).
# This script:
#   • builds W from the empirical edgelist (binary incidence)
#   • computes λ_max(K) via power iteration without forming K
#   • reports the admissible |ρ| bound and (optionally) ρ(M) for a
#     user-supplied ρ value
# Outputs an optional Parquet summary for reproducibility.
# ============================================================

using DataFrames
using Parquet2
using SparseArrays
using LinearAlgebra
using Printf

# ----------------------------
# Configuration via ENV
# ----------------------------
const EDGELIST_PATH = get(ENV, "EDGELIST_PATH", "temp/edgelist.parquet")
const PERSON_COL = Symbol(get(ENV, "PERSON_COL", "person_id"))
const FIRM_COL = Symbol(get(ENV, "FIRM_COL", "frame_id_numeric"))
const POWER_MAX_ITERS = parse(Int, get(ENV, "POWER_MAX_ITERS", "500"))
const POWER_TOL = parse(Float64, get(ENV, "POWER_TOL", "1e-9"))
const RHO_INPUT = get(ENV, "RHO", "")  # optional; if set, we evaluate ρ(M)
const OUT_PATH = get(ENV, "OUT_PATH", "temp/neumann_convergence.parquet")

# ----------------------------
# Helpers
# ----------------------------
function ensure_column(df::DataFrame, col::Symbol)
    col in propertynames(df) || error("Column $(String(col)) not found in $(EDGELIST_PATH)")
end

function build_incidence_matrix(df::DataFrame; person_col::Symbol, firm_col::Symbol)
    # Map external IDs to contiguous indices for sparse construction
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
    B .= sign.(B)  # binarize in case multiple spells per pair
    return SparseMatrixCSC{Float64,Int}(B), person_ids, firm_ids
end

function degree_vectors(W::SparseMatrixCSC{Float64,Int})
    deg_a = Array(sum(W, dims=2))[:, 1]
    deg_z = Array(sum(W, dims=1))[1, :]
    minimum(deg_a) > 0 || error("Some managers have zero degree; cannot form D_a⁻¹.")
    minimum(deg_z) > 0 || error("Some firms have zero degree; cannot form D_z⁻¹.")
    return deg_a, deg_z
end

function largest_eigen_manager_degree(W::SparseMatrixCSC{Float64,Int},
                                      deg_a::Vector{Float64},
                                      deg_z::Vector{Float64};
                                      maxiter::Int,
                                      tol::Float64)
    n_managers = size(W, 1)
    n_firms = size(W, 2)
    inv_deg_a = 1.0 ./ deg_a
    inv_deg_z = 1.0 ./ deg_z

    x = randn(n_managers)
    x ./= norm(x)
    tmp_firm = zeros(Float64, n_firms)
    y = similar(x)
    λ_old = 0.0

    for iter in 1:maxiter
        mul!(tmp_firm, transpose(W), x)   # tmp_firm = W' * x
        @inbounds for j in eachindex(tmp_firm)
            tmp_firm[j] *= inv_deg_z[j]
        end
        mul!(y, W, tmp_firm)              # y = W * tmp_firm
        @inbounds for i in eachindex(y)
            y[i] *= inv_deg_a[i]
        end
        λ = dot(x, y)
        nrm = norm(y)
        nrm == 0 && error("Encountered zero vector during power iteration; graph may be empty.")
        x .= y ./ nrm
        if iter > 1 && abs(λ - λ_old) <= tol * max(1.0, abs(λ))
            return λ, iter
        end
        λ_old = λ
    end
    return λ_old, maxiter
end

function largest_eigen_firm_degree(W::SparseMatrixCSC{Float64,Int},
                                   deg_a::Vector{Float64},
                                   deg_z::Vector{Float64};
                                   maxiter::Int,
                                   tol::Float64)
    n_managers = size(W, 1)
    n_firms = size(W, 2)
    inv_deg_a = 1.0 ./ deg_a
    inv_deg_z = 1.0 ./ deg_z

    x = randn(n_firms)
    x ./= norm(x)
    tmp_manager = zeros(Float64, n_managers)
    y = similar(x)
    λ_old = 0.0

    for iter in 1:maxiter
        mul!(tmp_manager, W, x)              # tmp_manager = W * x
        @inbounds for i in eachindex(tmp_manager)
            tmp_manager[i] *= inv_deg_a[i]
        end
        mul!(y, transpose(W), tmp_manager)   # y = W' * tmp_manager
        @inbounds for j in eachindex(y)
            y[j] *= inv_deg_z[j]
        end
        λ = dot(x, y)
        nrm = norm(y)
        nrm == 0 && error("Encountered zero vector during power iteration; graph may be empty.")
        x .= y ./ nrm
        if iter > 1 && abs(λ - λ_old) <= tol * max(1.0, abs(λ))
            return λ, iter
        end
        λ_old = λ
    end
    return λ_old, maxiter
end

function largest_eigen_manager_identity(W::SparseMatrixCSC{Float64,Int};
                                        maxiter::Int,
                                        tol::Float64)
    n_managers = size(W, 1)
    n_firms = size(W, 2)
    x = randn(n_managers)
    x ./= norm(x)
    tmp_firm = zeros(Float64, n_firms)
    y = similar(x)
    λ_old = 0.0

    for iter in 1:maxiter
        mul!(tmp_firm, transpose(W), x)   # tmp_firm = W' * x
        mul!(y, W, tmp_firm)              # y = W * tmp_firm
        λ = dot(x, y)
        nrm = norm(y)
        nrm == 0 && error("Encountered zero vector during power iteration; graph may be empty.")
        x .= y ./ nrm
        if iter > 1 && abs(λ - λ_old) <= tol * max(1.0, abs(λ))
            return λ, iter
        end
        λ_old = λ
    end
    return λ_old, maxiter
end

function largest_eigen_firm_identity(W::SparseMatrixCSC{Float64,Int};
                                     maxiter::Int,
                                     tol::Float64)
    n_managers = size(W, 1)
    n_firms = size(W, 2)
    x = randn(n_firms)
    x ./= norm(x)
    tmp_manager = zeros(Float64, n_managers)
    y = similar(x)
    λ_old = 0.0

    for iter in 1:maxiter
        mul!(tmp_manager, W, x)              # tmp_manager = W * x
        mul!(y, transpose(W), tmp_manager)   # y = W' * tmp_manager
        λ = dot(x, y)
        nrm = norm(y)
        nrm == 0 && error("Encountered zero vector during power iteration; graph may be empty.")
        x .= y ./ nrm
        if iter > 1 && abs(λ - λ_old) <= tol * max(1.0, abs(λ))
            return λ, iter
        end
        λ_old = λ
    end
    return λ_old, maxiter
end
function write_summary(path::String, df::DataFrame)
    dir = dirname(path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    Parquet2.writefile(path, df)
end

# ----------------------------
# Main
# ----------------------------
isfile(EDGELIST_PATH) || error("Missing edgelist at $(EDGELIST_PATH)")
df = Parquet2.readfile(EDGELIST_PATH) |> DataFrame
ensure_column(df, PERSON_COL)
ensure_column(df, FIRM_COL)
dropmissing!(df, [PERSON_COL, FIRM_COL])

W, person_ids, firm_ids = build_incidence_matrix(df; person_col=PERSON_COL, firm_col=FIRM_COL)
deg_a, deg_z = degree_vectors(W)

@printf("Managers: %d | Firms: %d | Bipartite edges: %d\n",
        size(W, 1), size(W, 2), nnz(W))
@printf("Min/Max manager degree: %.0f / %.0f\n", minimum(deg_a), maximum(deg_a))
@printf("Min/Max firm degree: %.0f / %.0f\n", minimum(deg_z), maximum(deg_z))

λ_deg_mgr, it_deg_mgr = largest_eigen_manager_degree(W, deg_a, deg_z; maxiter=POWER_MAX_ITERS, tol=POWER_TOL)
λ_deg_mgr < 0 && error("λ_max(K_degree_mgr) is negative; this should not happen for degree-normalized K.")
ρ_bound_deg_mgr = 1 / sqrt(λ_deg_mgr)

λ_deg_firm, it_deg_firm = largest_eigen_firm_degree(W, deg_a, deg_z; maxiter=POWER_MAX_ITERS, tol=POWER_TOL)
λ_deg_firm < 0 && error("λ_max(K_degree_firm) is negative; this should not happen for degree-normalized K.")
ρ_bound_deg_firm = 1 / sqrt(λ_deg_firm)

λ_id_mgr, it_id_mgr = largest_eigen_manager_identity(W; maxiter=POWER_MAX_ITERS, tol=POWER_TOL)
λ_id_mgr < 0 && error("λ_max(K_identity_mgr) is negative; graph may be malformed.")
ρ_bound_id_mgr = 1 / sqrt(λ_id_mgr)

λ_id_firm, it_id_firm = largest_eigen_firm_identity(W; maxiter=POWER_MAX_ITERS, tol=POWER_TOL)
λ_id_firm < 0 && error("λ_max(K_identity_firm) is negative; graph may be malformed.")
ρ_bound_id_firm = 1 / sqrt(λ_id_firm)

@printf("\nSpectrum summary (correlation parametrization):\n")
@printf("  Degree-normalized (manager graph)   K = D_a^{-1} W D_z^{-1} W':   λ_max ≈ %.6f (iters: %d) ⇒ |ρ| < %.6f\n",
        λ_deg_mgr, it_deg_mgr, ρ_bound_deg_mgr)
@printf("  Degree-normalized (firm graph)      K = D_z^{-1} W' D_a^{-1} W:   λ_max ≈ %.6f (iters: %d) ⇒ |ρ| < %.6f\n",
        λ_deg_firm, it_deg_firm, ρ_bound_deg_firm)
@printf("  Identity-scaled  (manager graph)    K = W W':                     λ_max ≈ %.6f (iters: %d) ⇒ |ρ| < %.6f\n",
        λ_id_mgr, it_id_mgr, ρ_bound_id_mgr)
@printf("  Identity-scaled  (firm graph)       K = W' W:                     λ_max ≈ %.6f (iters: %d) ⇒ |ρ| < %.6f\n",
        λ_id_firm, it_id_firm, ρ_bound_id_firm)

ρ_input_val = isempty(RHO_INPUT) ? nothing : parse(Float64, RHO_INPUT)
ρM_deg_mgr = nothing
ρM_deg_firm = nothing
ρM_id_mgr = nothing
ρM_id_firm = nothing
if ρ_input_val !== nothing
    ρM_deg_mgr = abs(ρ_input_val) * sqrt(λ_deg_mgr)
    ρM_deg_firm = abs(ρ_input_val) * sqrt(λ_deg_firm)
    ρM_id_mgr = abs(ρ_input_val) * sqrt(λ_id_mgr)
    ρM_id_firm = abs(ρ_input_val) * sqrt(λ_id_firm)
    status_deg_mgr = ρM_deg_mgr < 1 ? "OK: converges" : "FAILS: diverges"
    status_deg_firm = ρM_deg_firm < 1 ? "OK: converges" : "FAILS: diverges"
    status_id_mgr = ρM_id_mgr < 1 ? "OK: converges" : "FAILS: diverges"
    status_id_firm = ρM_id_firm < 1 ? "OK: converges" : "FAILS: diverges"
    @printf("\nFor |ρ| = %.6f:\n", abs(ρ_input_val))
    @printf("  Degree-normalized (manager) ρ(M) ≈ %.6f  [%s]\n", ρM_deg_mgr, status_deg_mgr)
    @printf("  Degree-normalized (firm)    ρ(M) ≈ %.6f  [%s]\n", ρM_deg_firm, status_deg_firm)
    @printf("  Identity-scaled  (manager)  ρ(M) ≈ %.6f  [%s]\n", ρM_id_mgr, status_id_mgr)
    @printf("  Identity-scaled  (firm)     ρ(M) ≈ %.6f  [%s]\n", ρM_id_firm, status_id_firm)
end

summary_df = DataFrame(
    n_managers = size(W, 1),
    n_firms = size(W, 2),
    n_edges = nnz(W),
    lambda_max_K_degree_manager = λ_deg_mgr,
    rho_bound_degree_manager = ρ_bound_deg_mgr,
    power_iters_degree_manager = it_deg_mgr,
    lambda_max_K_degree_firm = λ_deg_firm,
    rho_bound_degree_firm = ρ_bound_deg_firm,
    power_iters_degree_firm = it_deg_firm,
    lambda_max_K_identity_manager = λ_id_mgr,
    rho_bound_identity_manager = ρ_bound_id_mgr,
    power_iters_identity_manager = it_id_mgr,
    lambda_max_K_identity_firm = λ_id_firm,
    rho_bound_identity_firm = ρ_bound_id_firm,
    power_iters_identity_firm = it_id_firm,
    power_tol = POWER_TOL,
    rho_input = ρ_input_val === nothing ? missing : ρ_input_val,
    spectral_radius_M_degree_manager = ρM_deg_mgr === nothing ? missing : ρM_deg_mgr,
    spectral_radius_M_degree_firm = ρM_deg_firm === nothing ? missing : ρM_deg_firm,
    spectral_radius_M_identity_manager = ρM_id_mgr === nothing ? missing : ρM_id_mgr,
    spectral_radius_M_identity_firm = ρM_id_firm === nothing ? missing : ρM_id_firm,
)

if !isempty(OUT_PATH)
    write_summary(OUT_PATH, summary_df)
    @printf("Results saved to %s\n", OUT_PATH)
end
