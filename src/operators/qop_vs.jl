mutable struct QOpVS
    A::SparseMatrixCSC{Float64,Int}
    At::SparseMatrixCSC{Float64,Int}
    d_f::Vector{Float64}
    d_w::Vector{Float64}
    n_firms::Int
    inv_sa2::Float64
    inv_sz2::Float64
    cross::Float64
    rho_sq::Float64
    tmp_f::Vector{Float64}
    tmp_w::Vector{Float64}
end

function (op::QOpVS)(y::Vector{Float64}, x::Vector{Float64})
    n = length(x)
    nf = op.n_firms
    @views xf = x[1:nf]
    @views xw = x[(nf + 1):n]
    @views yf = y[1:nf]
    @views yw = y[(nf + 1):n]

    @inbounds for i in 1:nf
        yf[i] = (1.0 + op.rho_sq * (op.d_f[i] - 1.0)) * op.inv_sa2 * xf[i]
    end
    @inbounds for j in 1:(n - nf)
        yw[j] = (1.0 + op.rho_sq * (op.d_w[j] - 1.0)) * op.inv_sz2 * xw[j]
    end

    mul!(op.tmp_f, op.A, xw)
    @. yf -= op.cross * op.tmp_f
    mul!(op.tmp_w, op.At, xf)
    @. yw -= op.cross * op.tmp_w
    return y
end

function make_qop_vs(problem::GMRFProblem, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    inv_sa2 = 1.0 / sigma_a^2
    inv_sz2 = 1.0 / sigma_z^2
    return QOpVS(
        problem.A_prior,
        problem.At_prior,
        problem.d_f,
        problem.d_w,
        problem.N_firms,
        inv_sa2,
        inv_sz2,
        rho / (sigma_a * sigma_z),
        rho^2,
        zeros(problem.N_firms),
        zeros(problem.N_workers),
    )
end
