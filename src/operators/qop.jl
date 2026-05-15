mutable struct QOp
    A::SparseMatrixCSC{Float64,Int}
    At::SparseMatrixCSC{Float64,Int}
    df_is::Vector{Float64}
    dw_is::Vector{Float64}
    diag_f::Vector{Float64}
    diag_w::Vector{Float64}
    n_firms::Int
    inv_sa2::Float64
    inv_sz2::Float64
    cross::Float64
    tmp_f::Vector{Float64}
    tmp_w::Vector{Float64}
end

function (op::QOp)(y::Vector{Float64}, x::Vector{Float64})
    n = length(x)
    nf = op.n_firms
    @views xf = x[1:nf]
    @views xw = x[(nf + 1):n]
    @views yf = y[1:nf]
    @views yw = y[(nf + 1):n]

    @. yf = op.inv_sa2 * op.diag_f * xf
    @. yw = op.inv_sz2 * op.diag_w * xw

    @. op.tmp_w = op.dw_is * xw
    mul!(op.tmp_f, op.A, op.tmp_w)
    @. op.tmp_f = op.df_is * op.tmp_f
    @. yf -= op.cross * op.tmp_f

    @. op.tmp_f = op.df_is * xf
    mul!(op.tmp_w, op.At, op.tmp_f)
    @. op.tmp_w = op.dw_is * op.tmp_w
    @. yw -= op.cross * op.tmp_w
    return y
end

function make_qop(problem::GMRFProblem, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    inv_sa2 = 1.0 / sigma_a^2
    inv_sz2 = 1.0 / sigma_z^2
    cross = rho / (sigma_a * sigma_z)
    return QOp(
        problem.A_prior,
        problem.At_prior,
        problem.df_is,
        problem.dw_is,
        problem.diag_f,
        problem.diag_w,
        problem.N_firms,
        inv_sa2,
        inv_sz2,
        cross,
        zeros(problem.N_firms),
        zeros(problem.N_workers),
    )
end
