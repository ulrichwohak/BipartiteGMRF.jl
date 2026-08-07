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

function make_qop(model::BipartiteNormalizedModel, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    g = model.graph
    inv_sa2 = 1.0 / sigma_a^2
    inv_sz2 = 1.0 / sigma_z^2
    cross = rho / (sigma_a * sigma_z)
    return QOp(
        g.A, g.At,
        model.df_is, model.dw_is,
        ones(Float64, g.n_firms), ones(Float64, g.n_workers),
        g.n_firms, inv_sa2, inv_sz2, cross,
        zeros(g.n_firms), zeros(g.n_workers),
    )
end

function make_qop(model::BipartiteUnnormalizedModel, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    g = model.graph
    inv_sa2 = 1.0 / sigma_a^2
    inv_sz2 = 1.0 / sigma_z^2
    cross = rho / (sigma_a * sigma_z)
    return QOp(
        g.A, g.At,
        ones(Float64, g.n_firms), ones(Float64, g.n_workers),
        Float64.(g.d_f), Float64.(g.d_w),
        g.n_firms, inv_sa2, inv_sz2, cross,
        zeros(g.n_firms), zeros(g.n_workers),
    )
end

function make_qop(model::BipartiteSpectralModel, rho::Float64, sigma_a::Float64, sigma_z::Float64)
    g = model.graph
    s = model.spectral_is
    inv_sa2 = 1.0 / sigma_a^2
    inv_sz2 = 1.0 / sigma_z^2
    cross = rho / (sigma_a * sigma_z)
    return QOp(
        g.A, g.At,
        fill(s, g.n_firms), fill(s, g.n_workers),
        ones(Float64, g.n_firms), ones(Float64, g.n_workers),
        g.n_firms, inv_sa2, inv_sz2, cross,
        zeros(g.n_firms), zeros(g.n_workers),
    )
end
