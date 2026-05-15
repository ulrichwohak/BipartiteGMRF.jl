mutable struct MOp
    qop::Any
    VtV::SparseMatrixCSC{Float64,Int}
    tmp::Vector{Float64}
    lambda::Float64
end

function (op::MOp)(y::Vector{Float64}, x::Vector{Float64})
    op.qop(y, x)
    mul!(op.tmp, op.VtV, x)
    @. y = y + op.lambda * op.tmp
    return y
end
