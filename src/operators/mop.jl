mutable struct MOp{Q}
    qop::Q
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

# Congruence-scaled posterior operator used by the variance-stable SLQ path:
# B + lambda * S * VtV * S. The scaling removes sigma-dependent terms that
# otherwise dominate both log determinants and cancel only after estimation.
mutable struct ScaledMOp{Q}
    qop::Q
    VtV::SparseMatrixCSC{Float64,Int}
    scale::Vector{Float64}
    scaled_x::Vector{Float64}
    tmp::Vector{Float64}
    lambda::Float64
end

function (op::ScaledMOp)(y::Vector{Float64}, x::Vector{Float64})
    op.qop(y, x)
    @. op.scaled_x = op.scale * x
    mul!(op.tmp, op.VtV, op.scaled_x)
    @. y = y + op.lambda * op.scale * op.tmp
    return y
end
