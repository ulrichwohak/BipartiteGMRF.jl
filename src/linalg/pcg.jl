mutable struct PCGWorkspace
    x::Vector{Float64}
    r::Vector{Float64}
    z::Vector{Float64}
    p::Vector{Float64}
    Ap::Vector{Float64}
end

PCGWorkspace(n::Int) = PCGWorkspace(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n))

function pcg_solve!(
    ws::PCGWorkspace,
    mulA!,
    b::AbstractVector{<:Real};
    tol::Real=1e-6,
    maxiter::Int=700,
    Mdiag::Union{Nothing,AbstractVector{<:Real}}=nothing,
    diag_floor::Real=1e-12,
)
    x, r, z, p, Ap = ws.x, ws.r, ws.z, ws.p, ws.Ap
    n = length(b)
    fill!(x, 0.0)
    @inbounds for i in 1:n
        r[i] = Float64(b[i])
    end

    nb = norm(r)
    nb == 0.0 && return x, true, 0, 0.0

    if Mdiag === nothing
        z .= r
    else
        @inbounds for i in 1:n
            d = max(Float64(Mdiag[i]), Float64(diag_floor))
            z[i] = r[i] / d
        end
    end

    p .= z
    rz = dot(r, z)
    isfinite(rz) || return x, false, 0, Inf

    for it in 1:maxiter
        mulA!(Ap, p)
        denom = dot(p, Ap)
        isfinite(denom) && denom > 0 || return x, false, it, Inf
        alpha = rz / denom
        isfinite(alpha) || return x, false, it, Inf

        @inbounds @simd for i in 1:n
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        end

        relres = norm(r) / nb
        relres <= tol && return x, true, it, relres

        if Mdiag === nothing
            z .= r
        else
            @inbounds for i in 1:n
                d = max(Float64(Mdiag[i]), Float64(diag_floor))
                z[i] = r[i] / d
            end
        end

        rz_new = dot(r, z)
        isfinite(rz_new) && rz != 0.0 || return x, false, it, Inf
        beta = rz_new / rz
        isfinite(beta) || return x, false, it, Inf

        @inbounds @simd for i in 1:n
            p[i] = z[i] + beta * p[i]
        end
        rz = rz_new
    end
    return x, false, maxiter, norm(r) / nb
end
