mutable struct SLQWorkspace
    z::Vector{Float64}
    q::Vector{Float64}
    q0::Vector{Float64}
    w::Vector{Float64}
    alpha::Vector{Float64}
    beta::Vector{Float64}
end

SLQWorkspace(n::Int, k::Int) = SLQWorkspace(
    Vector{Float64}(undef, n),
    zeros(n),
    zeros(n),
    zeros(n),
    zeros(k),
    zeros(max(k - 1, 0)),
)

function slq_logdet_spd_mul_cached!(
    mulA!,
    n::Int,
    ws::SLQWorkspace;
    m::Int=30,
    k::Int=30,
    seed::Int=1,
    jitter::Real=0.0,
    ritz_floor::Real=0.0,
)
    k_eff = min(k, n)
    rng = MersenneTwister(seed)
    total = 0.0
    nz = sqrt(n)

    for _ in 1:m
        @inbounds for i in 1:n
            ws.z[i] = rand(rng, Bool) ? 1.0 : -1.0
        end
        @. ws.q = ws.z / nz
        fill!(ws.q0, 0.0)
        beta_prev = 0.0
        fill!(ws.alpha, 0.0)
        fill!(ws.beta, 0.0)
        t_stop = k_eff

        for t in 1:k_eff
            mulA!(ws.w, ws.q)
            jitter != 0 && (@. ws.w = ws.w + jitter * ws.q)
            ws.alpha[t] = dot(ws.q, ws.w)
            isfinite(ws.alpha[t]) || return NaN
            @. ws.w = ws.w - ws.alpha[t] * ws.q - beta_prev * ws.q0
            if t < k_eff
                recurrence_scale = max(abs(ws.alpha[t]), abs(beta_prev))
                beta_prev = norm(ws.w)
                isfinite(beta_prev) || return NaN
                ws.beta[t] = beta_prev
                # Relative breakdown is invariant to uniform precision
                # scaling; an absolute threshold truncates small SPD inputs.
                if beta_prev <= 1e-14 * recurrence_scale
                    t_stop = t
                    break
                end
                ws.q0 .= ws.q
                @. ws.q = ws.w / beta_prev
            end
        end

        T = SymTridiagonal(copy(view(ws.alpha, 1:t_stop)),
            t_stop > 1 ? copy(view(ws.beta, 1:(t_stop - 1))) : Float64[])
        E = try
            eigen(T)
        catch
            return NaN
        end
        vals = E.values
        u1 = @view E.vectors[1, :]
        s = 0.0
        @inbounds for j in 1:length(vals)
            # Never turn an indefinite/singular precision into a covariance
            # by flooring negative Ritz values. A failed trial is rejected by
            # the likelihood, without changing the graph or rho domain.
            lambda_j = vals[j]
            isfinite(lambda_j) && lambda_j > max(0.0, ritz_floor) || return NaN
            s += log(lambda_j) * (u1[j]^2)
        end
        total += (nz^2) * s
    end
    return total / m
end
