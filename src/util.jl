const BIG_NLL = 1e30

to_float_nan(x) = x isa Missing ? NaN : Float64(x)

function normalize_decomp_target(target::Symbol)
    target in (:estimation, :likelihood, :likelihood_mean) && return :estimation
    target in (:personyear, :edge) && return target
    throw(ArgumentError("Unknown decomposition target $(target). Use :estimation, :personyear, or :edge."))
end

rhoeps_from_unconstrained(u::Real) = 0.999 / (1.0 + exp(-Float64(u)))

function rhoeps_to_unconstrained(r::Real)
    x = Float64(r) / 0.999
    x = min(max(x, 1e-12), 1.0 - 1e-12)
    return log(x / (1.0 - x))
end

function effective_match_weights(T::AbstractVector{<:Real}, rho_eps::Float64)
    0.0 <= rho_eps < 1.0 ||
        throw(ArgumentError("rho_eps must satisfy 0 <= rho_eps < 1; got $(rho_eps)."))
    return Float64.(T) ./ (1.0 .+ rho_eps .* (Float64.(T) .- 1.0))
end

function edge_ssw(x)
    mu = mean(x)
    s = 0.0
    @inbounds for v in x
        d = Float64(v) - mu
        s += d * d
    end
    return s
end

function leading_singular_value(
    A::SparseMatrixCSC{Float64,Int},
    At::SparseMatrixCSC{Float64,Int};
    maxiter::Int=200,
    tol::Float64=1e-10,
    seed::Int=12345,
)
    m = size(A, 2)
    rng = MersenneTwister(seed)
    x = randn(rng, m)
    x ./= norm(x)
    sigma_old = 0.0
    tmp = Vector{Float64}(undef, size(A, 1))
    for _ in 1:maxiter
        mul!(tmp, A, x)
        mul!(x, At, tmp)
        lambda = norm(x)
        lambda > 0 || return 0.0
        x ./= lambda
        sigma_new = sqrt(lambda)
        if abs(sigma_new - sigma_old) / max(sigma_new, 1e-15) < tol
            return sigma_new
        end
        sigma_old = sigma_new
    end
    return sigma_old
end

function is_forest(A::SparseMatrixCSC{Float64,Int})
    n_f, n_w = size(A)
    parent = collect(1:(n_f + n_w))

    function find_root(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end

    rows, cols, _ = findnz(A)
    for (i, j) in zip(rows, cols)
        u = i
        v = n_f + j
        ru = find_root(u)
        rv = find_root(v)
        ru == rv && return false
        parent[ru] = rv
    end
    return true
end

function unpack_params(params::AbstractVector{<:Real}; rho_limit::Real=0.99)
    limit = validate_rho_limit(rho_limit)
    # Clamp to strictly interior: tanh saturates to ±1.0 for |x| ≳ 19,
    # which would put ρ exactly on the boundary and trip validation in
    # precision_matrix.  nextfloat(-limit) < ρ < prevfloat(limit) keeps
    # Q positive-definite and the optimizer probe safe.
    rho_raw = limit * tanh(Float64(params[1]))
    rho = clamp(rho_raw, nextfloat(-limit), prevfloat(limit))
    return (
        rho = rho,
        sigma_a = exp(Float64(params[2])),
        sigma_z = exp(Float64(params[3])),
        sigma_epsilon = exp(Float64(params[4])),
    )
end

default_rho_start(rho_limit::Real) = min(0.5, 0.5 * validate_rho_limit(rho_limit))

function initial_params(
    rho_fixed::Union{Nothing,Float64},
    estimate_rho_eps::Bool;
    rho_limit::Real=0.99,
)
    limit = validate_rho_limit(rho_limit)
    p = rho_fixed === nothing ?
        [atanh(default_rho_start(limit) / limit), log(0.7), log(0.04), log(0.4)] :
        [log(0.7), log(0.04), log(0.4)]
    estimate_rho_eps && push!(p, rhoeps_to_unconstrained(0.5))
    return p
end

function full_params(
    params::Vector{Float64},
    rho_fixed::Union{Nothing,Float64},
    estimate_rho_eps::Bool;
    rho_limit::Real=0.99,
)
    if rho_fixed === nothing
        return params
    end
    limit = validate_rho_limit(rho_limit)
    abs(rho_fixed) < limit ||
        throw(ArgumentError("rho_fixed must lie in (-$(limit), $(limit)); got $(rho_fixed)."))
    rho_code = atanh(rho_fixed / limit)
    return estimate_rho_eps ?
        [rho_code, params[1], params[2], params[3], params[4]] :
        [rho_code, params[1], params[2], params[3]]
end

function finite_or_big(x::Real)
    return isfinite(x) ? Float64(x) : BIG_NLL
end
