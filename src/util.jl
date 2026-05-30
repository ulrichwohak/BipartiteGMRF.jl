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
    return (
        rho = limit * tanh(Float64(params[1])),
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

# Inverse standard-normal CDF via Acklam's rational approximation
# (|absolute error| < 1.2e-9 over the open interval). Used for Wald confidence
# intervals so the package needs neither Distributions nor SpecialFunctions.
function norm_quantile(p::Real)
    0.0 < p < 1.0 || throw(ArgumentError("norm_quantile expects p in (0, 1); got $(p)."))
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow = 0.02425
    pq = Float64(p)
    if pq < plow
        q = sqrt(-2.0 * log(pq))
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
               ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
    elseif pq <= 1.0 - plow
        q = pq - 0.5
        r = q * q
        return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
               (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1.0)
    else
        q = sqrt(-2.0 * log(1.0 - pq))
        return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
                ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
    end
end
