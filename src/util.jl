const BIG_NLL = 1e30

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

# AR(1) within-firm error correlation eta in (-1, 1), estimated on an
# unconstrained scale. tanh saturates to +-1.0 for |u| ~> 19, so clamp to the
# strictly-interior Float64 values to keep the correlation matrix non-singular
# (same philosophy as the rho codec in unpack_params).
eta_from_unconstrained(u::Real) = clamp(tanh(Float64(u)), nextfloat(-1.0), prevfloat(1.0))

function eta_to_unconstrained(e::Real)
    -1.0 < e < 1.0 ||
        throw(ArgumentError("eta must satisfy -1 < eta < 1; got $(e)."))
    return atanh(clamp(Float64(e), nextfloat(-1.0), prevfloat(1.0)))
end

function effective_match_weights(T::AbstractVector{<:Real}, rho_eps::Float64)
    0.0 <= rho_eps < 1.0 ||
        throw(ArgumentError("rho_eps must satisfy 0 <= rho_eps < 1; got $(rho_eps)."))
    return Float64.(T) ./ (1.0 .+ rho_eps .* (Float64.(T) .- 1.0))
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

# The rho_limit passed to the parameter codecs below always comes from a
# constructed model, where it was validated once; they only convert.
function unpack_params(params::AbstractVector{<:Real}; rho_limit::Real=0.99)
    limit = Float64(rho_limit)
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

default_rho_start(rho_limit::Real) = min(0.5, 0.5 * Float64(rho_limit))

# ─── init: caller-supplied starting values ────────────────────────────────
#
# `init` is a NamedTuple in structural units, every field optional; whatever is
# missing falls back to the heuristic below, field by field. Sigmas arrive here
# already divided by `stats.y_std` (the caller writes them in original outcome
# units — see `_initial_point`).

const INIT_KEYS = (:rho, :sigma_a, :sigma_z, :sigma_epsilon, :rho_eps, :eta,
                   :omega, :phi, :r, :delta)

# The subset whose availability depends on the fit: `rho` can be pinned by
# fix_rho, and the rest belong to optional blocks. Validation iterates over
# *this* tuple rather than over the caller's `status`, so a `status` that
# forgets a key fails loud (:absent) instead of letting that init field slip
# through unvalidated and unused. The three sigmas are always estimated.
const INIT_OPTIONAL_KEYS = (:rho, :rho_eps, :eta, :omega, :phi, :r, :delta)

# `params(result)` also carries :beta, which is a profiled-out coefficient
# vector rather than a starting value. Accept and ignore it so that
# `init = params(result)` works without hand-editing.
const INIT_IGNORED_KEYS = (:beta,)

# Why a parameter might not be estimated in this fit, keyed by the init field.
const INIT_ABSENT_HINT = (
    rho_eps = "pass weighting=Weighting(observations=:effective, estimate_rho_eps=true)",
    eta = "pass error_eta=:estimate",
    omega = "pass error_groups with more than one group size class",
    phi = "pass error_blocks=:iw",
    r = "pass error_blocks=:iw",
    delta = "pass error_blocks=:iw",
)

const INIT_PINNED_HINT = (
    rho = "fix_rho holds rho at its given value",
    rho_eps = "estimate_rho_eps is false, so rho_eps is held fixed",
    eta = "error_eta was given a numeric value",
    r = "the solver holds r at a fixed value",
    delta = "the solver holds delta at a fixed value",
)

init_field(::Nothing, ::Symbol) = nothing
init_field(init::NamedTuple, name::Symbol) = get(init, name, nothing)

function validate_positive_init(value, name::Symbol)
    value isa Real ||
        throw(ArgumentError("init.$(name) must be a real number; got $(typeof(value))."))
    x = Float64(value)
    isfinite(x) && x > 0 ||
        throw(ArgumentError("init.$(name) must be finite and positive; got $(value)."))
    return x
end

function validate_real_init(value, name::Symbol)
    value isa Real ||
        throw(ArgumentError("init.$(name) must be a real number; got $(typeof(value))."))
    return Float64(value)
end

"""
    validate_init(init, status; rho_limit=0.99)

Check an `init` NamedTuple against the parameters this fit actually estimates.
`status` maps each optional block parameter (`rho`, `rho_eps`, `eta`, `omega`,
`phi`, `r`, `delta`) to `:free`, `:absent`, or — for a parameter that is part
of the fit but held fixed — the value it is pinned at (`:pinned` when that
value is not a number). A field for an `:absent` parameter is an error;
silently ignoring it would look like a working warm start. A field for a pinned
parameter is ignored, with a warning *unless* it already agrees with the pinned
value, which is what keeps `init = params(result)` quiet when re-fitting the
same configuration. Fields whose value is `nothing` count as not supplied.
"""
function validate_init(
    init::Union{Nothing,NamedTuple},
    status::NamedTuple;
    rho_limit::Real=0.99,
)
    init === nothing && return nothing

    for k in keys(init)
        (k in INIT_KEYS || k in INIT_IGNORED_KEYS) || throw(ArgumentError(
            "Unknown init field :$(k). Accepted fields: " *
            join(string.(INIT_KEYS), ", ") * " (:beta is accepted and ignored)."))
    end

    limit = Float64(rho_limit)
    for k in INIT_OPTIONAL_KEYS
        v = init_field(init, k)
        v === nothing && continue
        st = get(status, k, :absent)
        if st === :absent
            hint = get(INIT_ABSENT_HINT, k, "it is not part of this fit")
            throw(ArgumentError(
                "init.$(k) was given, but this fit does not estimate $(k); $(hint)."))
        elseif st === :free
            validate_init_domain(v, k, limit)
        else
            # Pinned: the value is discarded, so its domain is not checked.
            # Stay quiet when it already agrees with the value in effect —
            # re-fitting the same configuration with init = params(result)
            # legitimately carries the pinned parameter along.
            pinned = st isa Real ? Float64(st) : nothing
            agrees = pinned !== nothing && v isa Real &&
                isapprox(Float64(v), pinned; rtol=1e-8, atol=1e-12)
            if !agrees
                hint = get(INIT_PINNED_HINT, k, "it is held fixed for this fit")
                @warn "init.$(k) is ignored: $(hint)."
            end
        end
    end

    # The three sigmas are estimated in every fit, so they are always checked.
    for name in (:sigma_a, :sigma_z, :sigma_epsilon)
        v = init_field(init, name)
        v === nothing || validate_positive_init(v, name)
    end

    return nothing
end

# Domain of one init field, given that the fit does estimate it. `omega` and
# `eta` are checked by their codecs (init_omega_codes, eta_to_unconstrained),
# which have the length and correlation bounds respectively.
function validate_init_domain(v, name::Symbol, rho_limit::Float64)
    if name === :rho
        abs(validate_real_init(v, name)) < rho_limit || throw(ArgumentError(
            "init.rho must satisfy |rho| < rho_limit = $(rho_limit); got $(v). " *
            "With rho_limit=:auto the limit is resolved from the " *
            "non-backtracking spectrum and can sit well below 0.99."))
    elseif name === :rho_eps
        # rhoeps_to_unconstrained clamps silently at 0.999, so reject rather
        # than quietly relocate the start.
        0.0 <= validate_real_init(v, name) < 0.999 || throw(ArgumentError(
            "init.rho_eps must satisfy 0 <= rho_eps < 0.999; got $(v)."))
    elseif name === :phi
        validate_positive_init(v, name)
    elseif name === :delta
        # omega_bar = phi*delta/(delta-2) only exists for delta > 2.
        d = validate_real_init(v, name)
        isfinite(d) && d > 2.0 || throw(ArgumentError(
            "init.delta must be finite and greater than 2; got $(v)."))
    elseif name === :r
        # The exact lower end of the PD domain depends on the largest block
        # size and is checked where that is known (optimize_emiw).
        -1.0 < validate_real_init(v, name) < 1.0 || throw(ArgumentError(
            "init.r must satisfy -1 < r < 1; got $(v)."))
    end
    return nothing
end

"""
    rescale_init_sigmas(init, y_std)

Map an `init` written in original outcome units onto the standardized scale the
optimizer works in. `suffstats(...; standardize=true)` divides `y` by `y_std`
and every scale is multiplied back by `y_std` on the way out, so a caller's
`sigma_a`, `sigma_z`, `sigma_epsilon` (and the EMIW scale `phi`, a variance)
are the ones that need converting. `rho`, `rho_eps`, `eta`, `omega` and `r` are
scale-free and pass through untouched.
"""
function rescale_init_sigmas(init::Union{Nothing,NamedTuple}, y_std::Real)
    init === nothing && return nothing
    s = Float64(y_std)
    s == 1.0 && return init
    out = init
    for name in (:sigma_a, :sigma_z, :sigma_epsilon)
        v = init_field(init, name)
        v === nothing && continue
        out = merge(out, NamedTuple{(name,)}((validate_positive_init(v, name) / s,)))
    end
    phi = init_field(init, :phi)
    phi === nothing ||
        (out = merge(out, (phi = validate_positive_init(phi, :phi) / s^2,)))
    return out
end

"""
    init_omega_codes(init, n_omega)

Starting values for the `log omega` ladder (free error classes `2..C`). Accepts
either `n_omega` entries (the free classes alone) or `n_omega + 1` (the full
ladder as reported in `error_class_variances`, whose leading entry is pinned at
1). Falls back to `omega = 1` for every class.
"""
function init_omega_codes(init::Union{Nothing,NamedTuple}, n_omega::Int)
    v = init_field(init, :omega)
    v === nothing && return zeros(Float64, n_omega)
    v isa AbstractVector || throw(ArgumentError(
        "init.omega must be a vector of class variances; got $(typeof(v))."))
    w = Float64[validate_real_init(x, :omega) for x in v]
    if length(w) == n_omega + 1
        isapprox(w[1], 1.0; atol=1e-8) || throw(ArgumentError(
            "init.omega of length $(n_omega + 1) is read as the full class ladder, " *
            "whose first entry is pinned at 1; got $(w[1])."))
        w = w[2:end]
    elseif length(w) != n_omega
        throw(ArgumentError(
            "init.omega must have $(n_omega) entries (the free classes 2..$(n_omega + 1)) " *
            "or $(n_omega + 1) (the full ladder with a leading 1); got $(length(w))."))
    end
    for (i, x) in enumerate(w)
        isfinite(x) && x > 0 ||
            throw(ArgumentError("init.omega entries must be finite and positive; entry $(i) is $(x)."))
    end
    return log.(w)
end

"""
    init_eta_code(init)

Starting value for the AR(1) within-firm correlation on the unconstrained
scale, defaulting to `eta = 0.5`.
"""
function init_eta_code(init::Union{Nothing,NamedTuple})
    v = init_field(init, :eta)
    return eta_to_unconstrained(v === nothing ? 0.5 : Float64(v))
end

# Starting values assume the outcome has been standardized to unit variance
# (the suffstats default): a mid-range rho, most variance in the residual and
# the firm side, little in the worker side. They are heuristics, not tuned
# per dataset; the optimizer is expected to move far from them. `init` replaces
# them field by field.
function initial_params(
    rho_fixed::Union{Nothing,Float64},
    estimate_rho_eps::Bool;
    rho_limit::Real=0.99,
    init::Union{Nothing,NamedTuple}=nothing,
)
    limit = Float64(rho_limit)
    rho0 = init_field(init, :rho)
    sa0 = init_field(init, :sigma_a)
    sz0 = init_field(init, :sigma_z)
    se0 = init_field(init, :sigma_epsilon)
    p = Float64[]
    if rho_fixed === nothing
        # Guarded here too, not only in validate_init: a bare atanh past the
        # limit would otherwise surface as a contextless DomainError.
        rho0 === nothing || validate_init_domain(rho0, :rho, limit)
        r0 = rho0 === nothing ? default_rho_start(limit) : Float64(rho0)
        push!(p, atanh(r0 / limit))
    end
    push!(p, log(sa0 === nothing ? 0.7 : validate_positive_init(sa0, :sigma_a)))
    push!(p, log(sz0 === nothing ? 0.04 : validate_positive_init(sz0, :sigma_z)))
    push!(p, log(se0 === nothing ? 0.4 : validate_positive_init(se0, :sigma_epsilon)))
    if estimate_rho_eps
        re0 = init_field(init, :rho_eps)
        if re0 === nothing
            push!(p, rhoeps_to_unconstrained(0.5))
        else
            validate_init_domain(re0, :rho_eps, limit)
            push!(p, rhoeps_to_unconstrained(Float64(re0)))
        end
    end
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
    limit = Float64(rho_limit)
    abs(rho_fixed) < limit ||
        throw(ArgumentError("rho_fixed must lie in (-$(limit), $(limit)); got $(rho_fixed)."))
    rho_code = atanh(rho_fixed / limit)
    return vcat(rho_code, params)
end

function finite_or_big(x::Real)
    return isfinite(x) ? Float64(x) : BIG_NLL
end
