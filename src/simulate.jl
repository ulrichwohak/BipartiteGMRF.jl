"""
    simulate(model::AbstractBipartiteModel, firm_ids, worker_ids;
             ρ, σ_a, σ_z, σ_ε, rng=Random.default_rng())

Simulate outcomes from a bipartite GMRF.

Draws latent effects `θ = [α; z] ~ GMRF(0, Q(ρ, σ_a, σ_z))` and generates
observations `y_k = α[firm_ids[k]] + z[worker_ids[k]] + ε_k` where
`ε_k ~ N(0, σ_ε²)`.

Returns a `NamedTuple` with fields:
- `y`: simulated outcome vector
- `firm_effects`: sampled firm effects α
- `worker_effects`: sampled worker effects z
- `firm_ids`, `worker_ids`: the input ID vectors

Residuals are drawn i.i.d.; the within-match residual correlation `rho_eps`
of `Weighting(observations=:effective)` is not simulated. To simulate that
data-generating process, add a match-level `N(0, √rho_eps ⋅ σ_ε)` draw shared
by repeated `(firm, worker)` pairs on top of an i.i.d. part scaled by
`√(1 − rho_eps)`.
"""
function simulate(
    model::AbstractBipartiteModel,
    firm_ids::AbstractVector{<:Integer},
    worker_ids::AbstractVector{<:Integer};
    ρ::Real,
    σ_a::Real,
    σ_z::Real,
    σ_ε::Real,
    X::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    β::Union{Nothing,AbstractVector{<:Real}}=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    length(firm_ids) == length(worker_ids) ||
        throw(ArgumentError("firm_ids and worker_ids must have the same length."))
    σ_ε > 0 || throw(ArgumentError("σ_ε must be positive; got $(σ_ε)."))

    g = model.graph
    all(1 .<= firm_ids .<= g.n_firms) ||
        throw(ArgumentError("firm_ids must be in 1:$(g.n_firms)."))
    all(1 .<= worker_ids .<= g.n_workers) ||
        throw(ArgumentError("worker_ids must be in 1:$(g.n_workers)."))

    # Draw latent effects from the GMRF
    gmrf = model(; ρ=ρ, σ_a=σ_a, σ_z=σ_z)
    θ = rand(rng, gmrf)
    α = θ[1:g.n_firms]
    z = θ[g.n_firms+1:end]

    # Generate observations: y_k = α[f_k] + z[w_k] + X[k,:]'β + ε_k
    K = length(firm_ids)
    y = Vector{Float64}(undef, K)
    @inbounds for k in 1:K
        y[k] = α[firm_ids[k]] + z[worker_ids[k]] + σ_ε * randn(rng)
    end
    if X !== nothing && β !== nothing
        y .+= X * β
    end

    return (
        y = y,
        firm_effects = α,
        worker_effects = z,
        firm_ids = firm_ids,
        worker_ids = worker_ids,
    )
end

"""
    simulate(model::AbstractBipartiteModel, A::SparseMatrixCSC;
             ρ, σ_a, σ_z, σ_ε, rng=Random.default_rng())

Simulate outcomes from a bipartite GMRF using an adjacency matrix to define
firm-worker pairs. Each nonzero entry `A[i,j]` generates one observation.
"""
function simulate(
    model::AbstractBipartiteModel,
    A::SparseMatrixCSC;
    ρ::Real,
    σ_a::Real,
    σ_z::Real,
    σ_ε::Real,
    rng::AbstractRNG=Random.default_rng(),
)
    rows, cols, _ = findnz(A)
    return simulate(model, rows, cols; ρ=ρ, σ_a=σ_a, σ_z=σ_z, σ_ε=σ_ε, rng=rng)
end
