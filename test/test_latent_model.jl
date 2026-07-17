using GaussianMarkovRandomFields: precision_matrix, model_name, hyperparameters, constraints
import GaussianMarkovRandomFields

@testset "LatentModel interface" begin
    A = sparse(
        [1, 1, 2, 2, 3],
        [1, 2, 2, 3, 3],
        ones(5), 3, 3,
    )
    params = (ρ=0.5, σ_a=1.0, σ_z=0.8)
    models = [
        BipartiteNormalizedModel(A),
        BipartiteUnnormalizedModel(A),
        BipartiteSpectralModel(A),
    ]

    @testset "$(nameof(typeof(m)))" for m in models
        # length
        @test length(m) == 6
        @test length(m) == m.graph.n_firms + m.graph.n_workers

        # hyperparameters returns name→type pairs
        hp = hyperparameters(m)
        @test hp isa NamedTuple
        @test keys(hp) == (:ρ, :σ_a, :σ_z)
        @test all(v -> v === Real, values(hp))

        # model_name returns a Symbol
        mn = model_name(m)
        @test mn isa Symbol
        @test startswith(string(mn), "bipartite_")

        # constraints returns nothing (unconstrained)
        @test constraints(m) === nothing

        # mean returns zero vector of correct length
        μ = GaussianMarkovRandomFields.mean(m; params...)
        @test μ == zeros(6)

        # precision_matrix returns a sparse matrix of correct size
        Q = precision_matrix(m; params...)
        @test size(Q) == (6, 6)
        @test Q isa SparseMatrixCSC
        @test issymmetric(Matrix(Q))
        @test isposdef(Symmetric(Matrix(Q)))

        # callable returns a GMRF
        gmrf = m(; params...)
        @test gmrf isa GaussianMarkovRandomFields.AbstractGMRF

        # rand produces vector of correct length
        x = rand(gmrf)
        @test length(x) == 6
        @test all(isfinite, x)

        # rand with RNG is reproducible
        x1 = rand(MersenneTwister(42), gmrf)
        x2 = rand(MersenneTwister(42), gmrf)
        @test x1 == x2

        # var returns marginal variances
        v = GaussianMarkovRandomFields.var(gmrf)
        @test length(v) == 6
        @test all(x -> x > 0, v)

        # logpdf returns finite scalar
        lp = GaussianMarkovRandomFields.logpdf(gmrf, x)
        @test isfinite(lp)
        @test lp isa Real
    end

    @testset "VarianceStable" begin
        A_tree = sparse([1, 2], [1, 1], ones(2), 2, 1)
        m = BipartiteVarianceStableModel(A_tree)
        @test length(m) == 3
        @test model_name(m) == :bipartite_variance_stable
        @test constraints(m) === nothing

        Q = precision_matrix(m; params...)
        @test size(Q) == (3, 3)
        @test isposdef(Symmetric(Matrix(Q)))

        gmrf = m(; params...)
        x = rand(gmrf)
        @test length(x) == 3
    end

    @testset "model_name values" begin
        @test model_name(BipartiteNormalizedModel(A)) == :bipartite_normalized
        @test model_name(BipartiteUnnormalizedModel(A)) == :bipartite_unnormalized
        @test model_name(BipartiteSpectralModel(A)) == :bipartite_spectral
    end
end

@testset "to_model" begin
    A = sparse([1, 1, 2, 2, 3], [1, 2, 2, 3, 3], ones(5), 3, 3)

    @test to_model(NormalizedPrior(), A) isa BipartiteNormalizedModel
    @test to_model(UnnormalizedPrior(), A) isa BipartiteUnnormalizedModel
    @test to_model(SpectralPrior(), A) isa BipartiteSpectralModel
    @test to_model(VarianceStablePrior(), sparse([1, 2], [1, 1], ones(2), 2, 1)) isa BipartiteVarianceStableModel

    # rho_limit propagates
    m = to_model(NormalizedPrior(rho_limit=0.95), A)
    @test m.rho_limit == 0.95
end

@testset "simulate" begin
    A = sparse([1, 1, 2, 2, 3], [1, 2, 2, 3, 3], ones(5), 3, 3)
    model = BipartiteNormalizedModel(A)
    params = (ρ=0.5, σ_a=1.0, σ_z=0.8, σ_ε=0.3)

    # adjacency-matrix overload
    sim = simulate(model, A; params..., rng=MersenneTwister(42))
    @test length(sim.y) == 5
    @test length(sim.firm_effects) == 3
    @test length(sim.worker_effects) == 3
    @test all(isfinite, sim.y)

    # reproducible with same rng seed
    sim2 = simulate(model, A; params..., rng=MersenneTwister(42))
    @test sim.y == sim2.y
    @test sim.firm_effects == sim2.firm_effects

    # index-based overload
    rows, cols, _ = findnz(A)
    sim3 = simulate(model, rows, cols; params..., rng=MersenneTwister(42))
    @test sim3.y == sim.y

    # error on out-of-range indices
    @test_throws ArgumentError simulate(model, [4], [1]; params...)
    @test_throws ArgumentError simulate(model, [1], [4]; params...)
    @test_throws ArgumentError simulate(model, [1, 2], [1]; params...)
    @test_throws ArgumentError simulate(model, [1], [1]; ρ=0.5, σ_a=1.0, σ_z=0.8, σ_ε=-0.1)
end
