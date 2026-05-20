@testset "prepare" begin
    p = GMRFProblem(synthetic_df(); prior=NormalizedPrior(), weighting=Weighting())
    @test p.N_firms == 3
    @test p.N_workers == 4
    @test p.K == 8
    @test p.firm_ids isa Vector{Int}
    @test p.worker_ids isa Vector{Int}
    @test p.firm_to_index isa Dict{Int,Int}
    @test p.worker_to_index isa Dict{Int,Int}
    @test p.firm_to_index[1] == 1
    @test p.worker_to_index[10] == 1
    @test p.metadata.outcome == :y
    fields = fieldnames(GMRFProblem)
    rebuilt = GMRFProblem(; NamedTuple{fields}(map(field -> getfield(p, field), fields))...)
    @test rebuilt.K == p.K
    @test rebuilt.metadata == p.metadata

    legacy_aliases = (:N_F, :N_M, :firms, :people, :A_fm, :At_fm, :cnt_m)
    public_names = propertynames(p)
    @test all(alias -> alias ∉ public_names, legacy_aliases)
    @test_deprecated getproperty(p, :N_F)
    @test_deprecated getproperty(p, :N_M)
    @test_deprecated getproperty(p, :firms)
    @test_deprecated getproperty(p, :people)
    @test_deprecated getproperty(p, :A_fm)
    @test_deprecated getproperty(p, :At_fm)
    @test_deprecated getproperty(p, :cnt_m)

    @test NormalizedPrior(rho_limit=0.4).rho_limit == 0.4
    @test UnnormalizedPrior(rho_limit=0.4).rho_limit == 0.4
    @test SpectralPrior(rho_limit=0.4).rho_limit == 0.4
    @test VarianceStablePrior(rho_limit=0.4).rho_limit == 0.4
    @test_throws ArgumentError NormalizedPrior(rho_limit=1.0)

    custom = GMRFProblem(
        custom_column_df();
        outcome=:outcome,
        firm_id=:employer,
        worker_id=:person,
        on_missing=:drop,
    )
    @test custom.K == 5
    @test custom.firm_ids isa Vector{String}
    @test custom.worker_ids isa Vector{String}
    @test custom.firm_to_index isa Dict{String,Int}
    @test custom.worker_to_index isa Dict{String,Int}
    @test custom.firm_to_index["a"] == 1
    @test custom.worker_to_index["p1"] == 1
    @test_throws ArgumentError GMRFProblem(
        custom_column_df();
        outcome=:outcome,
        firm_id=:employer,
        worker_id=:person,
        on_missing=:error,
    )

    edge = GMRFProblem(repeated_df(); weighting=Weighting(observations=:edge))
    @test edge.K == 8
    @test edge.personyear_rows == 10

    effective = GMRFProblem(
        repeated_df();
        weighting=Weighting(observations=:effective, rho_eps=0.5),
    )
    @test effective.rho_eps_likelihood == 0.5
    @test effective.effective_weight_sum < effective.personyear_rows

    maxdeg = GMRFProblem(synthetic_df(); max_degree=2)
    @test maxdeg.N_firms <= 3
    @test maxdeg.N_workers <= 4

    @test_warn "variance-stable prior no longer guarantees" GMRFProblem(
        synthetic_df();
        prior=VarianceStablePrior(),
    )
    @test_throws ArgumentError GMRFProblem(
        synthetic_df();
        prior=VarianceStablePrior(strict_forest=true),
    )
end
