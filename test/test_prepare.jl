@testset "prepare" begin
    ss = suffstats(BipartiteNormalizedModel, synthetic_df())
    @test ss.N_firms == 3
    @test ss.N_workers == 4
    @test ss.K == 8
    @test ss.firm_ids isa Vector{Int}
    @test ss.worker_ids isa Vector{Int}
    @test ss.firm_to_index isa Dict{Int,Int}
    @test ss.worker_to_index isa Dict{Int,Int}
    @test ss.firm_to_index[1] == 1
    @test ss.worker_to_index[10] == 1
    @test ss.metadata.outcome == :y

    # GMRFProblem backward compatibility
    p = GMRFProblem(synthetic_df(); model_type=BipartiteNormalizedModel, weighting=Weighting())
    @test p.N_firms == 3
    @test p.N_workers == 4
    @test p.K == 8
    fields = fieldnames(GMRFProblem)
    rebuilt = GMRFProblem(; NamedTuple{fields}(map(field -> getfield(p, field), fields))...)
    @test rebuilt.K == p.K
    @test rebuilt.metadata == p.metadata

    @test BipartiteGMRF.rho_limit(BipartiteNormalizedModel(sparse(ones(2,2)); rho_limit=0.4)) == 0.4
    @test BipartiteGMRF.rho_limit(BipartiteUnnormalizedModel(sparse(ones(2,2)); rho_limit=0.4)) == 0.4
    @test BipartiteGMRF.rho_limit(BipartiteSpectralModel(sparse(ones(2,2)); rho_limit=0.4)) == 0.4
    @test BipartiteGMRF.rho_limit(BipartiteVarianceStableModel(sparse([1.0], [1.0], [1.0], 1, 1); rho_limit=0.4)) == 0.4
    @test_throws ArgumentError BipartiteNormalizedModel(sparse(ones(2,2)); rho_limit=1.0)

    custom_ss = suffstats(BipartiteNormalizedModel,
        custom_column_df();
        outcome=:outcome,
        firm_id=:employer,
        worker_id=:person,
        on_missing=:drop,
    )
    @test custom_ss.K == 5
    @test custom_ss.firm_ids isa Vector{String}
    @test custom_ss.worker_ids isa Vector{String}
    @test custom_ss.firm_to_index isa Dict{String,Int}
    @test custom_ss.worker_to_index isa Dict{String,Int}
    @test custom_ss.firm_to_index["a"] == 1
    @test custom_ss.worker_to_index["p1"] == 1
    @test_throws ArgumentError suffstats(BipartiteNormalizedModel,
        custom_column_df();
        outcome=:outcome,
        firm_id=:employer,
        worker_id=:person,
        on_missing=:error,
    )

    edge_ss = suffstats(BipartiteNormalizedModel, repeated_df(); weighting=Weighting(observations=:edge))
    @test edge_ss.K == 8
    @test edge_ss.personyear_rows == 10

    effective_ss = suffstats(BipartiteNormalizedModel,
        repeated_df();
        weighting=Weighting(observations=:effective, rho_eps=0.5),
    )
    @test effective_ss.rho_eps_likelihood == 0.5
    @test effective_ss.effective_weight_sum < effective_ss.personyear_rows

    maxdeg_ss = suffstats(BipartiteNormalizedModel, synthetic_df(); max_degree=2)
    @test maxdeg_ss.N_firms <= 3
    @test maxdeg_ss.N_workers <= 4

    @test_warn "variance-stable model no longer guarantees" BipartiteVarianceStableModel(
        suffstats(BipartiteVarianceStableModel, synthetic_df()).A_prior,
    )
    @test_throws ArgumentError BipartiteVarianceStableModel(
        suffstats(BipartiteVarianceStableModel, synthetic_df()).A_prior;
        strict_forest=true,
    )
end
