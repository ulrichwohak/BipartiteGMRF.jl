@testset "linalg" begin
    A = sparse([4.0 1.0 0.0; 1.0 3.0 0.5; 0.0 0.5 2.0])
    b = [1.0, 2.0, 3.0]
    ws = BipartiteGMRF.PCGWorkspace(length(b))
    mulA!(y, x) = mul!(y, A, x)
    x, ok, _, relres = BipartiteGMRF.pcg_solve!(ws, mulA!, b; tol=1e-10, maxiter=20)
    @test ok
    @test relres <= 1e-10
    @test x ≈ Matrix(A) \ b atol=1e-8

    @testset "unpreconditioned PCG restarts from the true residual" begin
        # This genuinely SPD matrix is ill-conditioned at Float64 precision.
        # The large iterate rounds away an update, so the recursively updated
        # residual becomes zero even though b - A*x is not small. The operator
        # only records checks; it always multiplies by the same matrix.
        ill_conditioned = [1.0 1.0; 1.0 nextfloat(1.0)]
        rhs = [1.0, -1.0]
        rhs_norm = norm(rhs)
        tolerance = 1e-12
        @test isposdef(ill_conditioned)
        for preconditioner in (nothing, ones(2))
            workspace = BipartiteGMRF.PCGWorkspace(2)
            checked_residuals = Tuple{Float64,Float64}[]
            function checked_mul!(out, v)
                mul!(out, ill_conditioned, v)
                if v === workspace.x
                    push!(checked_residuals,
                        (norm(workspace.r) / rhs_norm, norm(rhs - out) / rhs_norm))
                end
                return out
            end
            solution, success, iterations, residual = BipartiteGMRF.pcg_solve!(
                workspace, checked_mul!, rhs; tol=tolerance, maxiter=5,
                Mdiag=preconditioner)
            @test any(pair -> pair[1] <= tolerance && pair[2] > tolerance,
                checked_residuals)
            @test !success
            @test 0 < iterations <= 5
            @test residual == norm(rhs - ill_conditioned * solution) / rhs_norm
            @test residual > tolerance
        end
    end

    slq_ws = BipartiteGMRF.SLQWorkspace(size(A, 1), 3)
    estimated = BipartiteGMRF.slq_logdet_spd_mul_cached!(mulA!, size(A, 1), slq_ws;
        m=80, k=3, seed=2)
    @test isfinite(estimated)
    @test abs(estimated - logdet(Matrix(A))) < 0.5

    for diagonal in ([1.0, -2.0, 3.0], [0.0, 0.0, 0.0])
        invalid_mul!(out, v) = (@. out = diagonal * v)
        @test isnan(BipartiteGMRF.slq_logdet_spd_mul_cached!(invalid_mul!, 3,
            BipartiteGMRF.SLQWorkspace(3, 3); m=2, k=3, seed=2))
    end

    for scale in (1.0, 1e-10, 1e-16)
        diagonal = scale .* [1.0, 2.0, 3.0]
        scaled_mul!(out, v) = (@. out = diagonal * v)
        value = BipartiteGMRF.slq_logdet_spd_mul_cached!(scaled_mul!, 3,
            BipartiteGMRF.SLQWorkspace(3, 3); m=1, k=3, seed=2)
        @test value ≈ sum(log, diagonal) atol=1e-11
    end

    B = sparse([1, 2, 3, 3], [1, 1, 2, 3], [2.0, 1.0, 3.0, 4.0], 3, 3)
    Bt = copy(transpose(B))
    seeded = BipartiteGMRF.leading_singular_value(B, Bt; maxiter=1, tol=0.0, seed=7)
    @test seeded == BipartiteGMRF.leading_singular_value(B, Bt; maxiter=1, tol=0.0, seed=7)
    @test seeded != BipartiteGMRF.leading_singular_value(B, Bt; maxiter=1, tol=0.0, seed=8)
end
