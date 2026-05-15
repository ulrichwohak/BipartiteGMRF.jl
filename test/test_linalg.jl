@testset "linalg" begin
    A = sparse([4.0 1.0 0.0; 1.0 3.0 0.5; 0.0 0.5 2.0])
    b = [1.0, 2.0, 3.0]
    ws = BipartiteGMRF.PCGWorkspace(length(b))
    mulA!(y, x) = mul!(y, A, x)
    x, ok, _, relres = BipartiteGMRF.pcg_solve!(ws, mulA!, b; tol=1e-10, maxiter=20)
    @test ok
    @test relres <= 1e-10
    @test x ≈ Matrix(A) \ b atol=1e-8

    slq_ws = BipartiteGMRF.SLQWorkspace(size(A, 1), 3)
    estimated = BipartiteGMRF.slq_logdet_spd_mul_cached!(mulA!, size(A, 1), slq_ws;
        m=80, k=3, seed=2)
    @test isfinite(estimated)
    @test abs(estimated - logdet(Matrix(A))) < 0.5
end
