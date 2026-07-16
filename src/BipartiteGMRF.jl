module BipartiteGMRF

using ArnoldiMethod: partialeigen, partialschur
using DataFrames: DataFrame, combine, groupby, nrow
using FiniteDiff: finite_difference_gradient!
using LinearAlgebra: Symmetric, SymTridiagonal, cholesky, diag, dot, eigen, logdet, mul!, norm
import Optim:
    LBFGS,
    NelderMead,
    Options,
    converged as optim_converged,
    iterations as optim_iterations,
    minimum as optim_minimum,
    minimizer,
    only_fg!,
    optimize
using Printf: @sprintf
using Random: MersenneTwister, rand, randn
using SparseArrays: SparseMatrixCSC, findnz, nnz, sparse, spdiagm, spzeros
import StatsAPI: coef, loglikelihood, nobs
using Statistics: mean, std

export AbstractGMRFPrior,
    AbstractGMRFSolver,
    NormalizedPrior,
    UnnormalizedPrior,
    SpectralPrior,
    VarianceStablePrior,
    Weighting,
    GMRFProblem,
    HutchSLQ,
    ExactCholesky,
    GMRFResult,
    VarianceDecomposition,
    CovarianceOperator,
    CovarianceBlock,
    NBSpectrum,
    nb_spectrum,
    feasibility,
    rho_at_bound,
    gmrf_mle,
    solve,
    coef,
    loglikelihood,
    nobs,
    nll,
    converged,
    prior_decomposition,
    posterior_decomposition,
    prior_covariance,
    posterior_covariance,
    cov_block

include("types.jl")
include("util.jl")
include("nonbacktracking/graph.jl")
include("nonbacktracking/spectrum.jl")
include("nonbacktracking/feasibility.jl")
include("prepare.jl")
include("operators/qop.jl")
include("operators/qop_vs.jl")
include("operators/mop.jl")
include("linalg/pcg.jl")
include("linalg/slq.jl")
include("solvers/common.jl")
include("solvers/exact.jl")
include("solvers/hutch.jl")
include("decomposition/prior.jl")
include("decomposition/posterior.jl")
include("covariance/operator.jl")
include("covariance/extract.jl")
include("api.jl")

end
