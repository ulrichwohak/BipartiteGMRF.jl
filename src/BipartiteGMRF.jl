module BipartiteGMRF

using DataFrames: DataFrame, combine, groupby, nrow
using FiniteDiff: finite_difference_gradient!, finite_difference_hessian
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
using Random: MersenneTwister, rand, randn
using SparseArrays: SparseMatrixCSC, findnz, nnz, sparse, spdiagm
import StatsAPI: coef, loglikelihood, nobs, vcov, stderror, confint
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
    gmrf_mle,
    solve,
    coef,
    loglikelihood,
    nobs,
    nll,
    converged,
    observed_information,
    vcov,
    stderror,
    confint,
    with_standard_errors,
    prior_decomposition,
    posterior_decomposition,
    prior_covariance,
    posterior_covariance,
    cov_block

include("types.jl")
include("util.jl")
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
include("inference.jl")

end
