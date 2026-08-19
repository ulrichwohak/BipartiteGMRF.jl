module BipartiteGMRF

using ArnoldiMethod: partialeigen, partialschur
import CommonSolve: solve
import Distributions: SufficientStats, fit_mle, params, suffstats
using FiniteDiff: finite_difference_gradient!
import GaussianMarkovRandomFields
import GaussianMarkovRandomFields: LatentModel, precision_matrix, model_name, hyperparameters, constraints
using LinearAlgebra: Symmetric, SymTridiagonal, I, cholesky, diag, dot, eigen, issuccess, issymmetric, logdet, mul!, norm, tr
using LinearSolve: CHOLMODFactorization
import Optim:
    AffineSimplexer,
    LBFGS,
    NelderMead,
    Options,
    converged,
    iterations as optim_iterations,
    minimum as optim_minimum,
    minimizer,
    only_fg!,
    optimize
using Printf: @sprintf
import Random
using Random: AbstractRNG, MersenneTwister, rand, randn
using SparseArrays: SparseMatrixCSC, findnz, nnz, nonzeros, nzrange, rowvals, sparse, spdiagm, spzeros
import StatsAPI: StatisticalModel, aic, bic, coef, coefnames, dof, isfitted, islinear, loglikelihood, nobs
using SpecialFunctions: digamma, loggamma
using Statistics: mean, std

# Model types (LatentModel subtypes)
export AbstractBipartiteModel,
    BipartiteNormalizedModel,
    BipartiteUnnormalizedModel,
    BipartiteSpectralModel,
    BipartiteVarianceStableModel,
    BipartiteGraph,
    Weighting

# Solver types
export AbstractGMRFSolver,
    HutchSLQ,
    ExactCholesky,
    EMIWBlocks

# Sufficient statistics (extends the Distributions.jl fit_mle/suffstats API)
export SufficientStats,
    BipartiteGMRFStats,
    suffstats,
    fit_mle

# Data and result types
export GMRFResult,
    VarianceDecomposition,
    CovarianceOperator,
    CovarianceBlock

# Non-backtracking spectrum
export NBSpectrum,
    nb_spectrum,
    feasibility,
    rho_at_bound

# Public API
export solve,
    decompose,
    covariance,
    cov_block,
    simulate,
    coef,
    coefnames,
    params,
    dof,
    loglikelihood,
    nobs,
    aic,
    bic,
    isfitted,
    islinear,
    nll,
    converged

include("util.jl")
# graph.jl/spectrum.jl come before types.jl: BipartiteVarianceStableModel
# stores an NBSpectrum field.
include("nonbacktracking/graph.jl")
include("nonbacktracking/spectrum.jl")
include("types.jl")
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
include("solvers/emblocks.jl")
include("solvers/emiwblocks.jl")
include("decomposition/model.jl")
include("decomposition/fitted.jl")
include("covariance/operator.jl")
include("covariance/extract.jl")
include("stats.jl")
include("simulate.jl")
include("api.jl")
include("fit.jl")

end
