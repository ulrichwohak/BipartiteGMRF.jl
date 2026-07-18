module BipartiteGMRF

using ArnoldiMethod: partialeigen, partialschur
using DataFrames: DataFrame, combine, groupby, nrow
using FiniteDiff: finite_difference_gradient!
import GaussianMarkovRandomFields
import GaussianMarkovRandomFields: LatentModel, precision_matrix, model_name, hyperparameters, constraints
using LinearAlgebra: Symmetric, SymTridiagonal, cholesky, diag, dot, eigen, logdet, mul!, norm
using LinearSolve: CHOLMODFactorization
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
import Random
using Random: AbstractRNG, MersenneTwister, rand, randn
using SparseArrays: SparseMatrixCSC, findnz, nnz, sparse, spdiagm, spzeros
import StatsAPI: coef, loglikelihood, nobs
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
    ExactCholesky

# Sufficient statistics
export SufficientStats,
    BipartiteGMRFStats,
    suffstats,
    fit_mle

# Data and result types
export GMRFProblem,
    GMRFResult,
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
    loglikelihood,
    nobs,
    nll,
    converged

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
include("decomposition/model.jl")
include("decomposition/fitted.jl")
include("covariance/operator.jl")
include("covariance/extract.jl")
include("stats.jl")
include("simulate.jl")
include("api.jl")
include("fit.jl")

end
