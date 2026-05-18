using Test
using DataFrames
using LinearAlgebra
using SparseArrays
using BipartiteGMRF

include("fixtures/synthetic.jl")

include("test_prepare.jl")
include("test_operators.jl")
include("test_linalg.jl")
include("test_solvers.jl")
include("test_recover_synthetic.jl")
include("test_decomposition.jl")
include("test_covariance.jl")
include("integration.jl")
