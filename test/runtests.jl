using Test
using LinearAlgebra
using SparseArrays
using StatsAPI
using BipartiteGMRF

include("fixtures/synthetic.jl")

include("test_prepare.jl")
include("test_nb_spectrum.jl")
include("test_feasibility.jl")
include("test_latent_model.jl")
include("test_operators.jl")
include("test_linalg.jl")
include("test_likelihood_dense.jl")
include("test_error_cov.jl")
include("test_error_groups.jl")
include("test_solvers.jl")
include("test_solver_agreement.jl")
include("test_recover_synthetic.jl")
include("test_decomposition.jl")
include("test_covariance.jl")
include("integration.jl")
include("test_fit_mle.jl")
include("test_statsapi.jl")
# test_e2e_leaveout.jl is excluded from the default suite: it depends on the
# unregistered LeaveOut package (KSS leave-one-out reference implementation)
# and Graphs. Run it manually in an environment that provides both.
# test_jet.jl is excluded from the default suite to keep JET (and its heavy
# compile times) out of the test dependencies. Run it manually with:
#   julia --project -e 'using Pkg; Pkg.add("JET")' then include the file
# after fixtures/synthetic.jl.
include("test_mean_structure.jl")
include("test_aqua.jl")
