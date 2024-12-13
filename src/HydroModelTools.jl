module HydroModelTools

# base dependencies
using Reexport
using Dates
using DataFrames
using ProgressMeter
using IterTools: ncycle
using NamedTupleTools
using Statistics
using TOML
@reexport using ComponentArrays
using ComponentArrays: indexmap, getval

# solve ODEProblem
using OrdinaryDiffEq
using SciMLSensitivity

# Optimization algorithms
using Optimization
using OptimizationBBO
using OptimizationOptimisers

const version = VersionNumber(TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])

abstract type AbstractHydroSolver end
abstract type AbstractHydroOptimizer end

include("utils/ca.jl")
export update_ca, merge_ca
include("utils/callback.jl")
export get_callback_func, get_batch_callback_func

include("optimizer.jl")
export BatchOptimizer, HydroOptimizer, GradOptimizer

include("solver.jl")
export ODESolver, DiscreteSolver, ManualSolver

end # module HydroTools
