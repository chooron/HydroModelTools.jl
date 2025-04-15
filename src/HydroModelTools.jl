module HydroModelTools

# base dependencies
using Dates
using DataFrames
using ProgressMeter
using NamedTupleTools
using Statistics
using TOML
using ComponentArrays
using ComponentArrays: indexmap, getval
using Reexport

# solve ODEProblem
using OrdinaryDiffEq
using SciMLSensitivity

# Optimization algorithms
using Optimization

# DataInterpolations
@reexport using DataInterpolations

using HydroModelCore: get_name, get_input_names, get_output_names, get_param_names, get_state_names, get_nn_names, get_var_names,
    get_exprs, get_inputs, get_outputs, get_params, get_nns, get_vars
using HydroModelCore: AbstractComponent

const version = VersionNumber(TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])

abstract type AbstractHydroSolver end
abstract type AbstractHydroOptimizer end
abstract type AbstractHydroChain end

abstract type AbstractHydroTool <: AbstractComponent end
abstract type AbstractDataPreprocessor <: AbstractHydroTool end
abstract type AbstractDataPostprocessor <: AbstractHydroTool end
abstract type AbstractComponentDecorator <: AbstractHydroTool end

include("utils/ca.jl")
export update_ca, merge_ca

include("utils/callback.jl")
export get_callback_func, get_batch_callback_func

include("tools.jl")
export NamedTuplePreprocessor, NamedTuplePostprocessor, SelectComponentOutlet

include("optimizers.jl")
export HydroOptimizer

include("solvers.jl")
export ODESolver, DiscreteSolver

end # module HydroTools
