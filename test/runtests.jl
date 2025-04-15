using Aqua
using CSV
using DataFrames
using Lux
using Test
using StableRNGs
using ComponentArrays
using DataInterpolations
using OrdinaryDiffEq
using Statistics
using HydroModels
using HydroModelTools
using SciMLSensitivity
using OptimizationBBO
using OptimizationOptimisers
using Pipe

@testset "HydroModelTools.jl Solvers" begin
    include("solvers/run_solvers.jl")
end

@testset "HydroModelTools.jl Optimizers" begin
    include("optimizers/run_exphydro_optimize.jl")
    # include("optimizers/run_m50_optimize.jl")
end