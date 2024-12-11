using Aqua
using CSV
using DataFrames
using Lux
using Test
using ModelingToolkit
using Symbolics
using LuxCore
using StableRNGs
using ComponentArrays
using DataInterpolations
using OrdinaryDiffEq
using Statistics
using BenchmarkTools
using Graphs
using Plots
using HydroModels
using Test


@testset "HydroKit.jl" begin
    include("run_optimize.jl")
    include("run_solver.jl")
end