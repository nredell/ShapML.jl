using ShapML
using Test
using DataFrames
using Random
using Statistics
using Distributed

@testset "shap" begin
  include("shap.jl")
end
