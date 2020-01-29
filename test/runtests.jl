@everywhere begin
  using ShapML
  using Test
  using DataFrames
  using Random
  using Statistics
end

@testset "shap" begin
  @test include("shap.jl")
end
