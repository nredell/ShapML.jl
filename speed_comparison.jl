
# using ShapML
include("C:/Users/nickr/Desktop/github/julia/ShapML.jl/src/ShapML.jl")
using Random
using DataFrames
using RCall

n_features = [100]#[10, 20, 100]#, 200)
n_instances = [1000]#[1, 100, 1000]#, 5000)
n_models = length(n_features)
n_simulations = 3
seed = 1

n_monte_carlo = 20

R"""
library(fastshap)
library(ranger)

n_features <- 100# c(10, 20, 100)#, 200)
n_instances <- 1000# c(1, 100, 1000)#, 5000)
n_models <- length(n_features)
n_simulations <- 3
seed <- 1

data_train <- vector("list", n_models)
models <- vector("list", n_models)

for(i in 1:n_models) {

  data_train[[i]] <- fastshap::gen_friedman(n_samples = max(n_instances), n_features[i], seed = seed)

  models[[i]] <- ranger::ranger(y ~ ., data = data_train[[i]], seed = seed)
}

names(data_train) <- n_features
names(models) <- n_features
"""

R"""
predict_fun_julia <- function(model, data) {

  data_pred = data.frame("y_pred" = predict(model, data)$predictions)
  return(data_pred)
}
"""

R"""
conditions <- expand.grid("n_features" = n_features, "n_instances" = n_instances)
conditions$condition <- 1:nrow(conditions)
# We'll remove the 5000 instance, 200 feature condition to keep the runtime reasonable.
# conditions <- conditions[-nrow(conditions), ]

n_conditions <- nrow(conditions)
data_explain <- vector("list", n_conditions)
models_condition <- vector("list", n_conditions)

for(i in 1:n_conditions) {

  data_condition <- data_train[[which(names(data_train) == as.character(conditions$n_features[i]))]]
  data_explain[[i]] <- data_condition[1:conditions$n_instances[i], !names(data_condition) %in% "y"]
  models_condition[[i]] <- models[[which(names(models) == as.character(conditions$n_features[i]))]]
}
"""

data_explain = RCall.reval("data_explain")

models_condition = RCall.reval("models_condition")

n_conditions = RCall.reval("n_conditions")
n_conditions = convert(Integer, n_conditions)

for i in 1:n_conditions
    data_explain[i] = convert(DataFrame, data_explain[i])
end

predict_fun_julia = RCall.reval("predict_fun_julia")
predict_fun_julia = convert(Function, predict_fun_julia)

#------------------------------------------------------------------------------
data_shap_julia = [Array{Any}(undef, n_simulations) for i in 1:n_conditions]
runtime_julia = [Array{Any}(undef, n_simulations) for i in 1:n_conditions]

for i in 1:n_conditions
  for j in 1:n_simulations

      data_explain_temp = convert(DataFrame, data_explain[i])

      start_time = time()

      data_shap_julia[i][j] = Main.ShapML.shap(explain = data_explain_temp,
                                          reference = data_explain_temp,
                                          model = models_condition[i],
                                          predict_function = predict_fun_julia,
                                          sample_size = n_monte_carlo,
                                          seed = seed
                                          )

      stop_time = time()
      runtime_julia[i][j] = stop_time - start_time
  end
end
#------------------------------------------------------------------------------
7

using RDatasets
using DataFrames
using MLJ  # Machine learning

# Load data.
boston = RDatasets.dataset("MASS", "Boston")
boston = boston[repeat(1:size(boston, 1), inner = 3), :]
boston = hcat(boston, boston, boston, makeunique=true)
show(boston, allcols = true)
#------------------------------------------------------------------------------
# Train a machine learning model; currently limited to single outcome regression and binary classification.
outcome_name = "MedV"

# Data prep.
y, X = MLJ.unpack(boston, ==(Symbol(outcome_name)), colname -> true)

# Instantiate an ML model; choose any single-outcome ML model from any package.
random_forest = @load RandomForestRegressor pkg = "DecisionTree"
model = MLJ.machine(random_forest, X, y)

# Train the model.
using Random
Random.seed!(1)
fit!(model)

# Create a wrapper function that takes the following positional arguments: (1) a
# trained ML model from any Julia package, (2) a DataFrame of model features. The
# function should return a 1-column DataFrame of predictions--column names do not matter.
function predict_function(model, data)
  data_pred = DataFrame(y_pred = MLJ.predict(model, data))
  return data_pred
end
#------------------------------------------------------------------------------
# ShapML setup.
explain = copy(boston) # Compute Shapley feature-level predictions for 300 instances.
explain = select(explain, Not(Symbol(outcome_name)))  # Remove the outcome column.

reference = copy(boston)  # An optional reference population to compute the baseline prediction.
reference = select(reference, Not(Symbol(outcome_name)))

sample_size = 60  # Number of Monte Carlo samples.


target_features = nothing
parallel = nothing
seed = 1
precision = nothing

using Distributed
using DataFrames
using Random
