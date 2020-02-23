using Distributed
addprocs(6)  # 6 cores.

@everywhere begin
  using ShapML
  using DataFrames
  using MLJ
end

using RDatasets

# Load data.
boston = RDatasets.dataset("MASS", "Boston")
#------------------------------------------------------------------------------
# Train a machine learning model; currently limited to single outcome regression and binary classification.
outcome_name = "MedV"

# Data prep.
y, X = MLJ.unpack(boston, ==(Symbol(outcome_name)), colname -> true)

X_plus = reshape(randn(506 * 30), 506, 30)
X_plus = DataFrame(X_plus)

X = hcat(X, X_plus)

# Instantiate an ML model; choose any single-outcome ML model from any package.
random_forest = @load RandomForestRegressor pkg = "DecisionTree"
model = MLJ.machine(random_forest, X, y)

# Train the model.
fit!(model)

# function should return a 1-column DataFrame of predictions--column names do not matter.
@everywhere function predict_function(model, data)
  data_pred = DataFrame(y_pred = predict(model, data))
  return data_pred
end

# ShapML setup.
explain = copy(X[1:300, :]) # Compute Shapley feature-level predictions for 300 instances.
explain = select(explain, Not(Symbol(outcome_name)))  # Remove the outcome column.

reference = copy(X)  # An optional reference population to compute the baseline prediction.
reference = select(reference, Not(Symbol(outcome_name)))

sample_size = 60  # Number of Monte Carlo samples.
#------------------------------------------------------------------------------
# Compute stochastic Shapley values.
using BenchmarkTools

data_shap = @btime ShapML.shap(explain = explain,
                        reference = reference,
                        model = model,
                        predict_function = predict_function,
                        sample_size = sample_size,
                        parallel = :none,  # Parallel computation over "sample_size".
                        seed = 1
                        )

show(data_shap, allcols = true)
