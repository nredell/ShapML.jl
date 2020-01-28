# ShapML

[![Codecov](https://codecov.io/gh/nredell/ShapML.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nredell/ShapML.jl)

[![Build Status](https://travis-ci.org/nredell/ShapML.jl.svg?branch=master)](https://travis-ci.org/nredell/ShapML.jl)

## Install

``` julia
using Pkg
Pkg.add(PackageSpec(url = "https://github.com/nredell/ShapML.jl"))
```

## Example

``` julia
using ShapML
using RDatasets
using DataFrames
using MLJ

# Load data.
boston = RDatasets.dataset("MASS", "Boston")
#------------------------------------------------------------------------------
# Train a machine learning model; currently limited to single outcome regression and binary classification.
outcome_name = "MedV"

# Data prep.
y, X = MLJ.unpack(boston, ==(Symbol(outcome_name)), colname -> true)

# Instantiate ML model; choose any single outcome ML model.
random_forest = @load RandomForestRegressor pkg = "DecisionTree"
model = MLJ.machine(random_forest, X, y)

# Train the model.
fit!(model)

# Create a wrapper function that takes the following positional arguments: (1) a
# trained ML model from any Julia package, (2) a DataFrame of model features. The
# function should return a 1-column DataFrame of predictions--column names do not matter.
function predict_function(model, data)
  data_pred = DataFrame(y_pred = predict(model, data))
  return data_pred
end
#------------------------------------------------------------------------------
# ShapML setup.
explain = copy(boston[1:300, :]) # Compute Shapley feature-level predictions for 300 instaces.
explain = select(explain, Not(Symbol(outcome_name)))  # Remove the outcome column.

reference = copy(boston)  # An optional reference population to compute the baseline prediction.
reference = select(reference, Not(Symbol(outcome_name)))
#reference = reference[1:300, :]

sample_size = 60  # Number of Monte Carlo samples.
#------------------------------------------------------------------------------

data_shap = ShapML.shap(explain = explain,
                        #reference = reference,
                        model = model,
                        predict_function = predict_function,
                        sample_size = sample_size
                        )

first(data_shap, 20)

```
