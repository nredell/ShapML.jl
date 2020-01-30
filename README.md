[![Build Status](https://travis-ci.org/nredell/ShapML.jl.svg?branch=master)](https://travis-ci.org/nredell/ShapML.jl)
[![Codecov](https://codecov.io/gh/nredell/ShapML.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nredell/ShapML.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://nredell.github.io/ShapML.jl/dev)

# ShapML <img src="./tools/ShapML_logo.png" alt="ShapML logo" align="right" height="138.5" style="display: inline-block;">

The purpose of `ShapML` is to compute stochastic feature-level Shapley values which
can be used to (a) interpret and/or (b) assess the fairness of any machine learning model.
**[Shapley values](https://christophm.github.io/interpretable-ml-book/shapley.html)**
are an intuitive and theoretically sound model-agnostic diagnostic tool to understand both **global feature importance** across all instances in a data set and instance/row-level **local feature importance** in black-box machine learning models.

This package implements the algorithm described in
[Štrumbelj and Kononenko's (2014) sampling-based Shapley approximation algorithm](https://link.springer.com/article/10.1007%2Fs10115-013-0679-x)
to compute the stochastic Shapley values for a given model feature.

* **Flexibility**:
    + Shapley values can be estimated for <u>any machine learning model</u> using a simple user-defined
    `predict()` wrapper function.

* **Speed**:
    + The code itself hasn't necessarily been optimized for speed. The speed advantage of `ShapML`
    comes in the form of giving the user the ability to <u>select 1 or more target features of interest</u>
    and avoid having to compute Shapley values for all model features. This is especially
    useful in high-dimensional models as the computation of a Shapley value is exponential in the number of features.

## Install

``` julia
using Pkg
Pkg.add(PackageSpec(url = "https://github.com/nredell/ShapML.jl"))
```

## Example

``` julia
using ShapML
using Random
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

# Instantiate an ML model; choose any single-outcome ML model from any package.
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
explain = copy(boston[1:300, :]) # Compute Shapley feature-level predictions for 300 instances.
explain = select(explain, Not(Symbol(outcome_name)))  # Remove the outcome column.

reference = copy(boston)  # An optional reference population to compute the baseline prediction.
reference = select(reference, Not(Symbol(outcome_name)))

sample_size = 60  # Number of Monte Carlo samples.
#------------------------------------------------------------------------------
# Compute stochastic Shapley values.
Random.seed!(224)
data_shap = ShapML.shap(explain = explain,
                        reference = reference,
                        model = model,
                        predict_function = predict_function,
                        sample_size = sample_size
                        )

first(data_shap, 20)
```
![](./tools/shap_output.PNG)
