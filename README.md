[![Build Status](https://github.com/nredell/ShapML.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nredell/ShapML.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Codecov](https://codecov.io/gh/nredell/ShapML.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nredell/ShapML.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://nredell.github.io/ShapML.jl/dev)

# ShapML <img src="./tools/ShapML_logo.png" alt="ShapML logo" align="right" height="138.5" style="display: inline-block;">

The purpose of `ShapML` is to compute stochastic feature-level Shapley values which
can be used to (a) interpret and/or (b) assess the fairness of any machine learning model.
**[Shapley values](https://christophm.github.io/interpretable-ml-book/shapley.html)**
are an intuitive and theoretically sound model-agnostic diagnostic tool to understand both **global feature importance** across all instances in a data set and instance/row-level **local feature importance** in black-box machine learning models.

This package implements the algorithm described in
[Štrumbelj and Kononenko's (2014) sampling-based Shapley approximation algorithm](https://link.springer.com/article/10.1007%2Fs10115-013-0679-x)
to compute the stochastic Shapley values for a given instance and model feature.

* **Flexibility**:
    + Shapley values can be estimated for any machine learning model using a simple user-defined `predict()` wrapper function.

* **Speed**:
    + The speed advantage of `ShapML` comes in the form of giving the user the ability to select 1 or more target features of interest and avoid having to compute Shapley values for all model features (i.e., a subset of target features from a trained model will return the same feature-level Shapley values as the full model with all features). This is especially useful in high-dimensional models as the computation of a Shapley value is exponential in the number of features.


## Install

* **[pkg.julialang.org](https://pkg.julialang.org)**

``` julia
using Pkg
Pkg.add("ShapML")
```

* Development

``` julia
using Pkg
Pkg.add(PackageSpec(url = "https://github.com/nredell/ShapML.jl"))
```

## Documentation and Vignettes

* **[Docs](https://nredell.github.io/ShapML.jl/dev/)** (incuding examples from [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/))

* **[Consistency with TreeSHAP](https://nredell.github.io/ShapML.jl/dev/vignettes/consistency/)**

* **[Speed - Julia vs Python vs R](https://nredell.github.io/docs/julia_speed)**
