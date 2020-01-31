# ShapML.jl

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

```
using Pkg
Pkg.add("ShapML")
```
