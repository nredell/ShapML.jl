module ShapML

using Distributed
using DataFrames
using Random

include("shap_sample.jl")  # Load _shap_sample().
include("predict.jl")  # Load _predict().
include("aggregate.jl")  # Load _aggregate().

export shap

"""
    shap(explain::DataFrame,
         reference::Union{DataFrame, Nothing} = nothing,
         model,
         predict_function::Function,
         target_features::Union{Vector, Nothing} = nothing,
         sample_size::Integer = 60,
         parallel::Symbol = [:none, :samples, :features, :both],
         seed::Integer = 1,
         precision::Union{Integer, Nothing} = nothing,
         chunk::Bool = true,
         reconcile_instance::Bool = false
         )

Compute stochastic feature-level Shapley values for any ML model.

# Arguments
- `explain::DataFrame`: A DataFrame of model features with 1 or more instances to be explained using Shapley values.
- `reference`: Optional. A DataFrame with the same format as `explain` which serves as a reference group against which the Shapley value deviations from `explain` are compared (i.e., the model intercept).
- `model`: A trained ML model that is passed into `predict_function`.
- `predict_function`: A wrapper function that takes 2 required positional arguments–(1) the trained model from `model` and (2) a DataFrame of instances with the same format as `explain`. The function should return a 1-column DataFrame of model predictions; the column name does not matter.
- `target_features`: Optional. An `Array{String, 1}` of model features that is a subset of feature names in `explain` for which Shapley values will be computed. For high-dimensional models, selecting a subset of features may dramatically speed up computation time. The default behavior is to return Shapley values for all instances and features in `explain`.
- `sample_size::Integer`: The number of Monte Carlo samples used to compute the stochastic Shapley values for each feature.
- `parallel::Union{Symbol, Nothing}`: One of [:none, :samples, :features, :both]. Whether to perform the calculation serially (:none) or in parallel over Monte Carlo samples (:samples) with `pmap()` and/or multi-threaded over target features (:features) with @threads or :both.
- `seed::Integer`: A number passed to `Random.seed!()` to get reproducible results.
- `precision::Union{Integer, Nothing}`: The number of digits to `round()` results in the ouput (to reduce the size of the returned DataFrame).
- `chunk::Bool`: Default `true`. Increases speed on data with many instances and/or features. Calls the `predict()` function once per sample in `sample_size` instead of once per call to `ShapML.shap()`.
- `reconcile_instance`: EXPERIMENTAL. For each instance in `explain`, the stochastic feature-level Shapley values are adjusted so that their sum equals the model prediction. The adjustments are based on feature-level sampling variances and are typically small compared to the model prediction.

# Return
- A `size(explain, 1)` * `length(target_features)` row by 6 column DataFrame.
    + `index`: An instance in `explain`.
    + `feature_name`: Model feature.
    + `feature_value`: Feature value.
    + `shap_effect`: The average Shapley value across Monte Carlo samples.
    + `shap_effect_sd`: The standard deviation of Shapley values across Monte Carlo samples.
    + `intercept`: The average model prediction from `explain` or `reference`.
"""
function shap(;explain::DataFrame,
              reference::Union{DataFrame, Nothing} = nothing,
              model,
              predict_function::Function,
              target_features::Union{Vector, Nothing} = nothing,
              sample_size::Integer = 60,
              parallel::Union{Symbol, Nothing} = nothing,
              seed::Integer = 1,
              precision::Union{Integer, Nothing} = nothing,
              chunk::Bool = true,
              reconcile_instance::Bool = false
              )

    feature_names = String.(names(explain))
    feature_names_symbol = Symbol.(feature_names)

    if (target_features === nothing)

        target_features = copy(feature_names)  # Default is to explain with all features.

    else

        if !all(isa.(target_features, String))
            error(""""target_features" should be an array of feature names of type "String".""")
        end

        if !all(map(x -> any(x .== target_features), target_features))
            error("""One or more "target_features" is not in String.(names(explain)).""")
        end
    end
    #----------------------------------------------------------------------------
    n_instances_explain = size(explain, 1)
    n_features = size(explain, 2)
    n_target_features = length(target_features)
    #----------------------------------------------------------------------------
    if (reference === nothing)  # Default is to explain all instances in 'explain' without a specific reference group.

        reference = copy(explain)

    else

        if names(explain) != names(reference)
            error(""""explain" and "reference" should have the same model features and no outcome column.""")
        end
    end

    n_instances = size(reference, 1)
    #--------------------------------------------------------------------------
    # Parallel computation setup; the type of parallelization, if any, depends on
    # the 'parallel' argument.
    if (parallel === nothing)
        parallel = :none
    end

    if !any(parallel .== [:none, :samples, :features, :both])
         error(""""parallel" should be one of [:none, :samples, :features, :both].""")
    end
    #--------------------------------------------------------------------------
    # Create a vector of random seeds to get reproducible results with both
    # serial and parallel computations. This is not the perfect solution because there
    # could potentially be correlation between the seeds, but the effect on randomness,
    # if any, will be small. To-do: Pass in seed generator objects.
    Random.seed!(seed)
    seeds = abs.(rand(Int, sample_size))
    #--------------------------------------------------------------------------
    # Main Shapley value computation from _shap_sample(). This code is either
    # run serially or in parallel.
    if any(parallel .== [:none, :features])

        data_predict = _shap_sample(explain,
                                    reference,
                                    n_instances,
                                    n_instances_explain,
                                    n_features,
                                    n_target_features,
                                    target_features,
                                    feature_names,
                                    feature_names_symbol,
                                    sample_size,
                                    parallel,
                                    seeds,
                                    chunk,
                                    model,  # chunk = true.
                                    predict_function,  # chunk = true.
                                    precision  # chunk = true.
                                    )

    elseif any(parallel .== [:samples, :both])

        data_predict = pmap(_i -> _shap_sample(explain,
                                               reference,
                                               n_instances,
                                               n_instances_explain,
                                               n_features,
                                               n_target_features,
                                               target_features,
                                               feature_names,
                                               feature_names_symbol,
                                               sample_size,
                                               parallel,
                                               seeds[_i],
                                               chunk,
                                               model,  # chunk = true.
                                               predict_function,  # chunk = true.
                                               precision  # chunk = true.
                                               ), 1:sample_size)
    end  # End Shapley value Monte Carlo calculation.
    #--------------------------------------------------------------------------
    # Put all Frankenstein instances from all instances passed in 'explain' into
    # a single DataFrame for the user-defined predict() function.
    data_predict = vcat(data_predict...)

    if any(parallel .== [:samples, :both])
        data_predict = vcat(data_predict...)
        data_predict.sample = repeat(1:sample_size, inner = n_instances_explain * n_target_features * 2)
    end

    data_shap = _predict(reference = reference,  # input arg.
                         data_predict = data_predict,  # Calculated.
                         model = model,  # input arg.
                         predict_function = predict_function,  # input arg.
                         n_features = n_features,  # Calculated.
                         n_target_features = n_target_features,  # Calculated.
                         n_instances_explain = n_instances_explain,  # Calculated.
                         sample_size = sample_size,  # input arg.
                         precision = precision,  # input arg.
                         chunk = chunk,  # input arg.
                         reconcile_instance = reconcile_instance,  # input arg.
                         explain = explain  # input arg; needed if reconcile_instance = true.
                         )
    #--------------------------------------------------------------------------
    # Melt the input 'explain' DataFrame for merging the model features to the Shapley values.
    data_merge = DataFrames.stack(explain, feature_names_symbol)
    rename!(data_merge, Dict(:variable => "feature_name", :value => "feature_value"))
    data_merge.feature_name = String.(data_merge.feature_name)  # Coerce for merging.

    data_merge.index = repeat(1:n_instances_explain, n_features)  # The merge index for each instance.

    # Each instance in explain has one Shapley value per instance in a long DataFrame format.
    data_shap = leftjoin(data_shap, data_merge, on = [:index, :feature_name])

    # Re-order columns for easier reading.
    DataFrames.select!(data_shap, [:index, :feature_name, :feature_value, :shap_effect, :shap_effect_sd, :intercept])

    return data_shap

end  # End shap().
end  # End module.
