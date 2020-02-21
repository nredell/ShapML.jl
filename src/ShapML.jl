module ShapML

using DataFrames
using Random
using Distributed

include("predict.jl")  # Load _predict().
include("shap_sample.jl")  # Load _shap_sample().

export shap

"""
    shap(explain::DataFrame,
         reference = nothing,
         model,
         predict_function,
         target_features = nothing,
         sample_size::Integer = 60,
         parallel::Symbol = [:none, :samples])

Compute stochastic feature-level Shapley values for any ML model.

# Arguments
- `explain::DataFrame`: A DataFrame of model features with 1 or more instances to be explained using Shapley values.
- `reference`: Optional. A DataFrame with the same format as `explain` which serves as a reference group against which the Shapley value deviations from `explain` are compared (i.e., the model intercept).
- `model`: A trained ML model that is passed into `predict_function`.
- `predict_function`: A wrapper function that takes 2 required positional arguments–(1) the trained model from `model` and (2) a DataFrame of instances with the same format as `explain`. The function should return a 1-column DataFrame of model predictions; the column name does not matter.
- `target_features`: Optional. An `Array{String, 1}` of model features that is a subset of feature names in `explain` for which Shapley values will be computed. For high-dimensional models, selecting a subset of features may dramatically speed up computation time. The default behavior is to return Shapley values for all instances and features in `explain`.
- `sample_size::Integer`: The number of Monte Carlo samples used to compute the stochastic Shapley values for each feature.
- `parallel::Symbol`: One of [:none, :samples]. Whether to perform the calculation serially (:none) or in parallel (:samples) with `pmap()`.

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
              reference = nothing,
              model,
              predict_function,
              target_features = nothing,
              sample_size::Integer = 60,
              parallel::Symbol = [:none, :samples]
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
    n_features = size(explain, 2)
    #----------------------------------------------------------------------------
    if (reference === nothing)  # Default is to explain all instances in 'explain' without a specific reference group.

        reference = copy(explain)
        n_instances = size(reference, 1)

    else

        if !isa(reference, DataFrame)
            error(""""reference" should be a "DataFrame" object.""")
        end

        if names(explain) != names(reference)
            error(""""explain" and "reference" should have the same model features and no outcome column.""")
        end

        n_instances = size(reference, 1)
    end
    #----------------------------------------------------------------------------
    # Parallel computation setup; the type of parallelization, if any, depends on
    # the 'parallel' argument.
    if isa(parallel, Array)
        parallel = parallel[1]  # Default is a non-parallel computation.
    end

     if !any(parallel .== [:none, :samples])
         error(""""parallel" should be one of "[:none, :samples]".""")
     end
    #--------------------------------------------------------------------------
    # A function that chooses serial or parallel computation depending on user input.
    data_sample = Array{Any}(undef, sample_size)

    if parallel == :none

        _i = nothing
        _shap_sample(explain, reference, n_instances, n_features, target_features,
                     feature_names, feature_names_symbol, sample_size, parallel, _i, data_sample)

    elseif parallel == :samples

        pmap(_i -> _shap_sample(explain, reference, n_instances, n_features, target_features,
                                feature_names, feature_names_symbol, sample_size, parallel, _i, data_sample),
                                1:sample_size)
    end
    #--------------------------------------------------------------------------
    # Put all Frankenstein instances from all instances passed in 'explain' into
    # a single data.frame for the user-defined predict() function.
    data_predict = vcat(data_sample...)

    data_shap = _predict(reference = reference,  # input arg.
                         data_predict = data_predict,  # Calculated.
                         model = model,  # input arg.
                         predict_function = predict_function,  # input arg.
                         n_features = n_features  # Calculated.
                         )
    #--------------------------------------------------------------------------
    # Melt the input 'explain' data.frame for merging the model features to the Shapley values.
    data_merge = DataFrames.stack(explain, feature_names_symbol)
    rename!(data_merge, Dict(:variable => "feature_name", :value => "feature_value"))
    data_merge.feature_name = String.(data_merge.feature_name)  # Coerce for merging.

    data_merge.index = repeat(1:size(explain, 1), n_features)  # The merge index for each instance.

    # Each instance in explain has one Shapley value per instance in a long data.frame format.
    data_out = join(data_shap, data_merge, on = [:index, :feature_name], kind = :left)

    # Re-order columns for easier reading.
    data_out = data_out[:, [:index, :feature_name, :feature_value, :shap_effect, :shap_effect_sd, :intercept]]

    return data_out

end  # End shap().
end  # End module.
