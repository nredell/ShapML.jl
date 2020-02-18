module ShapML

using DataFrames
using Random

include("zzz.jl")  # Load predict_shap().

export shap

"""
    shap(explain::DataFrame, reference = nothing, model,
         predict_function, target_features = nothing, sample_size::Integer = 60)

Compute stochastic feature-level Shapley values for any ML model.

# Arguments
- `explain::DataFrame`: A DataFrame of model features with 1 or more instances to be explained using Shapley values.
- `reference`: Optional. A DataFrame with the same format as `explain` which serves as a reference group against which the Shapley value deviations from `explain` are compared (i.e., the model intercept).
- `model`: A trained ML model that is passed into `predict_function`.
- `predict_function`: A wrapper function that takes 2 required positional arguments–(1) the trained model from `model` and (2) a DataFrame of instances with the same format as `explain`. The function should return a 1-column DataFrame of model predictions; the column name does not matter.
- `target_features`: Optional. An `Array{String, 1}` of model features that is a subset of feature names in `explain` for which Shapley values will be computed. For high-dimensional models, selecting a subset of features may dramatically speed up computation time. The default behavior is to return Shapley values for all instances and features in `explain`.
- `sample_size::Integer`: The number of Monte Carlo samples used to compute the stochastic Shapley values for each feature.

# Return
- A `size(explain, 1)` * `length(target_features)` row by 6 column DataFrame.
    + `index`: An instance in `explain`.
    + `feature_name`: Model feature.
    + `feature_value`: Feature value.
    + `shap_effect`: The average Shapley value across Monte Carlo samples.
    + `shap_effect_sd`: The standard deviation of Shapley values across Monte Carlo samples.
    + `intercept`: The average model prediction from `explain` or `reference`.
"""
function shap(;explain::DataFrame, reference = nothing, model,
    predict_function, target_features = nothing, sample_size::Integer = 60)

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
    data_sample = Array{Any}(undef, sample_size)
    for i in 1:sample_size  # Loop over Monte Carlo samples.

        # Shuffle the column indices, keeping all column indices.
        feature_indices_random = Random.randperm(n_features)

        feature_names_random = feature_names[feature_indices_random]

        # Select a reference instance that all instances in explain will be compared to in
        # this Monte Carlo iteration.
        reference_index = rand(1:n_instances)

        # Shuffle the column order for the randomly selected instance.
        reference_instance = reference[[reference_index], feature_indices_random]

        # For the instance(s) to be explained, shuffle the columns to match the randomly selected and shuffled instance.
        explain_instances = explain[1:size(explain, 1), feature_indices_random]

        data_sample_feature = Array{Any}(undef, length(target_features))
        for j in 1:length(target_features)  # Loop over model features in target_features.

            target_feature_index = (1:length(feature_names))[target_features[j] .== feature_names][1]

            target_feature_index_shuffled = (1:length(feature_names))[target_features[j] .== feature_names_random][1]

            # Create the Frankenstein instances: a combination of the instance to be explained with the
            # reference instance to create a new instance that [likely] does not exist in the dataset.

            # These instances have the real target feature and all features to the right of the shuffled
            # target feature index are from the random reference instance.

            # Then, the marginal feature effect, or stochastic Shapley value approximation,
            # is the difference in predicted values between 1 Frankenstein instance
            # that also replaces the target feature from the reference group and 1 Frankenstein
            # instance where the target feature remains unchanged from its value in explain.

            # Initialize the instances to be explained.
            explain_instance_real_target = copy(explain_instances)

            # Only create a Frankenstein instance if the target is not the last feature and there is actually
            # one or more features to the right of the target to replace with the reference.
            if target_feature_index_shuffled < n_features
              explain_instance_real_target = explain_instance_real_target[:, 1:target_feature_index_shuffled]
              explain_instance_real_target_fake_features = repeat(reference_instance[:, (target_feature_index_shuffled + 1):(n_features)], size(explain, 1))
              explain_instance_real_target = hcat(explain_instance_real_target, explain_instance_real_target_fake_features)
            end

            # These instances are otherwise the same as the Frankenstein instance created above with the
            # exception that the target feature is now replaced with the target feature in the random reference
            # instance. The difference in model predictions between these two Frankenstein instances is
            # what gives us the stochastic Shapley value approximation.
            explain_instance_fake_target = copy(explain_instance_real_target)
            explain_instance_fake_target[:, target_feature_index_shuffled] .= reference_instance[!, target_feature_index_shuffled]
            #------------------------------------------------------------------
            # Re-order columns for the user-defined predict() function.
            explain_instance_real_target = explain_instance_real_target[:, feature_names_symbol]
            explain_instance_fake_target = explain_instance_fake_target[:, feature_names_symbol]

            data_sample_feature[j] = vcat(explain_instance_real_target, explain_instance_fake_target)

            # Two Frankenstein instances per explained instance.
            data_sample_feature[j].index = repeat(1:size(explain, 1), outer = 2)
            data_sample_feature[j].feature_group = repeat(["real_target", "fake_target"], inner = size(explain, 1))
            data_sample_feature[j].feature_name = repeat([target_features[j]], size(data_sample_feature[j], 1))
            #data_explain_instance.causal = repeat([0], size(data_explain_instance, 1))
            #data_explain_instance.causal_type = repeat([missing], size(data_explain_instance, 1))
            data_sample_feature[j].sample = repeat([i], size(data_sample_feature[j], 1))

        end  # End 'j' loop for data_sample_feature.

        data_sample[i] = vcat(data_sample_feature...)

    end  # End 'i' loop for data_sample.
    #--------------------------------------------------------------------------
    # Put all Frankenstein instances from all instances passed in 'explain' into
    # a single data.frame for the user-defined predict() function.
    data_predict = vcat(data_sample...)

    data_shap = predict_shap(reference = reference,  # input arg.
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
