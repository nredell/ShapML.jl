module ShapML

using DataFrames
using Random

include("zzz.jl")  # Load predict_shap().

export shap

function shap(;explain::DataFrame, reference = nothing, model,
    predict_function, target_features = nothing, sample_size = 60)

    feature_names = String.(names(explain))

    if (target_features === nothing)
        target_features = feature_names  # Default is to explain with all features.
    end
    #----------------------------------------------------------------------------
    n_features = size(explain, 2)
    #----------------------------------------------------------------------------
    if (reference === nothing)  # Default is to explain all instances in 'explain' without a specific reference group.
        reference = copy(explain)
        n_instances = size(reference, 1)
    else
        n_instances = size(reference, 1)
    end
    #----------------------------------------------------------------------------
    data_sample = Array{Any}(undef, sample_size)
    for i in 1:sample_size  # Loop over Monte Carlo samples.

        # Select a reference instance.
        reference_index = rand(1:n_instances)

        # Shuffle the column indices, keeping all column indices.
        feature_indices_random = Random.randperm(n_features)

        feature_names_random = feature_names[feature_indices_random]

        # Shuffle the column order for the randomly selected instance.
        reference_instance = reference[reference_index, feature_indices_random]

        # For the instance(s) to be explained, shuffle the columns to match the randomly selected and shuffled instance.
        explain_instances = explain[:, feature_indices_random]

        data_sample_feature = Array{Any}(undef, length(target_features))
        for j in 1:length(target_features)

            target_feature_index = convert(Int, findall(occursin.(target_features[j], feature_names))[1])
            target_feature_index_shuffled = convert(Int, findall(occursin.(target_features[j], feature_names_random))[1])

            # Create the Frankenstein instances: a combination of the instance to be explained with the
            # reference instance to create a new instance that [likely] does not exist in the dataset.

            # These instances have the real target feature and all features to the right of the shuffled
            # target feature index are from the random reference instance.

            # Initialize the instances to be explained.
            explain_instance_real_target = copy(explain_instances)

            # Only create a Frankenstein instance if the target is not the last feature and there is actually
            # one or more features to the right of the target to replace with the reference.
            if target_feature_index_shuffled < n_features
              explain_instance_real_target = explain_instance_real_target[:, 1:target_feature_index_shuffled]
              explain_instance_real_target_fake_features = repeat(DataFrames.DataFrame(reference_instance[(target_feature_index_shuffled + 1):(n_features)]), n_instances)
              explain_instance_real_target = hcat(explain_instance_real_target, explain_instance_real_target_fake_features)
            end

            # These instances are otherwise the same as the Frankenstein instance created above with the
            # exception that the target feature is now replaced with the target feature in the random reference
            # instance. The difference in model predictions between these two Frankenstein instances is
            # what gives us the stochastic Shapley value approximation.
            explain_instance_fake_target = copy(explain_instance_real_target)
            explain_instance_fake_target[:, target_feature_index_shuffled] .= reference_instance[target_feature_index_shuffled]
            #------------------------------------------------------------------
            explain_instance_real_target = explain_instance_real_target[:, Symbol.(feature_names)]
            explain_instance_fake_target = explain_instance_fake_target[:, Symbol.(feature_names)]

            data_explain_instance = vcat(explain_instance_real_target, explain_instance_fake_target)

            # Two Frankenstein instances per explained instance.
            data_explain_instance.index = repeat(repeat(1:size(explain, 1)), 2)
            data_explain_instance.feature_group = collect(Iterators.flatten([repeat(["real_target"], size(explain, 1)), repeat(["fake_target"], size(explain, 1))]))
            data_explain_instance.feature_name = repeat([target_features[j]], size(data_explain_instance, 1))
            data_explain_instance.causal = repeat([0], size(data_explain_instance, 1))
            data_explain_instance.causal_type = repeat([missing], size(data_explain_instance, 1))
            data_explain_instance.sample = repeat([i], size(data_explain_instance, 1))
            data_sample_feature[j] = data_explain_instance

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
    # Melt the input 'explain' data.frame for merging the model features to the Shapley values. Suppress
    # the warning resulting from any factors and numeric features being combined into one 'feature_value'
    # column and coerced to characters.
    data_merge = DataFrames.stack(explain, Symbol.(feature_names))
    rename!(data_merge, Dict(:variable => "feature_name", :value => "feature_value"))
    data_merge.feature_name = String.(data_merge.feature_name)

    data_merge.index = repeat(1:size(explain, 1), n_features)  # The merge index for each instance.

    # Each instance in explain has one Shapley value per instance in a long data.frame format.
    data_out = join(data_shap, data_merge, on = [:index, :feature_name], kind = :left)

    # Re-order columns for easier reading.
    data_out = data_out[:, [:index, :feature_name, :feature_value, :shap_effect, :shap_effect_sd, :intercept]]

    return data_out

end  # End shap().
end  # End module.
