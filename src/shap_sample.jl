# The main Shapley sampling algorithm.
function _shap_sample(explain::DataFrame,
                      reference::DataFrame,
                      n_instances::Int64,
                      n_features::Int64,
                      target_features::Array{String,1},
                      feature_names::Array{String,1},
                      feature_names_symbol::Array{Symbol,1},
                      sample_size::Int64,
                      parallel::Symbol,
                      seeds
                      )

    if parallel == :none
        data_sample = Array{Any}(undef, sample_size)
    elseif parallel == :samples
        data_sample = Array{Any}(undef, 1)  # Parallel with pmap().
    end

    for i in if parallel == :none
        1:sample_size  # Non-parallel loop over Monte Carlo samples.
    elseif parallel == :samples
        1  # Parallel with pmap().
    end

        # Shuffle the column indices, keeping all column indices.
        Random.seed!(seeds[i])
        feature_indices_random = Random.randperm(n_features)

        feature_names_random = feature_names[feature_indices_random]

        # Select a reference instance that all instances in explain will be compared to in
        # this Monte Carlo iteration.
        Random.seed!(seeds[i])
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

    return data_sample
end  # End _shap_sample().
