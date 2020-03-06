# The main Shapley sampling algorithm.
function _shap_sample(explain::DataFrame,
                      reference::DataFrame,
                      n_instances::Int64,
                      n_instances_explain::Int64,
                      n_features::Int64,
                      target_features::Array{String,1},
                      feature_names::Array{String,1},
                      feature_names_symbol::Array{Symbol,1},
                      sample_size::Int64,
                      parallel::Symbol,
                      seeds::Union{Array, Integer}
                      )

    if any(parallel .== [:none, :features])
        data_sample = Array{DataFrame}(undef, sample_size)
    elseif any(parallel .== [:samples, :both])
        data_sample = Array{DataFrame}(undef, 1)  # Parallel with pmap().
    end

    for i in if any(parallel .== [:none, :features])
        1:sample_size  # Non-parallel loop over Monte Carlo samples.
    elseif any(parallel .== [:samples, :both])
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
        explain_instances = explain[!, feature_indices_random]

        #----------------------------------------------------------------------
        # Inner loop sampling over target features, creating Frankenstein instances.
        data_sample_feature = Array{Any}(undef, length(target_features))

        if any(parallel .== [:none, :samples])  # Single threaded based on user input.

            for j in 1:length(target_features)  # Loop over model features in target_features.

                data_sample_feature[j] = _shap_sample_features(explain_instances,
                                                               reference_instance,
                                                               n_instances_explain,
                                                               n_features,
                                                               target_features[j],
                                                               feature_names,
                                                               feature_names_symbol,
                                                               feature_names_random,
                                                               i
                                                               )
            end  # End single-threaded loop over target features.

        elseif any(parallel .== [:features, :both])  # Multi-threaded based on user input

            Base.Threads.@threads for j in 1:length(target_features)  # Loop over model features in target_features.

                data_sample_feature[j] = _shap_sample_features(explain_instances,
                                                               reference_instance,
                                                               n_instances_explain,
                                                               n_features,
                                                               target_features[j],
                                                               feature_names,
                                                               feature_names_symbol,
                                                               feature_names_random,
                                                               i
                                                               )
            end  # End multi-threaded loop over target features.
        end  # End inner loop with feature shuffing for each Monte Carlo sample.
        #----------------------------------------------------------------------

        data_sample[i] = vcat(data_sample_feature...)

    end  # End 'i' loop for data_sample.

    return data_sample
end  # End _shap_sample().
#------------------------------------------------------------------------------

# The inner loop function that shuffles features to create Frankenstein instances.
function _shap_sample_features(explain_instances::DataFrame,
                               reference_instance::DataFrame,
                               n_instances_explain::Int64,
                               n_features::Int64,
                               target_features::String,
                               feature_names::Array{String,1},
                               feature_names_symbol::Array{Symbol,1},
                               feature_names_random::Array{String,1},
                               i::Int64
                               )

        target_feature_index_shuffled = (1:n_features)[target_features .== feature_names_random][1]

        # Create the Frankenstein instances: a combination of the instance to be explained with the
        # reference instance to create a new instance that [likely] does not exist in the dataset.

        # These instances have the real target feature and all features to the right of the shuffled
        # target feature index are from the random reference instance.

        # Then, the marginal feature effect, or stochastic Shapley value approximation,
        # is the difference in predicted values between 1 Frankenstein instance
        # that also replaces the target feature from the reference group and 1 Frankenstein
        # instance where the target feature remains unchanged from its value in explain.

        # Initialize the instances to be explained.
        # THIS WAS REMOVED TO REDUCE MEMORY CONSUMPTION.
        # explain_instance_real_target = copy(explain_instances)

        # Only create a Frankenstein instance if the target is not the last feature and there is actually
        # one or more features to the right of the target to replace with the reference.
        if target_feature_index_shuffled < n_features
          explain_instances = hcat(explain_instances[:, 1:target_feature_index_shuffled],
                                   repeat(reference_instance[:, (target_feature_index_shuffled + 1):(n_features)], n_instances_explain))
        end

        # These instances are otherwise the same as the Frankenstein instance created above with the
        # exception that the target feature is now replaced with the target feature in the random reference
        # instance. The difference in model predictions between these two Frankenstein instances is
        # what gives us the stochastic Shapley value approximation.
        explain_instances_fake_target = copy(explain_instances)
        explain_instances_fake_target[:, target_feature_index_shuffled] .= reference_instance[!, target_feature_index_shuffled]
        #------------------------------------------------------------------
        # Re-order columns for the user-defined predict() function and concatenate them vertically.
        explain_instances = vcat(explain_instances[:, feature_names_symbol], explain_instances_fake_target[:, feature_names_symbol])

        # Two Frankenstein instances per explained instance.
        explain_instances.index = repeat(1:n_instances_explain, outer = 2)
        explain_instances.feature_group = repeat(["real_target", "fake_target"], inner = n_instances_explain)
        explain_instances.feature_name = repeat([target_features], size(explain_instances, 1))
        #data_explain_instance.causal = repeat([0], size(data_explain_instance, 1))
        #data_explain_instance.causal_type = repeat([missing], size(data_explain_instance, 1))
        explain_instances.sample = repeat([i], size(explain_instances, 1))

        return(explain_instances)
end  # End _shap_sample_features
