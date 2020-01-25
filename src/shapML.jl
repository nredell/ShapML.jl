#------------------------------------------------------------------------------
module shapML

using DataFrames
using Random

export shap

function shap(;explain::DataFrame, reference = nothing, model,
    predict_function, target_features = nothing, sample_size = 60)

    if (target_features === nothing)
        target_features = String.(names(explain))  # Default is to explain with all features.
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

        feature_names_random = String.(names(explain))[feature_indices_random]

        # Shuffle the column order for the randomly selected instance.
        reference_instance = reference[reference_index, feature_indices_random]

        # For the instance(s) to be explained, shuffle the columns to match the randomly selected and shuffled instance.
        explain_instances = explain[:, feature_indices_random]

        data_sample_feature = Array{Any}(undef, length(target_features))
        for j in 1:length(target_features)

        end  # End 'j' loop over target features.
    end  # End 'i' loop over Monte Carlo samples.
end  # End shap().
end  # End module.
#------------------------------------------------------------------------------
