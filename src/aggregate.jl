# _group_by() is a custom function that aggregates the Shapley values in _predict() by exploiting the
# known structure of 'data_predicted'. It is both quicker and more memory efficient than
# the equivalent code below:

# DataFrames.by(data_predicted, [:index, :feature_name],
#               shap_effect_sd = :shap_effect => x -> Statistics.std(x),
#               shap_effect = :shap_effect => x -> Statistics.mean(x))
using Statistics

function _aggregate(data_predicted::DataFrame, sample_size::Integer, n_instances_explain::Integer, n_target_features::Integer)

  feature_name = repeat(data_predicted.feature_name[1:n_target_features], outer = n_instances_explain)
  index = repeat(1:n_instances_explain, inner = n_target_features)

  indices = map(i -> i * (n_target_features * sample_size), vcat([0], collect(1:(n_instances_explain - 1))))
  indices_list = Array{Any}(undef, n_instances_explain)
  for i in 1:n_instances_explain
     indices_list[i] = map(j -> (j + indices[i]):n_target_features:((n_target_features * sample_size + j - 1) + indices[i]), 1:n_target_features)
  end

  shap_effect_instance_feature = [Array{Any}(undef, n_target_features) for i in 1:n_instances_explain]
  shap_effect_instances = Array{Any}(undef, n_instances_explain)
  for i in 1:n_instances_explain
    for j in 1:n_target_features
      shap_effect_instance_feature[i][j] = reshape([Statistics.mean(data_predicted[indices_list[i][j], :shap_effect]), Statistics.std(data_predicted[indices_list[i][j], :shap_effect])], 1, 2)
    end
    shap_effect_instances[i] = vcat(shap_effect_instance_feature[i]...)
  end

  data_predicted = DataFrames.DataFrame(vcat(shap_effect_instances...))
  rename!(data_predicted, [:shap_effect, :shap_effect_sd])
  data_predicted.index = index
  data_predicted.feature_name = feature_name

  return data_predicted
end  # End _group_by().
