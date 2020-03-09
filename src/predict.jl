# Internal prediction function. Used at the end of shap().
using Statistics

function _predict(;reference::DataFrame, data_predict::DataFrame, model, predict_function, n_features::Integer,
                  n_target_features::Integer, n_instances_explain::Integer, sample_size::Integer, precision::Union{Integer, Nothing})

  data_model = data_predict[:, 1:n_features]
  data_meta = data_predict[:, (n_features + 1):size(data_predict, 2)]

  # User-defined predict() function
  data_predicted = predict_function(model, data_model)

  # Returns a length 1 numeric vector of the average prediction--i.e., intercept--from the reference group.
  intercept = Statistics.mean(data_predicted[:, 1])

  data_predicted = hcat(data_meta, data_predicted, copycols = false)
  #----------------------------------------------------------------------------
  # Cast the data.frame to, for each random sample, take the difference between the Frankenstein
  # instances.
  user_fun_y_pred_name = names(data_predicted)[end]

  data_predicted = DataFrames.unstack(data_predicted, [:index, :sample, :feature_name], :feature_group, user_fun_y_pred_name)

  # Shapley value for each Monte Carlo sample for each instance.
  data_predicted.shap_effect = data_predicted.real_target - data_predicted.fake_target
  #----------------------------------------------------------------------------
  DataFrames.select!(data_predicted, [:index, :sample, :feature_name, :shap_effect])
  #----------------------------------------------------------------------------
  # Final Shapley value calculation collapsed across Monte Carlo samples.

  # data_predicted = DataFrames.by(data_predicted, [:index, :feature_name],
  #                                shap_effect_sd = :shap_effect => x -> Statistics.std(x),
  #                                shap_effect = :shap_effect => x -> Statistics.mean(x),
  #                                )
  
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
  #----------------------------------------------------------------------------
  data_predicted.intercept = repeat([intercept], size(data_predicted, 1))

  # Optional rounding of Shapley results to reduce dataset size.
  if precision !== nothing
    data_predicted.intercept .= round.(data_predicted.intercept, digits = precision)
    data_predicted.shap_effect .= round.(data_predicted.shap_effect, digits = precision)
    data_predicted.shap_effect_sd .= round.(data_predicted.shap_effect_sd, digits = precision)
  end

  return data_predicted
end  # End _predict()
