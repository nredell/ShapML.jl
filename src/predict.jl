# Internal prediction function. Used at the end of shap().
using Statistics

function _predict(;reference::DataFrame, data_predict::DataFrame, model, predict_function, n_features::Integer,
                  n_target_features::Integer, n_instances_explain::Integer, sample_size::Integer, precision::Union{Integer, Nothing})

  # For robustness across prediction functions from other languages (R and Python),
  # the modeling dataset needs to be created before it's passed in predict_function().
  # This is data-to-be-predicted but the name is kept the same throughout to reduce memory.
  data_predicted = data_predict[:, 1:n_features]

  # User-defined predict() function.
  data_predicted = predict_function(model, data_predicted)

  # Returns a length 1 numeric vector of the average prediction--i.e., intercept--from the reference group.
  intercept = Statistics.mean(data_predicted[:, 1])

  # A dataset with meta-data and the predictions in the last column.
  data_predicted = hcat(data_predict[:, (n_features + 1):size(data_predict, 2)], data_predicted, copycols = false)
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
  # Final Shapley value calculation collapsed across Monte Carlo samples using a
  # custom _aggregate() function for speed and reduced memory. The dimensions of
  # the output DataFrame, 'data_predicted', are ('n instances' * 'n features') by 5.
  data_predicted = _aggregate(data_predicted, sample_size, n_instances_explain, n_target_features)
  #----------------------------------------------------------------------------
  data_predicted.intercept = repeat([intercept], size(data_predicted, 1))

  # Optional rounding of Shapley results to reduce dataset size.
  if precision !== nothing
    data_predicted.intercept .= round.(data_predicted.intercept, digits = precision)
    data_predicted.shap_effect .= round.(data_predicted.shap_effect, digits = precision)
    data_predicted.shap_effect_sd .= round.(data_predicted.shap_effect_sd, digits = precision)
  end

  return data_predicted
end  # End _predict().
