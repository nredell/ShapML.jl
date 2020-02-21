# Internal prediction function. Used at the end of shap().
using Statistics

function _predict(;reference::DataFrame, data_predict::DataFrame, model, predict_function, n_features::Integer)

  data_model = data_predict[:, 1:n_features]
  data_meta = data_predict[:, (n_features + 1):size(data_predict, 2)]

  # User-defined predict() function
  data_predicted = predict_function(model, data_model)

  data_predicted = hcat(data_meta, data_predicted)

  # Returns a length 1 numeric vector of the average prediction--i.e., intercept--from the reference group.
  intercept = Statistics.mean(predict_function(model, reference)[:, 1])
  #----------------------------------------------------------------------------
  # Cast the data.frame to, for each random sample, take the difference between the Frankenstein
  # instances.
  user_fun_y_pred_name = names(data_predicted)[end]

  data_predicted = DataFrames.unstack(data_predicted, [:index, :sample, :feature_name], :feature_group, user_fun_y_pred_name)

  # Shapley value for each Monte Carlo sample for each instance.
  data_predicted.shap_effect = data_predicted.real_target - data_predicted.fake_target
  #----------------------------------------------------------------------------
  data_predicted = DataFrames.select(data_predicted, [:index, :sample, :feature_name, :shap_effect])

  # Final Shapley value calculation collapsed across Monte Carlo samples.
  data_predicted = DataFrames.by(data_predicted, [:index, :feature_name],
                                 shap_effect_sd = :shap_effect => x -> std(x),
                                 shap_effect = :shap_effect => x -> mean(x),
                                 )

  data_predicted.intercept = repeat([intercept], size(data_predicted, 1))

  return data_predicted

end
