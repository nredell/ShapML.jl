# Internal prediction function. Used at the end of shap().
using Statistics

function _predict(;reference::DataFrame, data_predict::DataFrame, model, predict_function::Function, n_features::Integer,
                  n_target_features::Integer, n_instances_explain::Integer, sample_size::Integer, precision::Union{Integer, Nothing},
                  chunk::Bool, reconcile_instance::Bool, explain::DataFrame)

  if !chunk
    # For robustness across prediction functions from other languages (R and Python),
    # the modeling dataset needs to be created before it's passed in predict_function().
    # This is data-to-be-predicted but the name is kept the same throughout to reduce memory.
    data_predicted = data_predict[:, 1:n_features]

    # User-defined predict() function.
    data_predicted = predict_function(model, data_predicted)

    # A dataset with meta-data and the predictions in the last column.
    data_predicted = hcat(data_predict[:, (n_features + 1):size(data_predict, 2)], data_predicted, copycols = false)

  else

    data_predicted = copy(data_predict)
  end
  #----------------------------------------------------------------------------
  # Cast the DataFrame to, for each random sample, take the difference between the Frankenstein
  # instances.
  user_fun_y_pred_name = names(data_predicted)[end]

  # Returns a length 1 numeric vector of the average prediction--i.e., intercept--from the reference group.
  intercept = Statistics.mean(data_predicted[:, user_fun_y_pred_name])

  data_predicted = DataFrames.unstack(data_predicted, [:index, :sample, :feature_name], :feature_group, user_fun_y_pred_name)

  # Shapley value for each Monte Carlo sample for each instance.
  data_predicted.shap_effect = data_predicted.real_target - data_predicted.fake_target
  #----------------------------------------------------------------------------
  DataFrames.select!(data_predicted, [:index, :sample, :feature_name, :shap_effect])
  #----------------------------------------------------------------------------
  # Final Shapley value calculation collapsed across Monte Carlo samples using a
  # custom _aggregate() function for speed and reduced memory. The dimensions of
  # the output DataFrame, 'data_predicted', are ('n instances' * 'n features') by 5.
  if reconcile_instance
    data_predicted = _aggregate(data_predicted, sample_size, n_instances_explain, n_target_features, reconcile_instance)
  else
    data_predicted = _aggregate(data_predicted, sample_size, n_instances_explain, n_target_features, reconcile_instance)
  end
  #----------------------------------------------------------------------------
  # Adjust the instance-level Shapley values so that the sum across features equals the model prediction.
  if reconcile_instance
    # User-defined predict() function.
    data_model_pred = predict_function(model, explain)

    rename!(data_model_pred, Dict(user_fun_y_pred_name => "model_pred"))

    data_model_pred[:, :index] .= 1:n_instances_explain

    # data_predicted = join(data_predicted, data_model_pred, on = [:index], kind = :left)
    data_predicted = leftjoin(data_predicted, data_model_pred, on = [:index])
    data_shap_pred = DataFrames.combine(DataFrames.groupby(data_predicted, [:index]), :shap_effect => (x -> sum(x)) => :shap_pred)
    data_shap_pred.shap_pred .= data_shap_pred.shap_pred .+ intercept

    # data_predicted = join(data_predicted, data_shap_pred, on = [:index], kind = :left)
    data_predicted = leftjoin(data_predicted, data_shap_pred, on = [:index])

    data_predicted[:, :residual] = data_predicted.model_pred - data_predicted.shap_pred

    data_scale = DataFrames.combine(DataFrames.groupby(data_predicted, [:index]), :shap_effect_var => (x -> x / maximum(x)) => :variance_scale)

    data_predicted[:, :variance_scale] .= data_scale.variance_scale

    data_scale = DataFrames.combine(DataFrames.groupby(data_predicted, [:index]), :variance_scale => (x -> sum(x)) => :variance_scale_total)

    # data_predicted = join(data_predicted, data_scale, on = [:index], kind = :left)
    data_predicted = leftjoin(data_predicted, data_scale, on = [:index])

    data_predicted[:, :shap_effect] = data_predicted.shap_effect .+ (data_predicted.residual .* (data_predicted.variance_scale .- (data_predicted.variance_scale .* data_predicted.variance_scale_total) ./
                                                                 (1 .+ data_predicted.variance_scale_total)))

    DataFrames.select!(data_predicted, [:index, :feature_name, :shap_effect, :shap_effect_sd])
  end
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
