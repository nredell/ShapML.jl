module TestShap

using ShapML
using Test
using DataFrames
using Random
using Statistics
using Distributed

@testset "shap return values are correct." begin

    #--------------------------------------------------------------------------
    data = DataFrame(y = 1:10, x1 = 1:10, x2 = 1:10, x3 = 1:10)

    function model_mean(data)
        y_pred = Array{Float64}(undef, size(data, 1))
        for i in 1:size(data, 1)
            y_pred[i] = mean(data[i, 2:end])
        end
        return DataFrame(y_pred = y_pred)
    end

    function predict_function(model, data)
      data_pred = model(data)
      return data_pred
    end

    data_shap = ShapML.shap(explain = data[:, 2:end],
                            model = model_mean,
                            predict_function = predict_function,
                            sample_size = 10
                            )

    #--------------------------------------------------------------------------
    # Test that each instance has a Shapley value for each feature.
    @test size(data_shap, 1) == (size(data, 1) * (size(data, 2) - 1))
    #--------------------------------------------------------------------------
    # Test that the Shapley values are the same for a subset of target features.
    data_shap_feature = ShapML.shap(explain = data[:, 2:end],
                                    model = model_mean,
                                    predict_function = predict_function,
                                    target_features = ["x2"],
                                    sample_size = 10
                                    )

    @test all(data_shap[data_shap.feature_name .== "x2", [:shap_effect]].shap_effect .== data_shap_feature[data_shap_feature.feature_name .== "x2", [:shap_effect]].shap_effect)
    #--------------------------------------------------------------------------
end
#------------------------------------------------------------------------------
@testset "The NULL model returns 0 Shapley values." begin

    #--------------------------------------------------------------------------
    data = DataFrame(y = 1:10, x1 = 1:10, x2 = 1:10)

    function model_mean(data)
        return DataFrame(y_pred = repeat([0], size(data, 1)))
    end

    function predict_function(model, data)
      data_pred = model(data)
      return data_pred
    end

    data_shap = ShapML.shap(explain = data[:, 2:end],
                            model = model_mean,
                            predict_function = predict_function,
                            sample_size = 10
                            )

    @test all(data_shap[:, [:shap_effect]].shap_effect .== 0)
    @test all(data_shap[:, [:shap_effect_sd]].shap_effect_sd .== 0)
    #--------------------------------------------------------------------------
end
end  # End test module.
