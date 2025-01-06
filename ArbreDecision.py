from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import  DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from numpy import median


def Arbre_de_decision(train_vector,test_df,engines_list):

    maxDepth = [5, 10]

    maxBins = [4, 8]


    for m1 in maxDepth:
        for m2 in maxBins:
            
            dt = DecisionTreeRegressor(featuresCol = 'features', labelCol = 'label', maxDepth=m1, maxBins=m2)

            dt_model = dt.fit(train_vector)
        
            
            r_squared_list_dt = []
            mse_list_dt = []
            rmse_list_dt = []
            mae_list_dt = []  
            
            for e in engines_list: 
                test = test_df.filter(test_df.id == e).select(
                            test_df["s12_rollingmean_4_norm"],test_df["s7_rollingmean_4_scaled"],  test_df["rul"])
                
                test_vector = test.rdd.map(lambda x: [x[5], Vectors.dense(x[0:5])]).toDF(['label','features'])
                
                pred2 = dt_model.transform(train_vector)

                evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='label')
                
                r_squared_list_dt.append(evaluator.setMetricName('r2').evaluate(pred2))
                mse_list_dt.append(evaluator.setMetricName('mse').evaluate(pred2))
                rmse_list_dt.append(evaluator.setMetricName('rmse').evaluate(pred2))
                mae_list_dt.append(evaluator.setMetricName('mae').evaluate(pred2))                     
                
            print ("median R-Squared: ", median(r_squared_list_dt))
            print ("median MSE: ",median(mse_list_dt) )
            print ("median RMSE: ",median(rmse_list_dt)) 
            print ("median MAE: ",median(mae_list_dt))
            
         