from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors

def regression(train_vector,test_df,engines_list):

    lambda_param = [0]

    # set elastic net param
    elastic_net_param = [0]
        
    # build and evaluate linear regression model
    for l in lambda_param:
        for e in elastic_net_param:
            
            # init LR instance with parameters
            lr = LinearRegression(featuresCol = 'features', labelCol = 'label', maxIter=100, 
                                regParam=l, elasticNetParam=e)

            # fit linear regression model
            lr_model = lr.fit(train_vector)

            # print "Lambda = ", str(l), " elastic net = ", str(e)
            
            ## get and print model coefficients
            print("LR Coefficients: " + str(lr_model.coefficients))      
            
            # set lists of evaluation metrics
            r_squared_list_lr = []
            mse_list_lr = []
            rmse_list_lr = []
            mae_list_lr = []  
            
            # loop over list of engines to validate the model of each engine
            for e in engines_list[:8]:                                                                
                print ("Engine # ", e)
                
                # select test features (with same features set as of train vector)
                # test = test_df.filter(test_df.engine == e).select(test_df[ "sensor12_rollingmean_4_norm"],test_df["sensor7_rollingmean_4_norm"],test_df["rul"])
                
                # # build test features vector
                # test_vector = test.rdd.map(lambda x: [x[2], Vectors.dense(x[0:2])]).toDF(['label','features'])
                
                # get predictions
                pred = lr_model.transform(train_vector)

                ## evaluate regression model
                evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='label')
                
                # append evaluation metrics to evaluation lists
                r_squared_list_lr.append(evaluator.setMetricName('r2').evaluate(pred))
                mse_list_lr.append(evaluator.setMetricName('mse').evaluate(pred))
                rmse_list_lr.append(evaluator.setMetricName('rmse').evaluate(pred))
                mae_list_lr.append(evaluator.setMetricName('mae').evaluate(pred))
                
                # # print evaluation metrics
                # print ("R-squared= ", evaluator.setMetricName('r2').evaluate(pred))
                # print ("MSE= ", evaluator.setMetricName('mse').evaluate(pred))
                # print ("RMSE= ", evaluator.setMetricName('rmse').evaluate(pred))
                # print ("MAE= ", evaluator.setMetricName('mae').evaluate(pred))

                print("average R-Squared: ", sum(r_squared_list_lr) / float(len(r_squared_list_lr)))
                print("average MSE: ",sum(mse_list_lr) / float(len(mse_list_lr)))
                print( "average RMSE: ",sum(rmse_list_lr) / float(len(rmse_list_lr)))
                print( "average MAE: ",sum(mae_list_lr) / float(len(mae_list_lr)))
