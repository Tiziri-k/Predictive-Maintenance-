
from __future__ import division
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import  BinaryClassificationEvaluator 
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import confusion_matrix
import numpy as np


def get_col_as_nparray(df, col_name):
   
    return np.array(df.select(col_name).collect())

def get_cm_metrics(matrix):
    
    d = matrix.shape[0]
    
    correct_pred = 0
    
    s1 = 0
    s2 = 0

    for i in range(0,d):
        correct_pred+= matrix[i,i]
        
        if np.sum(matrix[i,:])!= 0:
            s1+= matrix[i,i] / np.sum(matrix[i,:])
        
        if np.sum(matrix[:,i])!= 0:
            s2+= matrix[i,i] / np.sum(matrix[:,i])
    
    accuracy = correct_pred / np.sum(matrix)
    macro_p = s1 / d
    micro_p = correct_pred / np.sum(matrix)
    macro_r = s2 / d
    micro_r = micro_p
    macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r)
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r)
    
    return accuracy, macro_p, micro_p, macro_r, micro_r, macro_f1, micro_f1

def get_col_as_list(df, col_name):
    
    return df.select(col_name).rdd.flatMap(lambda x: x).collect()

def get_prediction_lists(prediction_df):
    l1 = list(map(int, get_col_as_list(prediction_df, "label")))
    l2 = list(map(int, get_col_as_list(prediction_df, "prediction")))
    m1 = get_col_as_nparray(prediction_df, "probability")
    
    return l1, l2, m1

def LogsticRegression(train_vector, test_df, engines_list):
    lr = LogisticRegression(featuresCol='features', labelCol='label', family='binomial', maxIter=10)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

    paramGrid = (ParamGridBuilder()
                 .addGrid(lr.regParam, [0.1, 0.3])
                 .addGrid(lr.elasticNetParam, [0.0])
                 .build())

    crossval = CrossValidator(estimator=lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=10)

    cvModel = crossval.fit(train_vector)

    for e in engines_list:
        print("Engine # ", e)
        test = test_df.filter(test_df.engine == e).select(test_df["sensor4_rollingmean_4_norm"],
                                                          test_df["sensor3_rollingmean_4_norm"],
                                                          test_df["classification"])

        test_vector = test.rdd.map(lambda x: [x[2], Vectors.dense(x[0:2])]).toDF(['label', 'features'])

        prediction_lr = cvModel.transform(test_vector)

        # Get labels, predictions, and probabilities as Python lists and arrays
        labels, preds, probs = get_prediction_lists(prediction_lr)

        # Calculate confusion matrix
        logreg_matrix = confusion_matrix(labels, preds)

        print('Confusion Matrix:\n', logreg_matrix)

        accuracy, macro_p, _, macro_r, _, macro_f1, _ = get_cm_metrics(logreg_matrix)
        print("accuracy= ", round(accuracy, 3))
        print("macro precision= ", round(macro_p, 3))
        print("macro recall= ", round(macro_r, 3))
        print("macro F1= ", round(macro_f1, 3))
