
from __future__ import division
from pyspark.ml.classification import   RandomForestClassifier
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

    for i in range(d):
        correct_pred += matrix[i, i]

        if np.sum(matrix[i, :]) != 0:
            s1 += matrix[i, i] / np.sum(matrix[i, :])

        if np.sum(matrix[:, i]) != 0:
            s2 += matrix[i, i] / np.sum(matrix[:, i])

    accuracy = correct_pred / np.sum(matrix)
    macro_p = s1 / d
    micro_p = correct_pred / np.sum(matrix)
    macro_r = s2 / d
    micro_r = micro_p

    if (macro_p + macro_r) == 0:
        macro_f1 = 0
    else:
        macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r)

    if (micro_p + micro_r) == 0:
        micro_f1 = 0
    else:
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r)

    return accuracy, macro_p, micro_p, macro_r, micro_r, macro_f1, micro_f1

def get_col_as_list(df, col_name):
    
    return df.select(col_name).rdd.flatMap(lambda x: x).collect()

def get_prediction_lists(prediction_df):
    l1 = list(map(int, get_col_as_list(prediction_df, "label")))
    l2 = list(map(int, get_col_as_list(prediction_df, "prediction")))
    m1 = get_col_as_nparray(prediction_df, "probability")
    
    return l1, l2, m1

def summarize_metrics(eval_list):
    metrics_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]

    summaries = {metrics_names[i]: { np.mean(metric)}
                 for i, metric in enumerate(eval_list)}
    print(f"Random Forest: \n {summaries}")


def RandomForest(train_vector, test_df, engines_list):
    rf = RandomForestClassifier(featuresCol='features', labelCol='label')

    paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="label", metricName="areaUnderROC")

    crossval = CrossValidator(estimator=rf,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=5)

    cvModel_rf = crossval.fit(train_vector)

    accuracy_list_rf = []
    mac_p_list_rf = []
    mac_r_list_rf = []
    mac_F1_list_rf = []

    for e in engines_list:
        # print("Engine # ", e)

        test = test_df.filter(test_df.engine == e).select(
            test_df["sensor3_rollingmean_4_norm"],
            test_df["sensor4_rollingmean_4_norm"],
            test_df["sensor7_rollingmean_4_norm"],
            test_df["sensor12_rollingmean_4_norm"],
            test_df["sensor17_rollingmean_4_norm"],
            test_df["classification"]
        )

        test_vector = test.rdd.map(lambda x: [x[5], Vectors.dense(x[0:5])]).toDF(['label', 'features'])

        prediction_rf = cvModel_rf.transform(train_vector)

        labels, preds, probs = get_prediction_lists(prediction_rf)

        rf_matrix = confusion_matrix(labels, preds)

        # print('Confusion Matrix:\n', rf_matrix)

        accuracy, macro_p, micro_p, macro_r, _, macro_f1, micro_f1 = get_cm_metrics(rf_matrix)
        accuracy_list_rf.append(round(accuracy, 3))
        mac_p_list_rf.append(round(macro_p, 3))
        mac_r_list_rf.append(round(macro_r, 3))
        mac_F1_list_rf.append(round(macro_f1, 3))
        # print("accuracy= ", round(accuracy, 3))
        # print("macro precision= ", round(macro_p, 3))
        # print("macro recall= ", round(macro_r, 3))
        # print("macro F1= ", round(macro_f1, 3))

        # print(accuracy_list_rf)
    summarize_metrics([accuracy_list_rf, mac_p_list_rf, mac_r_list_rf, mac_F1_list_rf])
    #return accuracy_list_rf, mac_p_list_rf, mac_r_list_rf, mac_F1_list_rf
