# MACHINE LEARNING MODEL EVALUATION MODULE STAGE 6

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


test_data = pd.read_csv("./data/data_preprocessing/test_data_preprocessed.csv")


# LOADING THE SAVED MODEL ARTIFACTS FOR MAKING THE PREDICTIONS

# THE PATH TO THE MODEL FOLDER
model_path = os.path.join(os.getcwd(), "model", "model_developed_artifacts")

with open(os.path.join(model_path, "model_artifact.pkl"), "rb") as file:
    model_in_production = pickle.load(file)

# ACCESSING THE MODELS

model_logReg = model_in_production["LOG_REG"]
model_rfc = model_in_production["RFCLASSIFIER"]


# MODEL EVALUATION FUNCTION

def model_prediction_and_evaluation(df, model_in_prod):
    X_test = df.drop("Occupancy", axis=1)
    y_test = df[["Occupancy"]]

    X_test_arr = X_test.values
    y_test_arr = y_test.values.ravel()

    y_pred = model_in_prod.predict(X_test_arr)

    acc = accuracy_score(y_test_arr, y_pred)
    prec = precision_score(y_test_arr, y_pred)
    rec = recall_score(y_test_arr, y_pred)
    f1score = f1_score(y_test_arr, y_pred)
    cm = confusion_matrix(y_test_arr, y_pred)
    print(f"Confusion Matrix: \n{cm}")

    return acc, prec, rec, f1score


acc, prec, rec, f1score = model_prediction_and_evaluation(df=test_data, model_in_prod=model_logReg)

# THE PATH TO THE METRICS EVALUATION FOLDER
metrics_path = os.path.join(os.getcwd(), "model_evaluation", "classification_metrics")
os.makedirs(metrics_path, exist_ok=True)


# SAVING THE METRICS SCORE INTO DICTIONARY
model_metrics_dict = {
    "ACCURACY": acc,
    "PRECISION": prec,
    "RECALL": rec,
    "F1_SCORE": f1score
}

with open(os.path.join(metrics_path, "model_metrics.json"), "w") as file:
    json.dump(model_metrics_dict, file, indent=4)
