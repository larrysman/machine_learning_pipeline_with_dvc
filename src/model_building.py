# MACHINE LEARNING MODEL DEVELOPMENT MODULE STAGE 5

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv("./data/data_preprocessing/train_data_preprocessed.csv")

def model_building(df, model):
    X_train = df.drop("Occupancy", axis=1)
    y_train = df[["Occupancy"]]

    X_train_arr = X_train.values
    y_train_arr = y_train.values.ravel()

    model.fit(X_train_arr, y_train_arr)

    return model

log_reg_model = model_building(train_data, model=LogisticRegression())
rfc_model = model_building(train_data, model=RandomForestClassifier(random_state=42))

# THE PATH TO THE OUTPUT FOLDERS
model_path = os.path.join(os.getcwd(), "model", "model_developed_artifacts")
os.makedirs(model_path, exist_ok=True)


# SAVING THE MODEL ARTIFACTS INTO THE FOLDERS
model_artifacts = {
    "LOG_REG": log_reg_model,
    "RFCLASSIFIER": rfc_model
}

with open(os.path.join(model_path, "model_artifact.pkl"), "wb") as file:
    pickle.dump(model_artifacts, file)
