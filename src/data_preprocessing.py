# DATA PREPROCESSING MODULE STAGE 4

import os
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from imblearn.combine import SMOTEENN


train_data_prepro = pd.read_csv("./data/feature_engineering/train_engineered_data.csv").drop("index", axis=1)
test_data_prepro = pd.read_csv("./data/feature_engineering/test_engineered_data.csv").drop("index", axis=1)

# PREPROCESSING THE TRAINING DATA
def train_data_preprocessing(df):
    features = df.drop("Occupancy", axis=1)
    target = df[["Occupancy"]]

    num_feat = features.select_dtypes(["float64", "int64"])
    cat_feat = features.select_dtypes(["object"])

    # Numerical Preprocessing
    scaler = StandardScaler()
    min_max = MinMaxScaler()
    
    num_sc = scaler.fit_transform(num_feat)
    num_min = min_max.fit_transform(num_sc)
    num_preprocessed = pd.DataFrame(num_min, columns=num_feat.columns)

    # Categorical Preprocessing (Encoding)
    onehot = OneHotEncoder(sparse_output=False).set_output(transform="pandas")

    cat = onehot.fit_transform(cat_feat)
    cat_preprocessed = cat.reset_index(drop=True)

    features_preprocessed = pd.concat([num_preprocessed, cat_preprocessed, target], axis=1)

    # CORRECTING THE DATA IMBALANCE IN THE TARGET
    smote_enn = SMOTEENN()

    X = features_preprocessed.drop("Occupancy", axis=1)
    Y = features_preprocessed[["Occupancy"]]

    X_bal_feat, Y_bal_targ = smote_enn.fit_resample(X, Y)

    train_data_preprocessed = pd.concat([X_bal_feat, Y_bal_targ], axis=1)

    return onehot, scaler, min_max, train_data_preprocessed

onehot, scaler, min_max, train_data_preprocessed = train_data_preprocessing(train_data_prepro)


# PREPROCESSING THE TEST DATA
def test_data_preprocessing(df):
    features = df.drop("Occupancy", axis=1)
    target = df[["Occupancy"]]

    num_feat = features.select_dtypes(["float64", "int64"])
    cat_feat = features.select_dtypes(["object"])

    # Numerical Preprocessing
    scaler
    min_max
    
    num_sc = scaler.transform(num_feat)
    num_min = min_max.transform(num_sc)
    num_preprocessed = pd.DataFrame(num_min, columns=num_feat.columns)

    # Categorical Preprocessing (Encoding)
    onehot

    cat = onehot.transform(cat_feat)
    cat_preprocessed = cat.reset_index(drop=True)

    test_data_preprocessed = pd.concat([num_preprocessed, cat_preprocessed, target], axis=1)

    return test_data_preprocessed

test_data_preprocessed = test_data_preprocessing(test_data_prepro)


# THE PATH TO THE OUTPUT FOLDERS
data_path = os.path.join(os.getcwd(), "data", "data_preprocessing")
model_path = os.path.join(os.getcwd(), "model", "data_preprocessing_artifacts")

os.makedirs(data_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# SAVING THE DATA INTO THE FOLDERS
train_data_preprocessed.to_csv(os.path.join(data_path, "train_data_preprocessed.csv"), index=False)
test_data_preprocessed.to_csv(os.path.join(data_path, "test_data_preprocessed.csv"), index=False)

# SAVING THE DATA PREPROCESSING ARTIFACTS INTO THE FOLDERS
data_preprocessing_artifact = {
    "STANDARDSCALER": scaler,
    "MINMAXSCALER": min_max,
    "ONEHOTENCODER": onehot
}

with open(os.path.join(model_path, "data_preprocessing_artifacts.pkl"), "wb") as file:
    pickle.dump(data_preprocessing_artifact, file)



