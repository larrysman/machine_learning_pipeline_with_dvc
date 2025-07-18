# FEATURES ENGINEERING MODULE STAGE 3

import os
import numpy as np
import pandas as pd


# READ THE DATA FROM THE SRC/PREPARATION FOLDER FROM DATA PREPARATION STAGE

train_data_cleaned = pd.read_csv("./data/preparation/train_data_cleaned.csv")
test_data_cleaned = pd.read_csv("./data/preparation/test_data_cleaned.csv")


# FEATURE ENGINEERING USING THE DATE COLUMN

def feature_engineered_by_date(df):
    df["date"] = pd.to_datetime(df["date"])
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.day_name()
    df = df.drop(columns=["date"])
    return df

train_engineered = feature_engineered_by_date(train_data_cleaned)
test_engineered = feature_engineered_by_date(test_data_cleaned)

data_path = os.path.join(os.getcwd(), "data", "feature_engineering")
os.makedirs(data_path, exist_ok=True)


train_engineered.to_csv(os.path.join(data_path, "train_engineered_data.csv"), index=False)
test_engineered.to_csv(os.path.join(data_path, "test_engineered_data.csv"), index=False)
