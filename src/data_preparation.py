# DATA PREPARATION MODULE STAGE 2

import os
import numpy as np
import pandas as pd


# READ THE DATA FROM THE SRC/RAW FOLDER FROM DATA COLLECTION STAGE

train_data = pd.read_csv("./data/raw/train.csv")
test_data = pd.read_csv("./data/raw/test.csv")


# DATA PREPARATION

def remove_duplicates(df):
    df.drop_duplicates(keep="first", inplace=True)
    df.reset_index(inplace=True)
    return df

def fill_missing_value_with_median(df):
    for col in df.columns:
        if df[col].isnull().any():
            med_val = df[col].median()
            df[col].fillna(med_val, inplace=True)
    return df

train_data_cleaned = remove_duplicates(train_data)
test_data_cleaned = remove_duplicates(test_data)


data_path = os.path.join(os.getcwd(), "data", "preparation")
os.makedirs(data_path, exist_ok=True)

train_data_cleaned.to_csv(os.path.join(data_path, "train_data_cleaned.csv"), index=False)
test_data_cleaned.to_csv(os.path.join(data_path, "test_data_cleaned.csv"), index=False)