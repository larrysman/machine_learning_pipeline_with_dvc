# DATA COLLECTION MODULE STAGE 1

import os
import numpy as np
import pandas as pd
import config
from sklearn.model_selection import train_test_split


# PATH TO THE RAW DATA
raw_data_and_path = config.RAW_DATA_PATH["PATH"]


# READING THE DATA WITH PANDAS
data = pd.read_csv(raw_data_and_path)

# SPLITTING THE DATA INTO TRAIN AND TEST
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# CREATING THE OUTPUT FOLDERS TO STORE THE OUTCOMES OF THIS STAGE
data_path = os.path.join(os.getcwd(), "data", "raw")
os.makedirs(data_path, exist_ok=True)

# SAVING THE OUTCOMES FILE INTO THE CREATED FOLDERS
train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)