"""
Plan:
- Get models for persistence, climatology, linear regression, xgboost, and LSTM from whole dataset
    - Persistence: Nothing because there is nothing to train
    - Climatology: A CSV containing the averages for each day from 1980-2025
    - Linear Regression: Train & then save
    - XGBoost: Train & then save
    - LSTM: Train & save
- Also, save information on the dataset's mean and standard deviation
- Each trainable model will need its own data frame akin to what was used in the notebook

- Logical Flow:
    - Create functions for processing a dataset
    - Create functions for training & saving the models
        - Consider adding information about model performance: MAE, RMSE, ME
    - Main function will then run each process
"""

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from datetime import timedelta
import math
import os

SAVED_MODEL_PATH = r'./saved_models'
os.makedirs(SAVED_MODEL_PATH, exist_ok=True)

# Additional Metrics
def mean_error(true_y:np.ndarray, pred_y:np.ndarray):
    return (pred_y - true_y).mean(axis=0)

# Dataset Processing
def get_mean_std_temp(raw_ghcnd):
    pass

def climatological_ghcnd_dataset(raw_ghcnd):
    updated_df = raw_ghcnd.set_index("DATE")[["TMAX", "TMIN"]]
    updated_df.interpolate(method="time", inplace=True)
    return updated_df

def linear_ghcnd_dataset(raw_ghcnd):
    updated_df = raw_ghcnd.set_index("DATE")

    # Add Fourier Calendar Values
    updated_df["DATE_OF_YEAR"] = updated_df.index.map(
        lambda x: int(x.strftime('%j'))
    )

    updated_df["FOURIER_SIN"] = updated_df.index.map(
        lambda x: math.sin(2*math.pi*(int(x.strftime('%j')) - 1)/365)
    )

    updated_df["FOURIER_COS"] = updated_df.index.map(
        lambda x: math.cos(2*math.pi*(int(x.strftime('%j')) - 1)/365)
    )

    # Normalize Temperature
    MEAN_TEMP = updated_df[["TMAX", "TMIN"]].mean()
    STD_TEMP = updated_df[["TMAX", "TMIN"]].mean()
    updated_df[["TMAX", "TMIN"]] = (updated_df[["TMAX", "TMIN"]] - MEAN_TEMP)/STD_TEMP

    # Creating/Identifying Final Features
    TEMP_FEATURES = [
            "TMAX", "TMIN",
            "TMAX_lag1", "TMIN_lag1", 
            "TMAX_lag2", "TMIN_lag2", 
            "TMAX_lag3", "TMIN_lag3", 
            "TMAX_lag4", "TMIN_lag4", 
            "TMAX_lag5", "TMIN_lag5",
            "TMAX_lag6", "TMIN_lag6",
    ]

    FOURIER_FEATURES = [
        "FOURIER_SIN", "FOURIER_COS",
        "FOURIER_SIN_lag1", "FOURIER_COS_lag1",
        "FOURIER_SIN_lag2", "FOURIER_COS_lag2",
        "FOURIER_SIN_lag3", "FOURIER_COS_lag3",
        "FOURIER_SIN_lag4", "FOURIER_COS_lag4",
        "FOURIER_SIN_lag5", "FOURIER_COS_lag5",
        "FOURIER_SIN_lag6", "FOURIER_COS_lag6",
    ]

    updated_df["TMAX_lag1"] = updated_df["TMAX"].shift(1)
    updated_df["TMIN_lag1"] = updated_df["TMIN"].shift(1)
    updated_df["FOURIER_SIN_lag1"] = updated_df["FOURIER_SIN"].shift(1)
    updated_df["FOURIER_COS_lag1"] = updated_df["FOURIER_COS"].shift(1)

    updated_df["TMAX_lag2"] = updated_df["TMAX"].shift(2)
    updated_df["TMIN_lag2"] = updated_df["TMIN"].shift(2)
    updated_df["FOURIER_SIN_lag2"] = updated_df["FOURIER_SIN"].shift(2)
    updated_df["FOURIER_COS_lag2"] = updated_df["FOURIER_COS"].shift(2)

    updated_df["TMAX_lag3"] = updated_df["TMAX"].shift(3)
    updated_df["TMIN_lag3"] = updated_df["TMIN"].shift(3)
    updated_df["FOURIER_SIN_lag3"] = updated_df["FOURIER_SIN"].shift(3)
    updated_df["FOURIER_COS_lag3"] = updated_df["FOURIER_COS"].shift(3)

    updated_df["TMAX_lag4"] = updated_df["TMAX"].shift(4)
    updated_df["TMIN_lag4"] = updated_df["TMIN"].shift(4)
    updated_df["FOURIER_SIN_lag4"] = updated_df["FOURIER_SIN"].shift(4)
    updated_df["FOURIER_COS_lag4"] = updated_df["FOURIER_COS"].shift(4)

    updated_df["TMAX_lag5"] = updated_df["TMAX"].shift(5)
    updated_df["TMIN_lag5"] = updated_df["TMIN"].shift(5)
    updated_df["FOURIER_SIN_lag5"] = updated_df["FOURIER_SIN"].shift(5)
    updated_df["FOURIER_COS_lag5"] = updated_df["FOURIER_COS"].shift(5)

    updated_df["TMAX_lag6"] = updated_df["TMAX"].shift(6)
    updated_df["TMIN_lag6"] = updated_df["TMIN"].shift(6)
    updated_df["FOURIER_SIN_lag6"] = updated_df["FOURIER_SIN"].shift(6)
    updated_df["FOURIER_COS_lag6"] = updated_df["FOURIER_COS"].shift(6)

    # Simplifying Dataset
    ALL_FEATURES = TEMP_FEATURES + FOURIER_FEATURES
    updated_df = updated_df[6:][ALL_FEATURES]

    # Interpolate Data
    updated_df.interpolate(method="time", inplace=True)

    return updated_df

def xgboost_ghcnd_dataset(raw_ghcnd):
    updated_df = raw_ghcnd.set_index("DATE")

    # Add Fourier Calendar Values
    updated_df["DATE_OF_YEAR"] = updated_df.index.map(
        lambda x: int(x.strftime('%j'))
    )

    updated_df["FOURIER_SIN"] = updated_df.index.map(
        lambda x: math.sin(2*math.pi*(int(x.strftime('%j')) - 1)/365)
    )

    updated_df["FOURIER_COS"] = updated_df.index.map(
        lambda x: math.cos(2*math.pi*(int(x.strftime('%j')) - 1)/365)
    )

    # Normalize Temperature
    MEAN_TEMP = updated_df[["TMAX", "TMIN"]].mean()
    STD_TEMP = updated_df[["TMAX", "TMIN"]].mean()
    updated_df[["TMAX", "TMIN"]] = (updated_df[["TMAX", "TMIN"]] - MEAN_TEMP)/STD_TEMP
    
    # Creating/Identifying Final Features
    TEMP_FEATURES = [
            "TMAX", "TMIN",
            "TMAX_lag1", "TMIN_lag1", 
            "TMAX_lag2", "TMIN_lag2", 
            "TMAX_lag3", "TMIN_lag3", 
            "TMAX_lag4", "TMIN_lag4", 
            "TMAX_lag5", "TMIN_lag5",
            "TMAX_lag6", "TMIN_lag6",
    ]

    FOURIER_FEATURES = [
        "DATE_OF_YEAR",
        "FOURIER_SIN", "FOURIER_COS",
        "FOURIER_SIN_lag1", "FOURIER_COS_lag1",
        "FOURIER_SIN_lag2", "FOURIER_COS_lag2",
        "FOURIER_SIN_lag3", "FOURIER_COS_lag3",
        "FOURIER_SIN_lag4", "FOURIER_COS_lag4",
        "FOURIER_SIN_lag5", "FOURIER_COS_lag5",
        "FOURIER_SIN_lag6", "FOURIER_COS_lag6",
    ]

    updated_df["TMAX_lag1"] = updated_df["TMAX"].shift(1)
    updated_df["TMIN_lag1"] = updated_df["TMIN"].shift(1)
    updated_df["FOURIER_SIN_lag1"] = updated_df["FOURIER_SIN"].shift(1)
    updated_df["FOURIER_COS_lag1"] = updated_df["FOURIER_COS"].shift(1)

    updated_df["TMAX_lag2"] = updated_df["TMAX"].shift(2)
    updated_df["TMIN_lag2"] = updated_df["TMIN"].shift(2)
    updated_df["FOURIER_SIN_lag2"] = updated_df["FOURIER_SIN"].shift(2)
    updated_df["FOURIER_COS_lag2"] = updated_df["FOURIER_COS"].shift(2)

    updated_df["TMAX_lag3"] = updated_df["TMAX"].shift(3)
    updated_df["TMIN_lag3"] = updated_df["TMIN"].shift(3)
    updated_df["FOURIER_SIN_lag3"] = updated_df["FOURIER_SIN"].shift(3)
    updated_df["FOURIER_COS_lag3"] = updated_df["FOURIER_COS"].shift(3)

    updated_df["TMAX_lag4"] = updated_df["TMAX"].shift(4)
    updated_df["TMIN_lag4"] = updated_df["TMIN"].shift(4)
    updated_df["FOURIER_SIN_lag4"] = updated_df["FOURIER_SIN"].shift(4)
    updated_df["FOURIER_COS_lag4"] = updated_df["FOURIER_COS"].shift(4)

    updated_df["TMAX_lag5"] = updated_df["TMAX"].shift(5)
    updated_df["TMIN_lag5"] = updated_df["TMIN"].shift(5)
    updated_df["FOURIER_SIN_lag5"] = updated_df["FOURIER_SIN"].shift(5)
    updated_df["FOURIER_COS_lag5"] = updated_df["FOURIER_COS"].shift(5)

    updated_df["TMAX_lag6"] = updated_df["TMAX"].shift(6)
    updated_df["TMIN_lag6"] = updated_df["TMIN"].shift(6)
    updated_df["FOURIER_SIN_lag6"] = updated_df["FOURIER_SIN"].shift(6)
    updated_df["FOURIER_COS_lag6"] = updated_df["FOURIER_COS"].shift(6)

    ALL_FEATURES = TEMP_FEATURES + FOURIER_FEATURES

    updated_df = updated_df[6:][ALL_FEATURES]
    updated_df.interpolate(method="time", inplace=True)

    return updated_df

class RNN_GHCND_Dataset(Dataset):
    def __init__(self, raw_ghcd):
        # Get Raw Value
        self.updated_df = raw_ghcd.copy(deep=True)
        self.updated_df.set_index("DATE", inplace=True)

        # Normalize Temperature
        MEAN_TEMP = self.updated_df[["TMAX", "TMIN"]].mean()
        STD_TEMP = self.updated_df[["TMAX", "TMIN"]].std()
        self.updated_df[["TMAX", "TMIN"]] = (self.updated_df[["TMAX", "TMIN"]] - MEAN_TEMP)/STD_TEMP

        # Add Temporal Features
        self.updated_df["DATE_OF_YEAR"] = self.updated_df.index.map(
            lambda x: int(x.strftime('%j'))
        )
        self.updated_df["FOURIER_SIN"] = self.updated_df.index.map(
            lambda x: math.sin(2*math.pi*(int(x.strftime('%j')) - 1)/365)
        )
        self.updated_df["FOURIER_COS"] = self.updated_df.index.map(
            lambda x: math.cos(2*math.pi*(int(x.strftime('%j')) - 1)/365)
        )

        self.processed_df = self.updated_df[["TMAX", "TMIN", "FOURIER_SIN", "FOURIER_COS"]]
        self.processed_df.interpolate(method="time", inplace=True)

        # Get Lag Features Plus Current Time
        self.all_lag_features = []
        self.all_current_fourier = []
        self.all_target_temps = []

        for index, curr_date in enumerate(self.processed_df.index[6:], 6):
            self.all_lag_features.append(
                [
                    self.processed_df.loc[curr_date - timedelta(days=6), ["TMAX", "TMIN", "FOURIER_SIN", "FOURIER_COS"]].values,
                    self.processed_df.loc[curr_date - timedelta(days=5), ["TMAX", "TMIN", "FOURIER_SIN", "FOURIER_COS"]].values,
                    self.processed_df.loc[curr_date - timedelta(days=4), ["TMAX", "TMIN", "FOURIER_SIN", "FOURIER_COS"]].values,
                    self.processed_df.loc[curr_date - timedelta(days=3), ["TMAX", "TMIN", "FOURIER_SIN", "FOURIER_COS"]].values,
                    self.processed_df.loc[curr_date - timedelta(days=2), ["TMAX", "TMIN", "FOURIER_SIN", "FOURIER_COS"]].values,
                    self.processed_df.loc[curr_date - timedelta(days=1), ["TMAX", "TMIN", "FOURIER_SIN", "FOURIER_COS"]].values,
                ]
            )
            self.all_current_fourier.append(
                self.processed_df.loc[curr_date, ["FOURIER_SIN", "FOURIER_COS"]].values
            )
            self.all_target_temps.append(
                self.processed_df.loc[curr_date, ["TMAX", "TMIN"]].values
            )

        self.all_lag_features = np.array(self.all_lag_features)
        self.all_current_fourier = np.array(self.all_current_fourier)
        self.all_target_temps = np.array(self.all_target_temps)

    def __len__(self):
        return len(self.all_lag_features)

    def __getitem__(self, idx):
        return TensorDict(
            {
                "lag_features": torch.Tensor(self.all_lag_features[idx]),
                "fourier": torch.Tensor(self.all_current_fourier[idx]),
                "target": torch.Tensor(self.all_target_temps[idx])
            },
            batch_size=[]
        )

# Training Models
def train_climatology(train_df):
    tmp_df = train_df.copy(deep=True)
    tmp_df["DATE_OF_YEAR"] = train_df.index.strftime('%j').astype(int)
    climatological_model = tmp_df.groupby(by="DATE_OF_YEAR")[["TMAX", "TMIN"]].mean()

    climatological_model.to_csv(os.path.join(SAVED_MODEL_PATH, r'climatological_model.csv'))

    predicted_df = tmp_df[["DATE_OF_YEAR"]].join(climatological_model, on="DATE_OF_YEAR")[["TMAX", "TMIN"]]

    final_metrics = {
        'MAE': mean_absolute_error(train_df.values, predicted_df.values, multioutput="raw_values"),
        'RMSE': root_mean_squared_error(train_df.values, predicted_df.values, multioutput="raw_values"),
        'ME': mean_error(train_df.values, predicted_df.values),
    }

    return final_metrics



def train_linear():
    pass

def train_xgboost():
    pass

def train_lstm():
    pass

def main() -> None:
    raw_ghcnd = pd.read_csv(r'./data/raw/GHCND_1980_to_2026.csv', low_memory=False)
    raw_ghcnd["DATE"] = pd.to_datetime(raw_ghcnd["DATE"])

    print("Processing Datasets")
    climatological_train_set = climatological_ghcnd_dataset(raw_ghcnd)
    print("- Completed Climatology")
    
    linear_train_set = linear_ghcnd_dataset(raw_ghcnd)
    print(linear_train_set.head())
    print("- Completed Linear")
    
    xgboost_train_set = xgboost_ghcnd_dataset(raw_ghcnd)
    print(xgboost_train_set.head())
    print("- Completed XGBoost")

    lstm_train_set = RNN_GHCND_Dataset(raw_ghcnd)
    print("- Completed LSTM\n")


    print("Training Models")

    print("- Completed Climatology")
    climatological_metrics = train_climatology(climatological_train_set)
    print(climatological_metrics)

    print("- Completed Linear")

    print("- Completed XGBoost")

    print("- Completed LSTM\n")


    print("Results:")
