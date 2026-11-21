from IPython.display import display

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import PredefinedSplit, GridSearchCV

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import xgboost as xgb

import math
from datetime import timedelta, date

# New Metric Functions
def mean_error(true_y:np.ndarray, pred_y:np.ndarray):
    return (pred_y - true_y).mean(axis=0)

def baseline_skill(baseline_metrics, pred_metrics):
    return 1 - (pred_metrics['MAE']/baseline_metrics['MAE'])

# TRAIN_MEAN & TRAIN_STD
raw_ghcnd = pd.read_csv(r'../data/raw/GHCND_1980_to_2026.csv')
raw_ghcnd["DATE"] = pd.to_datetime(raw_ghcnd["DATE"])

TRAIN_MEAN = raw_ghcnd.set_index("DATE")[["TMAX", "TMIN"]][(1980 <= raw_ghcnd.set_index("DATE").index.year) & (raw_ghcnd.set_index("DATE").index.year <= 2010)].interpolate(method="time").mean()
TRAIN_STD = raw_ghcnd.set_index("DATE")[["TMAX", "TMIN"]][(1980 <= raw_ghcnd.set_index("DATE").index.year) & (raw_ghcnd.set_index("DATE").index.year <= 2010)].interpolate(method="time").std()

del raw_ghcnd

# Persistence Dataset Function & Class
class PersistenceBaseline():
    def __init__(self, train_df=None):
        pass

    def predict(self, input_df):
        output_df = input_df[["TMAX_lag1", "TMIN_lag1"]].rename(columns={"TMAX_lag1": "predicted_TMAX", "TMIN_lag1": "predicted_TMIN"})
        return output_df

def persistence_ghcnd_dataset(raw_ghcnd):
    updated_df = raw_ghcnd.set_index("DATE")

    TEMP_FEATURES = [
        "TMAX", "TMIN",
        "TMAX_lag1", "TMIN_lag1"
    ]

    # Create Lag Feature
    updated_df["TMAX_lag1"] = updated_df["TMAX"].shift(1)
    updated_df["TMIN_lag1"] = updated_df["TMIN"].shift(1)

    # Split Data
    updated_df = updated_df[1:]
    train_df = updated_df[(1980 <= updated_df.index.year) & (updated_df.index.year <= 2010)][TEMP_FEATURES]
    val_df = updated_df[(2011 <= updated_df.index.year) & (updated_df.index.year <= 2017)][TEMP_FEATURES]
    test_df = updated_df[(2018 <= updated_df.index.year) & (updated_df.index.year <= 2025)][TEMP_FEATURES]

    # Interpolate Data
    train_df.interpolate(method="time", inplace=True)
    val_df.interpolate(method="time", inplace=True)
    test_df.interpolate(method="time", inplace=True)

    # Apply Shifts and Take Out First 6 Value
    return train_df, val_df, test_df

# Climatological Dataset Function & Class
class ClimatologicalBaseline():
    def __init__(self, train_df):
        tmp_df = train_df.copy(deep=True)
        tmp_df["DATE_OF_YEAR"] = train_df.index.strftime('%j').astype(int)
        self.climatological_model = tmp_df.groupby(by="DATE_OF_YEAR")[["TMAX", "TMIN"]].mean()

    def predict(self, input_df):
        tmp_df = input_df.copy(deep=True)
        tmp_df["DATE_OF_YEAR"] = tmp_df.index.strftime('%j').astype(int)
        return tmp_df[["DATE_OF_YEAR"]].join(self.climatological_model, on="DATE_OF_YEAR")[["TMAX", "TMIN"]].rename(columns={"TMAX": "predicted_TMAX", "TMIN": "predicted_TMIN"})

def climatological_ghcnd_dataset(raw_ghcnd):
    updated_df = raw_ghcnd.set_index("DATE")

    # Add Values Fourier Calendar Values
    TEMP_FEATURES = [
        "TMAX", "TMIN",
    ]
    
    # Split Data
    train_df = updated_df[(1980 <= updated_df.index.year) & (updated_df.index.year <= 2010)][TEMP_FEATURES]
    val_df = updated_df[(2011 <= updated_df.index.year) & (updated_df.index.year <= 2017)][TEMP_FEATURES]
    test_df = updated_df[(2018 <= updated_df.index.year) & (updated_df.index.year <= 2025)][TEMP_FEATURES]

    # Interpolate Data
    train_df.interpolate(method="time", inplace=True)
    val_df.interpolate(method="time", inplace=True)
    test_df.interpolate(method="time", inplace=True)

    # Apply Shifts and Take Out First 6 Value
    return train_df, val_df, test_df

# Linear Dataset Function
def linear_ghcnd_dataset(raw_ghcnd):
    updated_df = raw_ghcnd.set_index("DATE")
    updated_df["DATE_OF_YEAR"] = updated_df.index.map(
        lambda x: int(x.strftime('%j'))
    )

    # Add Values Fourier Calendar Values
    updated_df["FOURIER_SIN"] = updated_df.index.map(
        lambda x: math.sin(2*math.pi*(int(x.strftime('%j')) - 1)/365)
    )

    updated_df["FOURIER_COS"] = updated_df.index.map(
        lambda x: math.cos(2*math.pi*(int(x.strftime('%j')) - 1)/365)
    )

    # Normalize Temperature
    updated_df[["TMAX", "TMIN"]] = (updated_df[["TMAX", "TMIN"]] - TRAIN_MEAN)/TRAIN_STD
    
    TEMP_FEATURES = [
        "TMAX", "TMIN",
        "TMAX_lag1", "TMIN_lag1", 
        "TMAX_lag2", "TMIN_lag2", 
        "TMAX_lag3", "TMIN_lag3", 
        "TMAX_lag4", "TMIN_lag4", 
        "TMAX_lag5", "TMIN_lag5",
        "TMAX_lag6", "TMIN_lag6",
    ]
    PRCP_FEATURES = [
        "PRCP_lag1",
        "PRCP_lag2",
        "PRCP_lag3",
        "PRCP_lag4",
        "PRCP_lag5",
        "PRCP_lag6",
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
    updated_df["PRCP_lag1"] = updated_df["PRCP"].shift(1)
    updated_df["FOURIER_SIN_lag1"] = updated_df["FOURIER_SIN"].shift(1)
    updated_df["FOURIER_COS_lag1"] = updated_df["FOURIER_COS"].shift(1)

    updated_df["TMAX_lag2"] = updated_df["TMAX"].shift(2)
    updated_df["TMIN_lag2"] = updated_df["TMIN"].shift(2)
    updated_df["PRCP_lag2"] = updated_df["PRCP"].shift(2)
    updated_df["FOURIER_SIN_lag2"] = updated_df["FOURIER_SIN"].shift(2)
    updated_df["FOURIER_COS_lag2"] = updated_df["FOURIER_COS"].shift(2)

    updated_df["TMAX_lag3"] = updated_df["TMAX"].shift(3)
    updated_df["TMIN_lag3"] = updated_df["TMIN"].shift(3)
    updated_df["PRCP_lag3"] = updated_df["PRCP"].shift(3)
    updated_df["FOURIER_SIN_lag3"] = updated_df["FOURIER_SIN"].shift(3)
    updated_df["FOURIER_COS_lag3"] = updated_df["FOURIER_COS"].shift(3)

    updated_df["TMAX_lag4"] = updated_df["TMAX"].shift(4)
    updated_df["TMIN_lag4"] = updated_df["TMIN"].shift(4)
    updated_df["PRCP_lag4"] = updated_df["PRCP"].shift(4)
    updated_df["FOURIER_SIN_lag4"] = updated_df["FOURIER_SIN"].shift(4)
    updated_df["FOURIER_COS_lag4"] = updated_df["FOURIER_COS"].shift(4)

    updated_df["TMAX_lag5"] = updated_df["TMAX"].shift(5)
    updated_df["TMIN_lag5"] = updated_df["TMIN"].shift(5)
    updated_df["PRCP_lag5"] = updated_df["PRCP"].shift(5)
    updated_df["FOURIER_SIN_lag5"] = updated_df["FOURIER_SIN"].shift(5)
    updated_df["FOURIER_COS_lag5"] = updated_df["FOURIER_COS"].shift(5)

    updated_df["TMAX_lag6"] = updated_df["TMAX"].shift(6)
    updated_df["TMIN_lag6"] = updated_df["TMIN"].shift(6)
    updated_df["PRCP_lag6"] = updated_df["PRCP"].shift(6)
    updated_df["FOURIER_SIN_lag6"] = updated_df["FOURIER_SIN"].shift(6)
    updated_df["FOURIER_COS_lag6"] = updated_df["FOURIER_COS"].shift(6)
    
    # Split Data
    ALL_FEATURES = TEMP_FEATURES + FOURIER_FEATURES
    updated_df = updated_df[6:]
    train_df = updated_df[(1980 <= updated_df.index.year) & (updated_df.index.year <= 2010)][ALL_FEATURES]
    val_df = updated_df[(2011 <= updated_df.index.year) & (updated_df.index.year <= 2017)][ALL_FEATURES]
    test_df = updated_df[(2018 <= updated_df.index.year) & (updated_df.index.year <= 2025)][ALL_FEATURES]

    # Interpolate Data
    train_df.interpolate(method="time", inplace=True)
    val_df.interpolate(method="time", inplace=True)
    test_df.interpolate(method="time", inplace=True)

    # Apply Shifts and Take Out First 6 Value
    return train_df, val_df, test_df

# XGBoost Dataset Function
def xgboost_ghcnd_dataset(raw_ghcnd):
    updated_df = raw_ghcnd.set_index("DATE")
    updated_df["DATE_OF_YEAR"] = updated_df.index.map(
        lambda x: int(x.strftime('%j'))
    )

    # Add Values Fourier Calendar Values
    updated_df["FOURIER_SIN"] = updated_df.index.map(
        lambda x: math.sin(2*math.pi*(int(x.strftime('%j')) - 1)/365)
    )

    updated_df["FOURIER_COS"] = updated_df.index.map(
        lambda x: math.cos(2*math.pi*(int(x.strftime('%j')) - 1)/365)
    )

    # Normalize Temperature
    updated_df[["TMAX", "TMIN"]] = (updated_df[["TMAX", "TMIN"]] - TRAIN_MEAN)/TRAIN_STD
    
    TEMP_FEATURES = [
        "TMAX", "TMIN",
        "TMAX_lag1", "TMIN_lag1", 
        "TMAX_lag2", "TMIN_lag2", 
        "TMAX_lag3", "TMIN_lag3", 
        "TMAX_lag4", "TMIN_lag4", 
        "TMAX_lag5", "TMIN_lag5",
        "TMAX_lag6", "TMIN_lag6",
    ]
    PRCP_FEATURES = [
        "PRCP_lag1",
        "PRCP_lag2",
        "PRCP_lag3",
        "PRCP_lag4",
        "PRCP_lag5",
        "PRCP_lag6",
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
    updated_df["PRCP_lag1"] = updated_df["PRCP"].shift(1)
    updated_df["FOURIER_SIN_lag1"] = updated_df["FOURIER_SIN"].shift(1)
    updated_df["FOURIER_COS_lag1"] = updated_df["FOURIER_COS"].shift(1)

    updated_df["TMAX_lag2"] = updated_df["TMAX"].shift(2)
    updated_df["TMIN_lag2"] = updated_df["TMIN"].shift(2)
    updated_df["PRCP_lag2"] = updated_df["PRCP"].shift(2)
    updated_df["FOURIER_SIN_lag2"] = updated_df["FOURIER_SIN"].shift(2)
    updated_df["FOURIER_COS_lag2"] = updated_df["FOURIER_COS"].shift(2)

    updated_df["TMAX_lag3"] = updated_df["TMAX"].shift(3)
    updated_df["TMIN_lag3"] = updated_df["TMIN"].shift(3)
    updated_df["PRCP_lag3"] = updated_df["PRCP"].shift(3)
    updated_df["FOURIER_SIN_lag3"] = updated_df["FOURIER_SIN"].shift(3)
    updated_df["FOURIER_COS_lag3"] = updated_df["FOURIER_COS"].shift(3)

    updated_df["TMAX_lag4"] = updated_df["TMAX"].shift(4)
    updated_df["TMIN_lag4"] = updated_df["TMIN"].shift(4)
    updated_df["PRCP_lag4"] = updated_df["PRCP"].shift(4)
    updated_df["FOURIER_SIN_lag4"] = updated_df["FOURIER_SIN"].shift(4)
    updated_df["FOURIER_COS_lag4"] = updated_df["FOURIER_COS"].shift(4)

    updated_df["TMAX_lag5"] = updated_df["TMAX"].shift(5)
    updated_df["TMIN_lag5"] = updated_df["TMIN"].shift(5)
    updated_df["PRCP_lag5"] = updated_df["PRCP"].shift(5)
    updated_df["FOURIER_SIN_lag5"] = updated_df["FOURIER_SIN"].shift(5)
    updated_df["FOURIER_COS_lag5"] = updated_df["FOURIER_COS"].shift(5)

    updated_df["TMAX_lag6"] = updated_df["TMAX"].shift(6)
    updated_df["TMIN_lag6"] = updated_df["TMIN"].shift(6)
    updated_df["PRCP_lag6"] = updated_df["PRCP"].shift(6)
    updated_df["FOURIER_SIN_lag6"] = updated_df["FOURIER_SIN"].shift(6)
    updated_df["FOURIER_COS_lag6"] = updated_df["FOURIER_COS"].shift(6)
    
    # Split Data
    ALL_FEATURES = TEMP_FEATURES + FOURIER_FEATURES
    updated_df = updated_df[6:]
    train_df = updated_df[(1980 <= updated_df.index.year) & (updated_df.index.year <= 2010)][ALL_FEATURES]
    val_df = updated_df[(2011 <= updated_df.index.year) & (updated_df.index.year <= 2017)][ALL_FEATURES]
    test_df = updated_df[(2018 <= updated_df.index.year) & (updated_df.index.year <= 2025)][ALL_FEATURES]

    # Interpolate Data
    train_df.interpolate(method="time", inplace=True)
    val_df.interpolate(method="time", inplace=True)
    test_df.interpolate(method="time", inplace=True)

    # Apply Shifts and Take Out First 6 Value
    return train_df, val_df, test_df

# RNN Functions and Class
class WeatherRNN(nn.Module):
    def __init__(self, RNN_HIDDEN=32, NN_HIDDEN=64):
        super().__init__()
        self.rnn = nn.LSTM(input_size=4, hidden_size=RNN_HIDDEN, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(RNN_HIDDEN + 2, NN_HIDDEN),
            nn.ReLU(),
            nn.Linear(NN_HIDDEN, NN_HIDDEN),
            nn.ReLU(),
            nn.Linear(NN_HIDDEN, 2)
        )

    def forward(self, x: TensorDict):
        _, (rnn_output, _) = self.rnn(x["lag_features"]) # (batch_size, 6, 4) -> (1, batch_size, RNN_HIDDEN)
        fc_input = torch.concat((rnn_output[-1], x["fourier"]), axis=-1)
        fc_output = self.fc(fc_input)
        return fc_output

class RNN_GHCND_Dataset(Dataset):
    def __init__(self, ghcnd_df:pd.DataFrame):
        # Get Raw Value
        self.raw_df = ghcnd_df.copy(deep=True)
        self.raw_df.set_index("DATE", inplace=True)
        
        # Normalize Temperature
        self.raw_df[["TMAX", "TMIN"]] = (self.raw_df[["TMAX", "TMIN"]] - TRAIN_MEAN)/TRAIN_STD

        # Add Temporal Features
        self.raw_df["DATE_OF_YEAR"] = self.raw_df.index.map(
                lambda x: int(x.strftime('%j'))
            )
        
        # Add Values Fourier Calendar Values
        self.raw_df["FOURIER_SIN"] = self.raw_df.index.map(
            lambda x: math.sin(2*math.pi*(int(x.strftime('%j')) - 1)/365)
        )
    
        self.raw_df["FOURIER_COS"] = self.raw_df.index.map(
            lambda x: math.cos(2*math.pi*(int(x.strftime('%j')) - 1)/365)
        )

        self.processed_df = self.raw_df[["TMAX", "TMIN", "FOURIER_SIN", "FOURIER_COS"]]
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

def train_one_epoch(model, loss_fn:nn.Module, device:torch.Device, dataloader:DataLoader, optimizer:torch.optim.Optimizer):
    model.train()

    running_loss = 0
    all_predicted_temps = []
    all_true_temps = []
    for index, batch in enumerate(dataloader):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        temp_predictions = model(batch)
        loss_value = loss_fn(temp_predictions, batch["target"])
        loss_value.backward()

        optimizer.step()
        running_loss += loss_value.item()*batch.size(0)

        all_predicted_temps.append(temp_predictions)
        all_true_temps.append(batch["target"])

    all_predicted_temps = torch.concat(all_predicted_temps, axis=0).detach().numpy()
    all_true_temps = torch.concat(all_true_temps, axis=0).detach().numpy()

    all_predicted_temps = (all_predicted_temps*TRAIN_STD.values) + TRAIN_MEAN.values
    all_true_temps = (all_true_temps*TRAIN_STD.values) + TRAIN_MEAN.values

    return {
        "MAE": mean_absolute_error(all_true_temps, all_predicted_temps, multioutput="raw_values"),
        "RMSE": root_mean_squared_error(all_true_temps, all_predicted_temps, multioutput="raw_values"),
        "ME": mean_error(all_true_temps, all_predicted_temps),
        "running_loss": running_loss/len(dataloader.dataset)
    }

def evaluate_model(model, loss_fn:nn.Module, device:torch.Device, dataloader:DataLoader):
    model.eval()
    total_loss = 0
    all_predicted_temps = []
    all_true_temps = []

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            batch = batch.to(device)

            temp_predictions = model(batch)
            loss_value = loss_fn(temp_predictions, batch["target"])

            total_loss += loss_value.item()*batch.size(0)
            all_predicted_temps.append(temp_predictions)
            all_true_temps.append(batch["target"])

    all_predicted_temps = torch.concat(all_predicted_temps, axis=0).detach().numpy()
    all_true_temps = torch.concat(all_true_temps, axis=0).detach().numpy()

    all_predicted_temps = (all_predicted_temps*TRAIN_STD.values) + TRAIN_MEAN.values
    all_true_temps = (all_true_temps*TRAIN_STD.values) + TRAIN_MEAN.values

    return {
        "MAE": mean_absolute_error(all_true_temps, all_predicted_temps, multioutput="raw_values"),
        "RMSE": root_mean_squared_error(all_true_temps, all_predicted_temps, multioutput="raw_values"),
        "ME": mean_error(all_true_temps, all_predicted_temps),
        "total_loss": total_loss/len(dataloader.dataset)
    }