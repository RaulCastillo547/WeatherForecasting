import numpy as np
import pandas as pd

import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import xgboost as xgb

from datetime import timedelta, date, datetime
import json
import math
import os
import random

SAVED_MODEL_PATH = r'./saved_models'

# Seed Information
SEED = 0

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Makes many operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# LSTM Model
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
def train_climatological(train_df):
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

def train_linear(train_df):
    linear_train_X, linear_train_y = train_df.drop(columns=["TMAX", "TMIN"]).values, train_df[["TMAX", "TMIN"]].values

    linear_model = LinearRegression()
    linear_model.fit(linear_train_X, linear_train_y)

    artifact = {
        'model': linear_model,
        'features': list(train_df.drop(columns=["TMAX", "TMIN"]).columns),
        'targets': ["TMAX", "TMIN"],
    }

    joblib.dump(artifact, os.path.join(SAVED_MODEL_PATH, r'linear_model.joblib'))
    
    linear_normalized_predictions = linear_model.predict(linear_train_X)

    linear_unnormalized_predictions = linear_normalized_predictions*STD_TEMP.values + MEAN_TEMP.values
    linear_unnormalized_y = linear_train_y*STD_TEMP.values + MEAN_TEMP.values

    final_metrics = {
        'MAE': mean_absolute_error(linear_unnormalized_y, linear_unnormalized_predictions, multioutput="raw_values"),
        'RMSE': root_mean_squared_error(linear_unnormalized_y, linear_unnormalized_predictions, multioutput="raw_values"),
        'ME': mean_error(linear_unnormalized_y, linear_unnormalized_predictions),
    }

    return final_metrics
    
def train_xgboost(train_df):
    xgboost_train_X, xgboost_train_y = train_df.drop(columns=["TMAX", "TMIN"]), train_df[["TMAX", "TMIN"]]

    MODEL_PARAMS = {
        'colsample_bytree': 1.0,
        'gamma': 0,
        'learning_rate': 0.1,
        'max_depth': 3,
        'n_estimators': 200,
        'subsample': 1.0,
        'random_state': SEED
    }

    xgboost_model = xgb.XGBRegressor(**MODEL_PARAMS)
    xgboost_model.fit(xgboost_train_X, xgboost_train_y)

    xgboost_model.save_model(os.path.join(SAVED_MODEL_PATH, 'xgbregressor_model.json'))

    xgboost_normalized_predictions = xgboost_model.predict(xgboost_train_X)
    xgboost_unnormalized_predictions = xgboost_normalized_predictions*STD_TEMP.values + MEAN_TEMP.values
    xgboost_unnormalized_y = xgboost_train_y*STD_TEMP.values + MEAN_TEMP.values

    final_metrics = {
        'MAE': mean_absolute_error(xgboost_unnormalized_y.values, xgboost_unnormalized_predictions, multioutput="raw_values"),
        'RMSE': root_mean_squared_error(xgboost_unnormalized_y.values, xgboost_unnormalized_predictions, multioutput="raw_values"),
        'ME': mean_error(xgboost_unnormalized_y.values, xgboost_unnormalized_predictions),
    }
    
    return final_metrics

def train_lstm(train_dataset):
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

        all_predicted_temps = (all_predicted_temps*STD_TEMP.values) + MEAN_TEMP.values
        all_true_temps = (all_true_temps*STD_TEMP.values) + MEAN_TEMP.values

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

        all_predicted_temps = (all_predicted_temps*STD_TEMP.values) + MEAN_TEMP.values
        all_true_temps = (all_true_temps*STD_TEMP.values) + MEAN_TEMP.values

        return {
            "MAE": mean_absolute_error(all_true_temps, all_predicted_temps, multioutput="raw_values"),
            "RMSE": root_mean_squared_error(all_true_temps, all_predicted_temps, multioutput="raw_values"),
            "ME": mean_error(all_true_temps, all_predicted_temps),
            "total_loss": total_loss/len(dataloader.dataset)
        }
    
    BATCH_SIZE = 64
    EPOCHS = 10

    RNN_HIDDEN=128
    NN_HIDDEN=32
    LEARNING_RATE=0.001

    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    rnn_model = WeatherRNN(RNN_HIDDEN, NN_HIDDEN).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch_index in range(EPOCHS):
        train_one_epoch(rnn_model, loss_fn, device, train_dataloader, optimizer)

    torch.save(rnn_model.state_dict(), os.path.join(SAVED_MODEL_PATH, "lstm_model.pt"))

    final_metrics = evaluate_model(rnn_model, loss_fn, device, train_dataloader)

    return final_metrics

def main() -> None:
    set_seed(SEED)

    os.makedirs(SAVED_MODEL_PATH, exist_ok=True)
    raw_ghcnd = pd.read_csv(r'./data/raw/GHCND_1980_to_2026.csv', low_memory=False)
    raw_ghcnd["DATE"] = pd.to_datetime(raw_ghcnd["DATE"])
    raw_ghcnd = raw_ghcnd[raw_ghcnd.DATE.lt('2026-01-01')]

    global MEAN_TEMP, STD_TEMP
    MEAN_TEMP = raw_ghcnd[["TMAX", "TMIN"]].mean()
    STD_TEMP = raw_ghcnd[["TMAX", "TMIN"]].std()

    with open(os.path.join(SAVED_MODEL_PATH, "metadata.json"), 'w+') as fp:
        metadata_json = {
            "MEAN_TEMP": dict(MEAN_TEMP),
            "STD_TEMP": dict(STD_TEMP),
            "timestamp": datetime.now().isoformat()
        }

        json.dump(metadata_json, fp)

    print("Processing Datasets")

    # print("- Completed Persistence")

    climatological_train_set = climatological_ghcnd_dataset(raw_ghcnd)
    print("- Completed Climatology")
    
    linear_train_set = linear_ghcnd_dataset(raw_ghcnd)
    print("- Completed Linear")
    
    xgboost_train_set = xgboost_ghcnd_dataset(raw_ghcnd)
    print("- Completed XGBoost")

    lstm_train_set = RNN_GHCND_Dataset(raw_ghcnd)
    print("- Completed LSTM\n")

    print("Training Models")

    # print("- Completed Persistence (Nothing to Save)")

    climatological_metrics = train_climatological(climatological_train_set)
    print("- Completed Climatology")

    linear_metrics = train_linear(linear_train_set)
    print("- Completed Linear")

    xgboost_metrics = train_xgboost(xgboost_train_set)
    print("- Completed XGBoost")

    lstm_metrics = train_lstm(lstm_train_set)
    print("- Completed LSTM\n")

    print("Results:")
    print(f"|{"Model":^18}|{"TMAX MAE":^16}|{"TMIN MAE":^16}|{"TMAX RMSE":^16}|{"TMIN RMSE":^16}|{"TMAX ME":^16}|{"TMIN ME":^16}|")
    print(f"|{"Climatological":^18}|{climatological_metrics["MAE"][0]:^16.4e}|{climatological_metrics["MAE"][1]:^16.4e}|{climatological_metrics["RMSE"][0]:^16.4e}|{climatological_metrics["RMSE"][1]:^16.4e}|{climatological_metrics["ME"][0]:^16.4e}|{climatological_metrics["ME"][1]:^16.4e}|")
    print(f"|{"Linear":^18}|{linear_metrics["MAE"][0]:^16.4e}|{linear_metrics["MAE"][1]:^16.4e}|{linear_metrics["RMSE"][0]:^16.4e}|{linear_metrics["RMSE"][1]:^16.4e}|{linear_metrics["ME"][0]:^16.4e}|{linear_metrics["ME"][1]:^16.4e}|")
    print(f"|{"XGBoost":^18}|{xgboost_metrics["MAE"][0]:^16.4e}|{xgboost_metrics["MAE"][1]:^16.4e}|{xgboost_metrics["RMSE"][0]:^16.4e}|{xgboost_metrics["RMSE"][1]:^16.4e}|{xgboost_metrics["ME"][0]:^16.4e}|{xgboost_metrics["ME"][1]:^16.4e}|")
    print(f"|{"LSTM":^18}|{lstm_metrics["MAE"][0]:^16.4e}|{lstm_metrics["MAE"][1]:^16.4e}|{lstm_metrics["RMSE"][0]:^16.4e}|{lstm_metrics["RMSE"][1]:^16.4e}|{lstm_metrics["ME"][0]:^16.4e}|{lstm_metrics["ME"][1]:^16.4e}|")
