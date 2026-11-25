# Weather Forecasting in Tampa International Airport (TPA)
This project aims to create machine learning models that predict a day's daily high and low temperatures using information from the past 6 days. These models will be deployed within an etl pipeline that runs daily temperature predictions throughout 2027. Users can view the predictions in an associated web application.

## Project Overview
### Goal
The project's development is split into three main stages:
- **Exploratory Data Analysis & Model Training:** Analyze bulk historical weather data from TPA to identify important features for predicting daily high and low temperature. The data will then be used to train a series of baseline and ml models for daily high and low temperature prediction
- **Data Engineering:** Create an ETL pipeline that collects recent temperature data from TPA and uses the previously defined models to create predictions for the current day's daily high and low temperatures. The etl pipeline script will be run daily throughout 2027, with predictions and input data being stored in SQL databases.
- **Web Development:** Create and deploy a web application than enables users to view daily temperature predictions made throughout 2027.

This project came out of a desire to practice "going beyond the jupyter notebook". I wanted to actually use the models I have trained in a deployed setting in addition to touch on skills related to Data Engineering (with ETL) and Software Engineering (with web development).

Currently, I am wrapping up model training and will begin working on creating the ETL pipeline before starting on the final web application. 

### Data Sources
This project makes use of two main data sources:
1. `data/raw/GHCND_1980_to_2026.csv`: CSV file collected from NOAA's larger GHCNd database used for training models. Contains daily weather data from TPA from 1/1/1980-10/18/2026; however, data from 1980-2025 was used
2. [NOAA's weather.gov API](https://api.weather.gov/): REST API offered by NOAA that provides realtime access to weather readings at TPA.

### File Structure

```text
WeatherPrediction/
├── artifacts/                          # Latest production models and final training metrics
├── data/                               # Training GHCND CSV and recent API call samples
├── notebooks/
│   ├── figures/                        # Figures generated from model_baseline_training.ipynb
│   ├── data_analysis.ipynb             # Exploratory data analysis
│   ├── model_baseline_training.ipynb   # Model analysis over train/validation/test splits
│   │                          
│   └── utils.py                        # Utilities for model_baseline_training.ipynb
├── src/
│   └── weather_forecasting/
│       └── model_training/             # Scripts for training saved models
├── new_README.md
├── pyproject.toml
└── requirements.txt
```

## EDA / Model Training Methodology

### Data Analysis / Feature Engineering
My EDA of `GHCND_1980_to_2026.csv` revealed a strong correlation between the daily high (TMAX) and low (TMIN) temperatures of previous days to the current day's daily high and low temperatures.

There were, of course, seasonality in the daily highs and lows has the year shifts between Spring, Summer, Fall, and Winter seaons.

To capture these insights as features for my ML models, I created lag features from TMAX/TMIN spanning the last 6 days (the maximum span NOAA's API goes back to) and generated fourier features over a period of 365 days for the current day and the last 6 days.

### Models / Baselines Evaluated
In `model_baseline_training.ipynb`, I evaluated 2 baseline models and 3 machine learning models across a train, validation, and test set split.

#### Baseline Models
- **Climatological:** Predicts a day's high and low temperatures to be the average high and low temperatures seen on that date of the year across the training set
- **Persistence:** Predicts a day's high and low temperatures to be the high and low temperatures of the previous days

#### Machine Learning Models
- **Linear Regression:** Finds a linear relationship between a day's high and low temperatures with the day's current fourier feature value and the high and low temperatures and fourier features of the last 6 days
- **XGBoost Regressor:** Uses an ensemble of decision trees to model nonlinear relationships between current day fourier features, lagged-fourier features, and lagged-temperature features and the day's high and low temperatures.
- **LSTM:** Uses an LSTM layer to learn patterns in daily high and low temperatures and fourier features over the last 6 days. The final LSTM layer's hidden output is then concatenated with the current day's fourier features to serve as inputs to a dense neural network that predicts the day's high and low temperatures

### Train / Validation / Test Split

| Split | Years |
|---|---|
| Training | 1980-2010 |
| Validation | 2011-2017 |
| Test | 2018-2025 |

### Results

This project measured model performance in terms of MAE (to get the average temperature deviation), ME (to detect tendencies to under- or over-predict temperatures), and RMSE (to understand model fit in those trained with MSE).

Each ML model substantially improved upon the climatological baseline across each metric in both high and low temperature prediction. This is likely due to the climatological models underpredicting temperatures given that training set years were likely colder than test set years. Improvements upon the persistence baseline were more modest, highlighting the predictive power of previous day temperatures on next day temperatures.

#### MAE Across Models & Baselines
![MAE metric plot](notebooks\figures\metric_plots\model_MAE_comparison.png)

#### ME Across Models & Baselines
![ME metric plot](notebooks\figures\metric_plots\model_ME_comparison.png)

#### RMSE Across Models & Baselines
![RMSE metric plot](notebooks\figures\metric_plots\model_RMSE_comparison.png)

#### Percent Improvement Against Climatological's MAE

![Climatological Baseline Plot](notebooks\figures\baseline_plots\baseline_skill_climatological.png)

#### Percent Improvement Against Persistence's MAE

![Persistence Baseline Plot](notebooks\figures\baseline_plots\baseline_skill_persistence.png)

#### Performance over 2025

![2025 Performance](notebooks\figures\prediction_plots\test_2025_model_predictions.png)

## ETL / Backend
- This section is work in progress

## Setup / Scripts
### Setting Up Local Environment
This project requires Python 3.14 or later. The recommend setup uses `uv`:
```
uv sync
```

Alternatively, you may create a venv and install used packages from `requirements.txt`:
```
python -m venv .venv
./.venv/Scripts/activate
python -m pip install -r requirements.txt
```

### Reproducing Production Training Results
This repository offers a python-based cli command for retraining production models. This won't be used during deployment, but is provided for replicability

On `uv`, run:
```
uv run training
```
Alternatively, on standard Python, run while in your venv:
```
training
```

### Running the Notebooks
To review work done during EDA and model training, you may open a jupyter lab instance.

On `uv`, run:
```
uv run jupyter lab
```

Alternatively, on standard Python, run while in your venv:
```
jupyter lab
```

Then open any of the notebooks provided in the `notebooks folder`.