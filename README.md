# Weather Forecasting

This project explores and compares statistical and machine-learning approaches for forecasting daily minimum and maximum temperatures at Tampa International Airport.

The work is organized in two Jupyter notebooks:

- `notebooks/data_analysis.ipynb`: examines the NOAA GHCND data, NOAA Weather.gov observations, and OpenWeather observations.
- `notebooks/model_baseline_training.ipynb`: creates baselines and trains and evaluates temperature forecasting models.

The reusable data-processing and model utilities used in the notebooks are in `notebooks/utils.py`. The package entry point in `src/model_training/` trains the models and saves their artifacts.

## Data

Raw data is stored in `data/raw/`:

- `GHCND_1980_to_2026.csv` - daily NOAA GHCND observations for Tampa International Airport, primarily containing the `TMAX` and `TMIN` targets.
- `weather_gov_data.csv` - observations collected from the NOAA Weather.gov API.
- `open_weather_data.csv` - recent forecast or observation data collected from the OpenWeather API.

The model pipeline uses daily GHCND temperatures from 1980 through 2025. Missing temperatures are time interpolated. Temperature values are normalized using the training-period mean and standard deviation.

The chronological splits used for experimentation are:

| Split | Years |
|---|---|
| Training | 1980-2010 |
| Validation | 2011-2017 |
| Test | 2018-2025 |

This prevents future observations from leaking into model training.

## Features

The models use a combination of:

- Previous six days of `TMAX`
- Previous six days of `TMIN`
- Calendar day-of-year
- Sine and cosine Fourier calendar features
- Lagged Fourier features for the linear and tree-based models

The RNN/LSTM pipeline uses six-day sequences containing temperature and Fourier features.

## Models

The project compares the following approaches:

| Model | Description |
|---|---|
| Persistence baseline | Predicts the next day to have the previous day's temperatures |
| Climatological baseline | Predicts the mean temperature for each day of the year |
| Linear regression | Uses lagged temperatures and seasonal Fourier features |
| XGBoost | Uses lagged temperatures, day-of-year, and Fourier features |
| LSTM | Uses a six-day sequence with recurrent and seasonal inputs |

Models are evaluated using mean absolute error (MAE), root mean squared error (RMSE), and mean error (ME), reported separately for `TMAX` and `TMIN`.

## Saved Artifacts

The `saved_models/` directory contains:

- `climatological_model.csv`
- `linear_model.joblib`
- `xgbregressor_model.json`
- `lstm_model.pt`
- `weather_linear_model.artifact.joblib`
- `metadata.json`, including temperature normalization values

## Setup

This project requires Python 3.14 or later. The recommended setup uses `uv`:

```powershell
uv sync
```

Alternatively, install the packages from `requirements.txt`:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproducing Training

Train all saved models from the project root:

```powershell
uv run model_training
```

The command reads `data/raw/GHCND_1980_to_2026.csv`, processes the data, and writes model files and normalization metadata to `saved_models/`.

## Running the Notebooks

Start Jupyter Lab:

```powershell
uv run jupyter lab
```

Then open:

1. `notebooks/data_analysis.ipynb`
2. `notebooks/model_baseline_training.ipynb`

Keep the repository root as the working directory when running notebook cells that use relative paths such as `../data/raw/...`.