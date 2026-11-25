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

### Dataset


### File Structure

### Roadmap

## EDA / Model Training Methodology

### Data Preparation

### Models / Baselines Evaluated

### Train / Validation / Test Split

## EDA / Model Training Results

## ETL / Backend
- This section is work in progress

## Setup / Scripts
This project requires Python 3.14 or later. The recommend setup uses `uv`:
```
uv sync
```

Alternatively, install the packages from `requirements.txt`:
```
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

### Reproducing Training
Train all saved models from the project root:
uv run training
The command reads

### Running the Notebooks
Start jupyter lab:
uv run jupyter lab

Then open:
1. Notebook 1
2. Notebook 2