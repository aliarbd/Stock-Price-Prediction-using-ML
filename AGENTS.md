# Stock Prediction Lab — AI Agent Guide

This repository is a stock prediction web application that combines historical market data ingestion, preprocessing, model training, strategy evaluation, walk-forward validation, and future forecasting in one pipeline. The app is built around a FastAPI backend and a Plotly-based dashboard UI.

## 1. Project Purpose

The project aims to:
- fetch historical stock data,
- preprocess it for time-series modeling,
- train and compare multiple predictive models,
- evaluate a simple trading strategy,
- run walk-forward validation,
- generate future-price forecasts,
- display results in a browser-based dashboard.

It is designed as both:
- a web application, and
- a reusable Python pipeline for analysis.

## 2. Main Tech Stack

- Python
- FastAPI for the web API
- Jinja2 + HTML/CSS/JavaScript for the UI
- Plotly for interactive charts
- pandas, numpy for data handling
- scikit-learn for classical ML models
- TensorFlow/Keras for LSTM/GRU
- XGBoost and LightGBM for tree-based models
- statsmodels for ARIMA/SARIMA
- tvDatafeed for fetching historical market data

## 3. Repository Structure

```text
thesis/
├── app.py                     # FastAPI entry point; serves UI and API endpoints
├── bind_test.py              # Small test/experiment script
├── colab_file.ipynb          # Notebook-based prototype or experiment file
├── colab_runner.py           # Helper to run the analysis workflow from Colab
├── README.md                 # Setup and run instructions
├── requirements.txt          # Python dependencies
├── utils.py                  # Utility helpers (if used by notebooks/scripts)
├── static/                   # Static frontend assets
│   ├── app.js                # Frontend logic for calling the API and rendering results
│   ├── styles.css            # UI styling
│   └── vendor/               # Third-party JS/CSS assets
├── templates/
│   └── index.html            # Main dashboard page
└── webapp/                   # Core backend package
    ├── __init__.py           # Package export file
    ├── config.py            # Model list, config dataclass, pipeline constants
    ├── data.py              # Data loading, preprocessing, sequence creation
    ├── future.py            # Future forecast generation
    ├── metrics.py           # Regression and direction-based metrics
    ├── models.py            # Model construction and random-seed setup
    ├── pipeline.py          # Main end-to-end analysis pipeline
    ├── strategy.py          # Strategy signal generation and metrics
    ├── visuals.py           # Plotly chart creation
    ├── walk_forward.py      # Walk-forward validation logic
    └── static/               # Optional app-specific static assets
        └── templates/        # Optional nested templates
```

## 4. How the Project Works

### A. User Interaction
1. The user opens the web dashboard at the root URL.
2. The frontend form collects configuration values such as:
   - stock symbol
   - exchange
   - dataset length
   - train split
   - time step
   - forecast window
   - walk-forward window
   - selected models
3. The browser sends a POST request to the API endpoint `/api/runs`.

### B. Backend Pipeline
1. The FastAPI app in [app.py](app.py) creates a job and submits it to a background executor.
2. The job calls [webapp/pipeline.py](webapp/pipeline.py).
3. The pipeline performs the full workflow:
   - load historical data,
   - preprocess data,
   - create time-series sequences,
   - train selected models,
   - evaluate model performance,
   - build strategy signals,
   - run walk-forward validation,
   - generate future forecasts,
   - package all outputs for the UI.

### C. Output to UI
The backend returns a result object containing:
- dataset summary
- comparison chart and table
- best model selection
- trading strategy chart and metrics
- walk-forward validation chart and table
- future prediction chart and table

The frontend in [static/app.js](static/app.js) renders these results in the dashboard.

## 5. Core Files and Their Roles

### app.py
Entry point for the web app.
Responsibilities:
- define FastAPI app
- serve the main HTML page
- expose API routes for running analysis and checking job status
- manage background job execution

### webapp/config.py
Contains:
- `MODEL_ORDER`: default model ordering
- model category sets such as `RNN_MODELS`, `TREE_MODELS`, `STATISTICAL_MODELS`
- `PipelineConfig`: dataclass holding all user-configurable settings

### webapp/data.py
Handles:
- loading historical data from `tvDatafeed`
- preprocessing into a close-price dataframe
- scaling values with `MinMaxScaler`
- creating sequences for supervised learning

### webapp/models.py
Defines:
- random seed handling
- LSTM and GRU model construction
- tree-based model initialization

### webapp/pipeline.py
The main orchestration file.
Responsibilities:
- run the full experiment pipeline
- train each selected model
- compute regression metrics
- build comparison visuals
- create strategy analysis outputs
- run walk-forward validation
- generate future forecasts
- return a structured result dictionary

### webapp/strategy.py
Builds the trading strategy logic.
Responsibilities:
- compute EMA-based filtering
- generate buy/sell signals
- produce trade-level results
- calculate strategy performance metrics like Sharpe ratio and drawdown

### webapp/walk_forward.py
Performs rolling validation.
Responsibilities:
- simulate predictions over a future validation window
- calculate forecasting metrics for each model

### webapp/future.py
Generates a forward-looking prediction range.
Responsibilities:
- simulate multi-step outputs for future days
- create a forecast dataframe for the UI

### webapp/visuals.py
Creates all Plotly charts:
- prediction comparison figure
- strategy signal figure
- equity curve figure
- walk-forward figure
- future forecast figure

### webapp/metrics.py
Contains metrics functions for:
- MSE
- MAE
- RMSE
- MAPE
- R²
- directional accuracy

### templates/index.html
The main frontend page.
It contains:
- configuration form
- progress bar area
- dataset summary area
- result panels for plots and tables

### static/app.js
Frontend logic.
Responsibilities:
- collect form data
- send requests to the backend
- poll job status
- render charts and tables
- handle download actions

## 6. Data Flow Summary

```text
User input -> FastAPI route -> PipelineConfig -> run_pipeline() ->
load_data() -> preprocess_data() -> create_sequences() ->
train selected models -> compute metrics ->
strategy analysis -> walk-forward validation -> future forecast ->
JSON/Plotly result -> frontend dashboard
```

## 7. How to Run the Project

### Create environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Start the app
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Or:
```bash
python app.py
```

Open:
```text
http://127.0.0.1:8000
```

## 8. Important Notes for Future Changes

- The backend expects a structured result dictionary from `run_pipeline()`.
- The frontend depends on specific keys such as `dataset_summary`, `comparison`, `strategy`, `walk_forward`, and `future_prediction`.
- If you add or rename output fields, update the frontend rendering logic in [static/app.js](static/app.js).
- If you change model names, make sure they are consistent across:
  - [webapp/config.py](webapp/config.py)
  - [webapp/models.py](webapp/models.py)
  - [webapp/visuals.py](webapp/visuals.py)
  - [webapp/pipeline.py](webapp/pipeline.py)
- Data fetching depends on `tvDatafeed`, so internet access and Git availability are required.

## 9. Suggested Mental Model for AI Agents

If you are modifying this project, think of it as a pipeline with three layers:
1. Input layer: configuration and market data
2. Modeling layer: training, validation, forecasting
3. Presentation layer: charts, tables, metrics, and UI rendering

Most changes should be made at the correct layer only:
- data or pipeline logic -> [webapp/pipeline.py](webapp/pipeline.py) or [webapp/data.py](webapp/data.py)
- model definitions -> [webapp/models.py](webapp/models.py)
- visual output -> [webapp/visuals.py](webapp/visuals.py)
- API/UI behavior -> [app.py](app.py) and [static/app.js](static/app.js)
