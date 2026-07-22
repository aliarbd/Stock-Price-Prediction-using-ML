# Stock Prediction Web Application

This project turns the original stock prediction workflow into a FastAPI web app. It can fetch market data, preprocess it, train and compare models, evaluate trading strategies, run walk-forward validation, and display forecast results with Plotly charts.

## Prerequisites

- Python 3.9+ (recommended: 3.10 or 3.11)
- pip
- Git (required for installing the TVDatafeed dependency)

## 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd thesis
```

## 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Run the application

You can start the app in either of these ways:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

or:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## 5. Health check

The app exposes a simple health endpoint:

```text
http://127.0.0.1:8000/health
```

## Optional: run the pipeline from Colab

You can also run the same analysis workflow from Colab:

```python
from colab_runner import run_analysis

result = run_analysis(
    stock_symbol="TSLA",
    exchange="NASDAQ",
    dataset_length=1000,
    time_step=60,
)
```

The returned object contains the structured outputs used by the web app.

## Troubleshooting

- If dependency installation fails, make sure Git is installed and available on your PATH.
- If the app does not start, verify that your virtual environment is activated.
- If you get a port-related error, try a different port such as 8001.
