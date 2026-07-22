# Stock Prediction Web Application

This project converts the existing Colab notebook into a modular FastAPI web application while preserving the original notebook pipeline:

- tvDatafeed market data fetch
- preprocessing and scaling
- sequence generation
- model training and comparison
- automatic best-model selection
- trading strategy evaluation
- walk-forward validation
- 5-day future forecasting
- Plotly visualizations

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000`.

## Colab usage

You can still call the same pipeline from Colab:

```python
from colab_runner import run_analysis
result = run_analysis(
    stock_symbol="TSLA",
    exchange="NASDAQ",
    dataset_length=1000,
    time_step=60,
)
```

`result` contains the same structured outputs used by the web app.
