import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import lightgbm as lgb
# import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(stock_ticker, start_date, end_date):
    """
    Fetches historical stock data for the given stock_ticker within the specified date range.
    """
    df = yf.download(stock_ticker, start_date, end_date)
    df.columns = df.columns.get_level_values(0).str.lower() # Convert column names to lowercase
    return df

def create_sequences(data, time_step):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def train_lstm_model(x_train, y_train):
    """
    Builds and trains an LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model

def train_gru_model(x_train, y_train):
    """
    Builds and trains a GRU model.
    """
    gru_model = Sequential()
    gru_model.add(GRU(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    gru_model.add(GRU(50, return_sequences=False))
    gru_model.add(Dense(25))
    gru_model.add(Dense(1))

    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    gru_model.fit(x_train, y_train, batch_size=1, epochs=1)
    return gru_model

def train_xgboost_model(x_train_xgb, y_train):
    """
    Initializes and trains an XGBoost Regressor model.
    """
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(x_train_xgb, y_train)
    return xgb_model

def train_random_forest_model(x_train_xgb, y_train):
    """
    Initializes and trains a RandomForestRegressor model.
    """
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(x_train_xgb, y_train)
    return rf_model

def train_linear_regression_model(x_train_xgb, y_train):
    """
    Initializes and trains a Linear Regression model.
    """
    lr_model = LinearRegression()
    lr_model.fit(x_train_xgb, y_train)
    return lr_model

def train_svr_model(x_train_xgb, y_train):
    """
    Initializes and trains an SVR model.
    """
    svr_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_model.fit(x_train_xgb, y_train)
    return svr_model

def predict_svr(svr_model, x_test_xgb, scaler):
    """
    Makes predictions using the trained SVR model and inverse scales them.
    """
    svr_predictions_scaled = svr_model.predict(x_test_xgb)
    svr_predictions = scaler.inverse_transform(svr_predictions_scaled.reshape(-1, 1))
    return svr_predictions

def train_lightgbm_model(x_train_xgb, y_train):
    """
    Initializes and trains a LightGBM Regressor model.
    """
    lgbm_model = lgb.LGBMRegressor(objective='regression', n_estimators=100, random_state=42)
    lgbm_model.fit(x_train_xgb, y_train)
    return lgbm_model

def predict_lightgbm(lgbm_model, x_test_xgb, scaler):
    """
    Makes predictions using the trained LightGBM model and inverse scales them.
    """
    lgbm_predictions_scaled = lgbm_model.predict(x_test_xgb)
    lgbm_predictions = scaler.inverse_transform(lgbm_predictions_scaled.reshape(-1, 1))
    return lgbm_predictions

# def train_arima_model(train_data):
#     """
#     Initializes and trains an ARIMA model using pmdarima's auto_arima.
#     """
#     # For ARIMA, we often use the unscaled training data or handle scaling explicitly.
#     # However, since the current setup uses scaled data for other models, 
#     # we will use the scaled training data for consistency in this utility function.
#     # The inverse transform will be handled during prediction.
#     arima_model = pm.auto_arima(train_data, 
#                                  start_p=1, start_q=1,
#                                  test='adf',       # use adftest to find optimal 'd'
#                                  max_p=3, max_q=3, # maximum p and q
#                                  m=1,              # frequency of series
#                                  d=None,           # let model determine 'd'
#                                  seasonal=False,   # No Seasonality
#                                  start_P=0, 
#                                  D=0,
#                                  trace=False,
#                                  error_action='ignore',  
#                                  suppress_warnings=True,
#                                  stepwise=True)
#     return arima_model

# def predict_arima(arima_model, test_data_unscaled, n_periods_to_predict, scaler):
#     """
#     Makes predictions using the trained ARIMA model and inverse scales them.
#     Note: test_data_unscaled is included for signature consistency but not directly used
#     by pmdarima's predict method for n_periods forecasting.
#     """
#     # ARIMA model was trained on scaled data, so its predictions will be scaled.
#     arima_predictions_scaled = arima_model.predict(n_periods=n_periods_to_predict)
#     arima_predictions = scaler.inverse_transform(arima_predictions_scaled.reshape(-1, 1))
#     return arima_predictions

def preprocess_data(df):
    """
    Scales the 'close' prices from the DataFrame using MinMaxScaler.
    """
    data = df[['close']]
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data, scaler

# def train_sarima_model(train_data):
#     """
#     Initializes and trains a SARIMA model using pmdarima's auto_arima with seasonal=True.
#     """
#     sarima_model = pm.auto_arima(train_data,
#                                  start_p=1, start_q=1,
#                                  test='adf',           # use adftest to find optimal 'd'
#                                  max_p=3, max_q=3,      # maximum p and q
#                                  m=12,                 # frequency of series (e.g., 12 for monthly data)
#                                  d=None,               # let model determine 'd'
#                                  seasonal=True,        # Enable Seasonality
#                                  start_P=0, start_Q=0,
#                                  max_P=2, max_Q=2,
#                                  D=1,                  # Order of the seasonal differencing
#                                  trace=False,
#                                  error_action='ignore',
#                                  suppress_warnings=True,
#                                  stepwise=True)
#     return sarima_model

# def predict_sarima(sarima_model, test_data_unscaled, n_periods_to_predict, scaler):
#     """
#     Makes predictions using the trained SARIMA model and inverse scales them.
#     Note: test_data_unscaled is included for signature consistency but not directly used
#     by pmdarima's predict method for n_periods forecasting.
#     """
#     # SARIMA model was trained on scaled data, so its predictions will be scaled.
#     sarima_predictions_scaled = sarima_model.predict(n_periods=n_periods_to_predict)
#     sarima_predictions = scaler.inverse_transform(sarima_predictions_scaled.reshape(-1, 1))
#     return sarima_predictions
