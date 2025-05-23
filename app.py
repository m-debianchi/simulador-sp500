import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def enrich_features(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Return'] = df['Close'].pct_change()
    return df

def run_simulator(df, forecast_days=10):
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.sort_values('Date')
    if 'Close/Last' in df.columns:
        df = df.rename(columns={'Close/Last': 'Close'})

    df = enrich_features(df)
    df_model = df[['Return', 'RSI', 'MACD', 'Signal_Line', 'DayOfWeek']].dropna()
    X = df_model[['RSI', 'MACD', 'Signal_Line', 'DayOfWeek']]
    y = df_model['Return']

    if len(X) < 40:
        st.error("Not enough data to train and test the model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=30)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    model_arima = ARIMA(y_train, order=(1, 0, 1))
    arima_fit = model_arima.fit()

    ridge_pred = ridge.predict(X_test)
    arima_pred = arima_fit.forecast(steps=30)

    ensemble = 0.5 * ridge_pred + 0.5 * arima_pred

    last_close = df['Close'].iloc[-1]
    forecast_returns = pd.Series(ensemble[-forecast_days:]).reset_index(drop=True)
    forecast_prices = [last_close * (1 + forecast_returns[0])]
    for r in forecast_returns[1:]:
        forecast_prices.append(forecast_prices[-1] * (1 + r))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(forecast_days), forecast_prices, marker='o', linestyle='-', label='Previsão de Preços')
    ax.set_title('Simulador de Previsão S&P 500 - Modelo Combinado')
    ax.set_xlabel('Dias no Futuro')
    ax.set_ylabel('Preço Estimado do Índice')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

st.title("Simulador de Previsão do S&P 500")
uploaded_file = st.file_uploader("Envie o arquivo CSV com histórico do S&P 500", type=["csv"])
forecast_days = st.slider("Dias de previsão", min_value=1, max_value=30, value=10)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    run_simulator(df, forecast_days)
