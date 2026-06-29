import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

def predict_suzlon():
    ticker = "SUZLON.NS"
    df = yf.Ticker(ticker).history(period="5y")

    if df.empty:
        return None

    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    last_day = len(df)
    future_days = [last_day + 365, last_day + 365*2, last_day + 365*3, last_day + 365*5]
    years = [2026, 2027, 2028, 2030]

    preds = model.predict(np.array(future_days).reshape(-1, 1))

    return {"current": y.iloc[-1], "predictions": dict(zip(years, preds))}