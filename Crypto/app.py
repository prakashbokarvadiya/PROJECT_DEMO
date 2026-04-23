#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

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import os
import threading

from flask import Flask, jsonify, render_template, request
from groq import Groq
import websocket

# =========================
# CONFIG
# =========================
SYMBOL = "btcusdt"
INTERVAL = "3m"

ist = pytz.timezone('Asia/Kolkata')
app = Flask(__name__)

# 🔐 API KEY (set in env)
client = Groq(api_key="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")

# =========================
# GLOBAL DATA STORE
# =========================
candles = []

# =========================
# WEBSOCKET DATA
# =========================
def on_message(ws, message):
    global candles

    data = json.loads(message)

    if "k" in data:
        k = data["k"]

        if k["x"]:  # candle closed
            candle = {
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"])
            }

            candles.append(candle)

            # keep last 50 candles
            if len(candles) > 50:
                candles = candles[-50:]

            print("📊 Candle:", candle)

def start_ws():
    socket = f"wss://stream.binance.com:9443/ws/{SYMBOL}@kline_{INTERVAL}"

    ws = websocket.WebSocketApp(socket, on_message=on_message)
    ws.run_forever()

# =========================
# LOAD MODELS
# =========================
model_lgb = joblib.load("lgb_model.pkl")
model_xgb = joblib.load("xgb_model.pkl")

try:
    scaler = joblib.load("scaler.pkl")
    USE_SCALER = True
except:
    USE_SCALER = False

with open("config.json") as f:
    config = json.load(f)

threshold = config.get("threshold", 0.4)

feature_cols = ["body", "upper_wick", "lower_wick", "range", "direction"]

# =========================
# FEATURE ENGINEERING
# =========================
def build_features(df):
    try:
        df["body"] = abs(df["close"] - df["open"])
        df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
        df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]
        df["range"] = df["high"] - df["low"]
        df["direction"] = (df["close"] > df["open"]).astype(int)

        df["avg_body_3"] = df["body"].rolling(3).mean()
        df["avg_body_10"] = df["body"].rolling(10).mean()
        df["volatility"] = df["range"].rolling(10).std()
        df["momentum"] = df["close"] - df["close"].shift(10)
        df["volume_avg_10"] = df["volume"].rolling(10).mean()
        df["volume_spike"] = df["volume"] / df["volume_avg_10"]

        df = df.dropna()

        if len(df) < 10:
            return None

        window = df.iloc[-10:]

        features = []
        for j in range(10):
            for col in feature_cols:
                features.append(window.iloc[j][col])

        last = df.iloc[-1]

        features.extend([
            last["avg_body_3"],
            last["avg_body_10"],
            last["volatility"],
            last["momentum"],
            last["volume_spike"]
        ])

        return np.array(features).reshape(1, -1)

    except Exception as e:
        print("FEATURE ERROR:", e)
        return None

# =========================
# PREDICTION
# =========================
def get_prediction():
    global candles

    if len(candles) < 20:
        return None

    df = pd.DataFrame(candles)

    X = build_features(df)
    if X is None:
        return None

    if USE_SCALER:
        X = scaler.transform(X)

    lgb_p = model_lgb.predict_proba(X)[0][1]
    xgb_p = model_xgb.predict_proba(X)[0][1]

    prob = (lgb_p + xgb_p) / 2
    label = "BIG 🔥" if prob > threshold else "SMALL ❄"

    return {
        "symbol": SYMBOL.upper(),
        "time": datetime.now(ist).strftime("%H:%M:%S"),
        "label": label,
        "prob": float(prob),
        "lgb": float(lgb_p),
        "xgb": float(xgb_p)
    }

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict_api():
    result = get_prediction()

    if result is None:
        return jsonify({"error": "Waiting for data..."}), 500

    return jsonify(result)

@app.route("/chat", methods=["POST"])
def chat():
    prediction_data = get_prediction()

    if prediction_data is None:
        return {"reply": "⚠️ Data abhi load ho raha hai"}

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": f"""
Explain this trading prediction in simple Hinglish:

Symbol: {prediction_data['symbol']}
Prediction: {prediction_data['label']}
Confidence: {prediction_data['prob']}
"""
            }
        ]
    )

    reply = completion.choices[0].message.content
    return {"reply": reply}

# =========================
# START
# =========================
if __name__ == "__main__":
    threading.Thread(target=start_ws, daemon=True).start()
    app.run(debug=True)