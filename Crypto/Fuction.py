import requests
import pandas as pd
import time

def data():
    BASE_URL = "https://api.binance.com/api/v3/klines"

    symbol = "XAUTUSDT"
    interval = "3m"
    limit = 1000
    total_candles_needed = 100000

    all_data = []
    end_time = None

    while len(all_data) < total_candles_needed:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        if end_time:
            params["endTime"] = end_time

        response = requests.get(BASE_URL, params=params)

        # ✅ Check HTTP error
        if response.status_code != 200:
            print("❌ HTTP Error:", response.status_code)
            break

        data = response.json()

        # ✅ Check Binance error response
        if isinstance(data, dict):
            print("❌ Binance Error:", data)
            break

        if not data:
            print("⚠ No more data")
            break

        all_data.extend(data)

        # ✅ Safe access
        try:
            end_time = data[0][0] - 1
        except Exception as e:
            print("❌ Data format issue:", e)
            break

        print(f"Fetched: {len(all_data)} candles")

        time.sleep(0.2)

    # Trim
    all_data = all_data[:total_candles_needed]

    if not all_data:
        print("❌ No data fetched. Check symbol.")
        return

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(all_data, columns=columns)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df.to_csv("data.csv", index=False)

    print("✅ CSV saved successfully!")

# run

import requests

def get_all_symbols():
    url = "https://api.binance.com/api/v3/exchangeInfo"

    response = requests.get(url)

    if response.status_code != 200:
        print("❌ Error:", response.status_code)
        return

    data = response.json()

    symbols = [s['symbol'] for s in data['symbols']]
    df=pd.DataFrame(symbols)
    df.to_csv("symbols.csv", index=False)

    #print(f"Total Symbols: {len(symbols)}\n")

    for sym in symbols:
        print(sym)



def create_features_and_target(df):
    df = df.copy()

    # --- Base Features ---
    df["body"] = abs(df["close"] - df["open"])
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = df["high"] - df["low"]
    df["direction"] = (df["close"] > df["open"]).astype(int)

    # --- Rolling Features ---
    df["avg_body_3"] = df["body"].rolling(3).mean()
    df["avg_body_10"] = df["body"].rolling(10).mean()
    df["volatility"] = df["range"].rolling(10).std()
    df["momentum"] = df["close"] - df["close"].shift(10)
    df["volume_avg_10"] = df["volume"].rolling(10).mean()
    df["volume_spike"] = df["volume"] / df["volume_avg_10"]

    # --- Target ---
    df["next_body"] = df["body"].shift(-1)
    df["target"] = (df["next_body"] >= 1.5 * df["avg_body_3"]).astype(int)

    # --- Drop NaN ---
    df.dropna(inplace=True)

    # --- Create X (last 10 candles flattened) ---
    feature_cols = ["body", "upper_wick", "lower_wick", "range", "direction"]

    X = []
    y = []

    for i in range(10, len(df) - 1):
        window = df.iloc[i-10:i]

        features = []
        for col in feature_cols:
            features.extend(window[col].values)

        # add derived features (current candle)
        features.extend([
            df.iloc[i]["avg_body_3"],
            df.iloc[i]["avg_body_10"],
            df.iloc[i]["volatility"],
            df.iloc[i]["momentum"],
            df.iloc[i]["volume_spike"]
        ])

        X.append(features)
        y.append(df.iloc[i]["target"])

    X = np.array(X)
    y = np.array(y)

    print("✅ Features shape:", X.shape)
    print("✅ Target shape:", y.shape)

    return X, y


df = pd.read_csv("BTCUSDT_3m_50000.csv")

X, y = create_features_and_target(df)