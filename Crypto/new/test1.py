# ==============================
# STEP 1: IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================
# STEP 2: LOAD DATA
# ==============================
df = pd.read_csv("sales_data.csv")   # <-- apna file name

df['SALES_DATE'] = pd.to_datetime(df['SALES_DATE'])
df = df.sort_values('SALES_DATE')
df.set_index('SALES_DATE', inplace=True)

print("Data Loaded:", df.shape)

# ==============================
# STEP 3: TARGET DEFINE
# ==============================
target = 'SALES_VALUE'

# ==============================
# STEP 4: FEATURE ENGINEERING
# ==============================
# Date features
df['day'] = df.index.day
df['month'] = df.index.month
df['week'] = df.index.isocalendar().week.astype(int)
df['day_of_week'] = df.index.dayofweek

# Lag features
df['lag_1'] = df[target].shift(1)
df['lag_2'] = df[target].shift(2)

# Drop NA
df.dropna(inplace=True)

print("After Feature Engineering:", df.shape)

# ==============================
# STEP 5: TRAIN-TEST SPLIT
# ==============================
X = df.drop(columns=[target])
y = df[target]

train_size = int(len(df) * 0.8)

X_train = X.iloc[:train_size]
X_test  = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test  = y.iloc[train_size:]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ==============================
# STEP 6: MODEL TRAINING
# ==============================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# STEP 7: EVALUATION
# ==============================
train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)

print("\n--- MODEL PERFORMANCE ---")
print("TRAIN MAE:", mean_absolute_error(y_train, train_pred))
print("TEST MAE:", mean_absolute_error(y_test, test_pred))

print("TRAIN R2:", r2_score(y_train, train_pred))
print("TEST R2:", r2_score(y_test, test_pred))

# ==============================
# STEP 8: FUTURE 90 DAYS PREDICTION
# ==============================
future_days = 90

future_dates = pd.date_range(
    start=df.index.max() + pd.Timedelta(days=1),
    periods=future_days
)

future_df = pd.DataFrame(index=future_dates)

# Date features
future_df['day'] = future_df.index.day
future_df['month'] = future_df.index.month
future_df['week'] = future_df.index.isocalendar().week.astype(int)
future_df['day_of_week'] = future_df.index.dayofweek

# SAME FEATURES AS TRAINING
feature_cols = X.columns

last_data = df.copy()
predictions = []

for i in range(future_days):
    row = future_df.iloc[i].copy()

    # Lag features
    row['lag_1'] = last_data[target].iloc[-1]
    row['lag_2'] = last_data[target].iloc[-2]

    # Missing features fill (important)
    for col in feature_cols:
        if col not in row.index:
            if col in last_data.columns:
                row[col] = last_data[col].iloc[-1]
            else:
                row[col] = 0

    # Column order match
    row = row[feature_cols]

    # Prediction
    pred = model.predict(row.values.reshape(1, -1))[0]
    predictions.append(pred)

    # Update dataset
    new_row = row.copy()
    new_row[target] = pred

    new_df = pd.DataFrame([new_row])
    new_df.index = [future_dates[i]]

    last_data = pd.concat([last_data, new_df])

print("\nNext 5 Predictions:", predictions[:5])

# ==============================
# STEP 9: VISUALIZATION
# ==============================
plt.figure(figsize=(12,5))

plt.plot(df.index, df[target], label='Actual')
plt.plot(future_dates, predictions, label='Forecast')

plt.legend()
plt.title("Sales Forecast - Next 90 Days")

plt.show()