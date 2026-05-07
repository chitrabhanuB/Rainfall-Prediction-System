# ============================================================
# FULL CODE : TRAINING + TESTING + MULTIPLE MODELS + GRAPHS + SAVE MODEL
# Rainfall Prediction System
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os


from io import StringIO

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================
# LOAD DATA
# ==============================
file = "Alice Springs.csv"

with open(file, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith("YEAR"):
        start = i
        break

df = pd.read_csv(StringIO("".join(lines[start:])))
df.columns = df.columns.str.strip()

print("✅ DATA LOADED")

# ==============================
# CLEAN DATA
# ==============================
df.replace(-999, np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)

df.to_csv("cleaned_dataset.csv", index=False)

# ==============================
# FEATURE ENGINEERING
# ==============================
df['DATE'] = pd.to_datetime(df['YEAR'], format='%Y') + pd.to_timedelta(df['DOY'] - 1, unit='D')
df['MONTH'] = df['DATE'].dt.month

def get_season(m):
    if m in [12,1,2]:
        return 0
    elif m in [3,4,5]:
        return 1
    elif m in [6,7,8]:
        return 2
    else:
        return 3

df['SEASON'] = df['MONTH'].apply(get_season)

# ==============================
# TRAIN TEST SPLIT (YEAR-WISE)
# ==============================
all_years = df[(df['YEAR'] >= 1995) & (df['YEAR'] <= 2020)]['YEAR'].unique()

np.random.seed(42)
np.random.shuffle(all_years)

split = int(0.8 * len(all_years))

train_years = all_years[:split]
test_years = all_years[split:]

train_df = df[df['YEAR'].isin(train_years)]
test_df = df[df['YEAR'].isin(test_years)]
future_df = df[df['YEAR'] >= 2021]

X_train = train_df.drop(['PRECTOTCORR', 'DATE'], axis=1)
y_train = train_df['PRECTOTCORR']

X_test = test_df.drop(['PRECTOTCORR', 'DATE'], axis=1)
y_test = test_df['PRECTOTCORR']

X_future = future_df.drop(['PRECTOTCORR', 'DATE'], axis=1)
y_future = future_df['PRECTOTCORR']

# ==============================
# INITIAL RANDOM FOREST (FEATURE IMPORTANCE)
# ==============================
rf_temp = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
rf_temp.fit(X_train, y_train)

print("\n🔥 FEATURE IMPORTANCE")
for name, val in zip(X_train.columns, rf_temp.feature_importances_):
    print(name, ":", round(val, 3))

# ==============================
# FEATURE SELECTION
# ==============================
important_features = [
    'RH2M', 'GWETTOP', 'QV2M',
    'ALLSKY_SFC_SW_DWN', 'T2MDEW',
    'T2M_MIN', 'DOY', 'WS2M'
]

X_train = X_train[important_features]
X_test = X_test[important_features]
X_future = X_future[important_features]

# ==============================
# MODELS
# ==============================
models = {
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=15, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
}

# ==============================
# TRAIN + PREDICT
# ==============================
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    train_pred = np.maximum(model.predict(X_train), 0)
    test_pred = np.maximum(model.predict(X_test), 0)

    results[name] = {
        "train": train_pred,
        "test": test_pred
    }

# ==============================
# EVALUATION FUNCTION
# ==============================
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, rmse, mae, mse

# ==============================
# PRINT RESULTS
# ==============================
print("\n📊 MODEL COMPARISON")

model_scores = {}

for name in models:
    train_metrics = evaluate(y_train, results[name]["train"])
    test_metrics = evaluate(y_test, results[name]["test"])

    model_scores[name] = test_metrics

    print(f"\n{name}")
    print("Train:", train_metrics)
    print("Test :", test_metrics)

# ==============================
# BEST MODEL
# ==============================
best_name = max(model_scores, key=lambda x: model_scores[x][0])
best_model = models[best_name]
best_pred = results[best_name]["test"]

print("\n🏆 BEST MODEL:", best_name)

# ==============================
# PCA
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca = PCA()
pca.fit(X_scaled)

print("\n📉 PCA Variance Top 5")
print(pca.explained_variance_ratio_[:5])

# ==============================
# FUTURE PREDICTION
# Option: sample random years from 2021-2025 for the future predictions
FUTURE_YEARS = list(range(2021, 2026))
# set SAMPLE_YEARS to an integer N to randomly sample N distinct years from FUTURE_YEARS
# set SAMPLE_YEARS = None to keep the original behaviour (all years >= 2021)
SAMPLE_YEARS = 3

# seed for reproducibility
np.random.seed(42)

if SAMPLE_YEARS is None:
    selected_future_df = future_df.copy()
    sampled_years = sorted(selected_future_df['YEAR'].unique().tolist())
else:
    # sample N distinct years from the FUTURE_YEARS pool
    sampled_years = list(np.random.choice(FUTURE_YEARS, size=min(SAMPLE_YEARS, len(FUTURE_YEARS)), replace=False))
    selected_future_df = future_df[future_df['YEAR'].isin(sampled_years)].copy()

X_future = selected_future_df.drop(['PRECTOTCORR', 'DATE'], axis=1)
y_future = selected_future_df['PRECTOTCORR']

future_pred = np.maximum(best_model.predict(X_future), 0)

results_future = selected_future_df[['YEAR', 'DOY']].copy()
results_future['Actual'] = y_future.values
results_future['Predicted'] = future_pred

print("\n📅 FUTURE PREDICTIONS (sampled years)")
print(results_future.head())

# FUTURE METRICS
r2_f = r2_score(y_future, future_pred)
rmse_f = np.sqrt(mean_squared_error(y_future, future_pred))
mae_f = mean_absolute_error(y_future, future_pred)
mse_f = mean_squared_error(y_future, future_pred)

print("\n📊 FUTURE DATA PERFORMANCE (sampled years)")
print("R2 Score :", r2_f)
print("RMSE     :", rmse_f)
print("MAE      :", mae_f)
print("MSE      :", mse_f)

# ==============================
# GRAPHS
# ==============================
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(best_pred[:100], label="Predicted")
plt.title("Actual vs Predicted Rainfall")
plt.legend()
plt.grid(True)
plt.savefig("static/graph1.png")
plt.close()

plt.figure(figsize=(12,5))
plt.plot(test_df['DATE'].values[:200], y_test.values[:200], label="Actual")
plt.plot(test_df['DATE'].values[:200], best_pred[:200], label="Predicted")
plt.title("Time Series Rainfall Prediction")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.savefig("static/graph2.png")
plt.close()

residuals = y_test.values - best_pred

plt.figure(figsize=(10,5))
plt.scatter(best_pred, residuals, alpha=0.6)
plt.axhline(y=0, linestyle='--')
plt.title("Residual Plot")
plt.grid(True)
plt.savefig("static/graph3.png")
plt.close()

# ==============================
# SAVE MODEL
# ==============================
with open("rainfall_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n✅ MODEL SAVED")


summary = {
    "best_model": best_name,
    "r2_score": r2_f,
    "rmse": rmse_f,
    "mae": mae_f
}

os.makedirs("static", exist_ok=True)

with open("static/results.json", "w") as f:
    json.dump(summary, f)

# -----------------------
# Write a richer results.json for the dashboard
# -----------------------
from datetime import datetime

# helper to convert numpy types
def _num(x):
    try:
        return float(x)
    except Exception:
        return None

# model comparison: collect metrics for each model
model_comparison = {}
for name in models:
    train_metrics = evaluate(y_train, results[name]["train"])
    test_metrics = evaluate(y_test, results[name]["test"])
    model_comparison[name] = {
        "train": {
            "r2": _num(train_metrics[0]),
            "rmse": _num(train_metrics[1]),
            "mae": _num(train_metrics[2]),
            "mse": _num(train_metrics[3])
        },
        "test": {
            "r2": _num(test_metrics[0]),
            "rmse": _num(test_metrics[1]),
            "mae": _num(test_metrics[2]),
            "mse": _num(test_metrics[3])
        }
    }

# PCA top5
pca_top5 = [ _num(x) for x in pca.explained_variance_ratio_[:5].tolist() ]

# future predictions (convert DataFrame to list of small dicts)
future_list = []
for _, row in results_future.iterrows():
    future_list.append({
        "year": int(row['YEAR']),
        "doy": int(row['DOY']),
        "actual": _num(row['Actual']),
        "predicted": _num(row['Predicted'])
    })

results_payload = {
    "timestamp": datetime.utcnow().isoformat() + 'Z',
    "best_model": best_name,
    "metrics_future": {
        "r2": _num(r2_f),
        "rmse": _num(rmse_f),
        "mae": _num(mae_f),
        "mse": _num(mse_f)
    },
    "model_comparison": model_comparison,
    "pca_top5": pca_top5,
    "future_predictions": future_list,
    "graphs": ["static/graph1.png", "static/graph2.png", "static/graph3.png"]
}

results_payload['sampled_years'] = sampled_years

with open("static/results.json", "w") as f:
    json.dump(results_payload, f, indent=2)