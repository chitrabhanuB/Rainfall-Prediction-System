# ============================================================
# FILE 2 : USER INPUT PREDICTION SYSTEM (FINAL ORDER FIXED)
# ============================================================

import pickle
import numpy as np

# ==============================
# LOAD MODEL
# ==============================
with open("rainfall_model.pkl", "rb") as f:
    model = pickle.load(f)

print("🌧️ Rainfall Prediction System")
print("Enter Input Values (Dataset Order):\n")

try:
    # ==============================
    # USER INPUT (DATASET ORDER)
    # ==============================
    DOY = int(input("DOY (1–365): "))
    ALLSKY_SFC_SW_DWN = float(input("ALLSKY_SFC_SW_DWN: "))
    T2MDEW = float(input("T2MDEW: "))
    T2M_MIN = float(input("T2M_MIN: "))
    RH2M = float(input("RH2M: "))
    QV2M = float(input("QV2M: "))
    WS2M = float(input("WS2M: "))
    GWETTOP = float(input("GWETTOP: "))

    # ==============================
    # FEATURE ARRAY (MUST MATCH TRAINING ORDER)
    # ==============================
    features = np.array([[
        RH2M,
        GWETTOP,
        QV2M,
        ALLSKY_SFC_SW_DWN,
        T2MDEW,
        T2M_MIN,
        DOY,
        WS2M
    ]])

    # ==============================
    # PREDICTION
    # ==============================
    prediction = model.predict(features)
    prediction = max(prediction[0], 0)

    print("\n🌧️ Predicted Rainfall:", round(prediction, 2), "mm")

except ValueError:
    print("\n❌ Invalid input! Please enter numeric values only.")

except Exception as e:
    print("\n❌ Error:", str(e))