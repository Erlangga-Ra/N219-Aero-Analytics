import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle

# 1. LOAD DATA
try:
    # Otomatis mendeteksi koma atau titik koma
    df = pd.read_csv('DATA EKSPERIMEN_FINAL.csv', sep=None, engine='python')
except Exception as e:
    print(f"Gagal membaca file: {e}")
    exit()

# 2. BERSIHKAN KOLOM
df.columns = df.columns.str.strip().str.upper()

# 3. FEATURE ENGINEERING
if 'ALFA' in df.columns:
    df['ALFA2'] = df['ALFA'] ** 2 
else:
    print("Kolom ALFA tidak ditemukan!")
    exit()

# 4. INPUT & OUTPUT
X = df[['TRIP', 'ALFA', 'BETA', 'ALFA2']] 
y = df[['CL', 'CD', 'CM25', 'CYAW', 'CROLL', 'CY']]

# 5. SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. TRAINING
model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=500, random_state=42))
model.fit(X_scaled, y)

# 7. SIMPAN
with open('n219_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Akurasi Model: {r2_score(y, model.predict(X_scaled)):.4f}")
print("🚀 Model & Scaler Berhasil Disimpan!")