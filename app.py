import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# --- 1. SETTING HALAMAN & STYLE ---
st.set_page_config(page_title="N219 Aero-Analytics", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #003366; }
    h1, h2, h3 { color: #003366; font-family: 'Segoe UI'; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = pickle.load(open('n219_model_v2.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_assets()

# --- 3. SIDEBAR CONTROLS (MANUAL INPUT) ---
st.sidebar.title("✈️ N219 CONTROL")
st.sidebar.markdown("---")

# Dropdown tetap untuk TRIP (karena pilihannya terbatas)
trip_val = st.sidebar.selectbox("Konfigurasi TRIP", [0, 6, 12, 17], 
                                format_func=lambda x: "NO VG" if x==0 else f"TRIP {x}")

# GANTI SLIDER MENJADI NUMBER INPUT (KETIK MANUAL)
alfa_val = st.sidebar.number_input(
    label="Sudut Serang (ALFA)", 
    min_value=-10.0, 
    max_value=25.0, 
    value=0.0, 
    step=0.1,
    format="%.2f" # Menampilkan 2 angka di belakang koma
)

beta_val = st.sidebar.number_input(
    label="Sudut Slip (BETA)", 
    min_value=-20.0, 
    max_value=20.0, 
    value=0.0, 
    step=0.1,
    format="%.2f"
)

st.sidebar.markdown("---")
st.sidebar.write("*Dibuat Oleh Veron Danda Erlangga*")

# --- 4. PREDIKSI DATA ---
def predict_data(t, a, b):
    raw = np.array([[t, a, b, a**2]])
    scaled = scaler.transform(raw)
    return model.predict(scaled)[0]

# Hitung titik saat ini
res_curr = predict_data(trip_val, alfa_val, beta_val)
c_cl, c_cd, c_cm = res_curr[0], res_curr[1], res_curr[2]
c_ld = c_cl / c_cd

# Hitung seluruh kurva ALFA (-10 s/d 25)
alfa_range = np.linspace(-15, 25, 200)
sweep_raw = np.array([[trip_val, a, beta_val, a**2] for a in alfa_range])
sweep_scaled = scaler.transform(sweep_raw)
preds = model.predict(sweep_scaled)

df_plot = pd.DataFrame(preds, columns=['CL', 'CD', 'CM25', 'CYAW', 'CROLL', 'CY'])
df_plot['ALFA'] = alfa_range
df_plot['LD'] = df_plot['CL'] / df_plot['CD']

# --- 5. TAMPILAN DASHBOARD ---
st.title("🛩️ N219 Aerodynamic Analytics")
st.markdown(f"**Konfigurasi Saat Ini:** TRIP {trip_val} | **BETA:** {beta_val}°")

# METRIK UTAMA
m1, m2, m3, m4 = st.columns(4)
m1.metric("Lift (CL)", f"{c_cl:.4f}")
m2.metric("Drag (CD)", f"{c_cd:.4f}")
m3.metric("Efficiency (L/D)", f"{c_ld:.4f}")
m4.metric("Moment (CM25)", f"{c_cm:.4f}")

st.markdown("---")

# LAYOUT GRAFIK 2X2
col_left, col_right = st.columns(2)

with col_left:
    # 1. Lift Curve
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_plot['ALFA'], y=df_plot['CL'], name="CL Curve", line=dict(color='blue', width=3)))
    fig1.add_trace(go.Scatter(x=[alfa_val], y=[c_cl], mode='markers', name="Posisi", marker=dict(size=12, color='red', symbol='cross')))
    fig1.update_layout(title="Lift Coefficient (CL vs ALFA)", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Efficiency Curve
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_plot['ALFA'], y=df_plot['LD'], name="L/D Ratio", line=dict(color='green', width=3)))
    fig2.add_trace(go.Scatter(x=[alfa_val], y=[c_ld], mode='markers', name="Posisi", marker=dict(size=12, color='red')))
    fig2.update_layout(title="Aerodynamic Efficiency (L/D vs ALFA)", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    # 3. Drag Polar
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_plot['CD'], y=df_plot['CL'], name="Drag Polar", line=dict(color='red', width=3)))
    fig3.add_trace(go.Scatter(x=[c_cd], y=[c_cl], mode='markers', name="Posisi", marker=dict(size=12, color='black')))
    fig3.update_layout(title="Drag Polar (CL vs CD)", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    # 4. Stability Curve
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df_plot['ALFA'], y=df_plot['CM25'], name="Pitching Moment", line=dict(color='purple', width=3)))
    fig4.add_hline(y=0, line_dash="dash", line_color="grey")
    fig4.update_layout(title="Longitudinal Stability (CM25 vs ALFA)", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig4, use_container_width=True)

st.write("---")
with st.expander("Klik untuk melihat Tabel Data Prediksi"):
    st.dataframe(df_plot.style.highlight_max(axis=0, subset=['LD']), use_container_width=True)
