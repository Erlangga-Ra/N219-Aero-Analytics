"""
N219 Aero Analytics - Streamlit Application

Features:
- Prediction using the final trained model
- Experimental-domain validation
- Extrapolation warning
- Experimental vs ML curves
- Drag polar
- CL/CD analysis
- In-domain optimization
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "n219_model_final.pkl"
DATA_FILE = BASE_DIR / "DATA EKSPERIMEN_FINAL.csv"
BOUNDARY_FILE = BASE_DIR / "experimental_boundaries.csv"
METRICS_FILE = BASE_DIR / "model_metrics.csv"

st.set_page_config(
    page_title="N219 Aero Analytics",
    page_icon="✈️",
    layout="wide",
)

TARGETS = ["CL", "CD", "CM25", "CYAW", "CROLL", "CY"]


@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, sep=";", engine="python")
    df.columns = df.columns.astype(str).str.strip().str.upper()
    df = df.loc[:, ~df.columns.str.startswith("UNNAMED")]

    for col in ["TRIP", "ALFA", "BETA"] + TARGETS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "MODE" in df.columns:
        df["MODE"] = df["MODE"].astype(str).str.strip().str.upper()

    return df.dropna(subset=["TRIP", "MODE", "ALFA", "BETA"] + TARGETS)


@st.cache_data
def load_boundaries():
    if not BOUNDARY_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(BOUNDARY_FILE)


@st.cache_data
def load_metrics():
    if not METRICS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_FILE)


def get_boundary(boundaries, mode, trip):
    if boundaries.empty:
        return None
    row = boundaries[
        (boundaries["MODE"] == mode) &
        (boundaries["TRIP"] == trip)
    ]
    return None if row.empty else row.iloc[0]


def validate_input(boundaries, mode, trip, alfa, beta):
    b = get_boundary(boundaries, mode, trip)
    if b is None:
        return False, "Boundary eksperimen untuk kombinasi ini belum tersedia."

    issues = []

    if not (b["ALFA_MIN"] <= alfa <= b["ALFA_MAX"]):
        issues.append(
            f"ALFA di luar {b['ALFA_MIN']:.2f}° s.d. {b['ALFA_MAX']:.2f}°"
        )

    if not (b["BETA_MIN"] <= beta <= b["BETA_MAX"]):
        issues.append(
            f"BETA di luar {b['BETA_MIN']:.2f}° s.d. {b['BETA_MAX']:.2f}°"
        )

    if issues:
        return False, " | ".join(issues)

    return True, "Input berada dalam rentang eksperimen."


def predict(bundle, trip, mode, alfa, beta):
    X = pd.DataFrame([{
        "TRIP": trip,
        "MODE": mode,
        "ALFA": alfa,
        "BETA": beta,
    }])
    pred = bundle["model"].predict(X)[0]
    return dict(zip(bundle["targets"], pred))


def make_alpha_curve(bundle, data, trip):
    exp = data[
        (data["MODE"] == "ALFA") &
        (data["TRIP"] == trip) &
        (np.isclose(data["BETA"], 0))
    ].copy()

    if exp.empty:
        return None

    amin, amax = exp["ALFA"].min(), exp["ALFA"].max()
    grid = np.linspace(amin, amax, 300)

    X = pd.DataFrame({
        "TRIP": trip,
        "MODE": "ALFA",
        "ALFA": grid,
        "BETA": 0.0,
    })
    pred = bundle["model"].predict(X)

    ml = pd.DataFrame({
        "ALFA": grid,
        "CL": pred[:, 0],
        "CD": pred[:, 1],
    })

    return exp, ml


def make_beta_curve(bundle, data, trip):
    exp = data[
        (data["MODE"] == "BETA") &
        (data["TRIP"] == trip) &
        (np.isclose(data["ALFA"], 0))
    ].copy()

    if exp.empty:
        return None

    bmin, bmax = exp["BETA"].min(), exp["BETA"].max()
    grid = np.linspace(bmin, bmax, 300)

    X = pd.DataFrame({
        "TRIP": trip,
        "MODE": "BETA",
        "ALFA": 0.0,
        "BETA": grid,
    })
    pred = bundle["model"].predict(X)

    ml = pd.DataFrame({
        "BETA": grid,
        "CL": pred[:, 0],
        "CD": pred[:, 1],
    })

    return exp, ml


def optimize_in_domain(bundle, data, mode, trip):
    if mode == "ALFA":
        exp = data[
            (data["MODE"] == "ALFA") &
            (data["TRIP"] == trip) &
            (np.isclose(data["BETA"], 0))
        ].copy()

        if exp.empty:
            return None

        grid = np.linspace(exp["ALFA"].min(), exp["ALFA"].max(), 2001)
        X = pd.DataFrame({
            "TRIP": trip,
            "MODE": "ALFA",
            "ALFA": grid,
            "BETA": 0.0,
        })
        fixed = "ALFA"
    else:
        exp = data[
            (data["MODE"] == "BETA") &
            (data["TRIP"] == trip) &
            (np.isclose(data["ALFA"], 0))
        ].copy()

        if exp.empty:
            return None

        grid = np.linspace(exp["BETA"].min(), exp["BETA"].max(), 2001)
        X = pd.DataFrame({
            "TRIP": trip,
            "MODE": "BETA",
            "ALFA": 0.0,
            "BETA": grid,
        })
        fixed = "BETA"

    pred = bundle["model"].predict(X)
    cl = pred[:, 0]
    cd = pred[:, 1]

    # Hindari pembagian dengan CD <= 0.
    ld = np.where(cd > 0, cl / cd, np.nan)
    idx = np.nanargmax(ld)

    ml_value = float(grid[idx])
    ml_cl = float(cl[idx])
    ml_cd = float(cd[idx])
    ml_ld = float(ld[idx])

    exp_ld = exp["CL"] / exp["CD"]
    exp_idx = exp_ld.idxmax()

    return {
        "MODE": mode,
        "TRIP": trip,
        "VARIABLE": fixed,
        "ML_OPT": ml_value,
        "ML_CL": ml_cl,
        "ML_CD": ml_cd,
        "ML_CL_CD": ml_ld,
        "EXP_OPT": float(exp.loc[exp_idx, fixed]),
        "EXP_CL": float(exp.loc[exp_idx, "CL"]),
        "EXP_CD": float(exp.loc[exp_idx, "CD"]),
        "EXP_CL_CD": float(exp_ld.loc[exp_idx]),
        "BOUNDARY": (
            ml_value <= grid.min() + 1e-9 or
            ml_value >= grid.max() - 1e-9
        ),
    }


# ------------------------- UI -------------------------

st.title("✈️ N219 Aero Analytics")
st.caption("Machine Learning Analysis untuk Data Aerodinamika N219")

if not MODEL_FILE.exists():
    st.error(
        "Model final belum ditemukan. Jalankan train.py terlebih dahulu "
        "untuk menghasilkan n219_model_final.pkl."
    )
    st.stop()

bundle = load_model()
data = load_data()
boundaries = load_boundaries()
metrics = load_metrics()

with st.sidebar:
    st.header("Input Parameter")

    mode = st.selectbox("Mode Eksperimen", ["ALFA", "BETA"])
    trip = st.selectbox("TRIP", [0, 6, 12, 17])

    b = get_boundary(boundaries, mode, trip)

    if mode == "ALFA":
        alfa_min = float(b["ALFA_MIN"]) if b is not None else -4.0
        alfa_max = float(b["ALFA_MAX"]) if b is not None else 18.5
        alfa = st.slider(
            "ALFA (°)",
            min_value=float(np.floor(alfa_min)),
            max_value=float(np.ceil(alfa_max)),
            value=0.0,
            step=0.1,
        )
        beta = st.number_input("BETA (°)", value=0.0, step=0.1)
    else:
        alfa = st.number_input("ALFA (°)", value=0.0, step=0.1)
        beta_min = float(b["BETA_MIN"]) if b is not None else -10.0
        beta_max = float(b["BETA_MAX"]) if b is not None else 10.0
        beta = st.slider(
            "BETA (°)",
            min_value=float(np.floor(beta_min)),
            max_value=float(np.ceil(beta_max)),
            value=0.0,
            step=0.1,
        )

    st.divider()
    st.write(f"**Model:** {bundle['model_name']}")

    if b is not None:
        st.caption(
            f"Domain {mode}: "
            f"ALFA {b['ALFA_MIN']:.2f}–{b['ALFA_MAX']:.2f}°, "
            f"BETA {b['BETA_MIN']:.2f}–{b['BETA_MAX']:.2f}°"
        )

valid, message = validate_input(boundaries, mode, trip, alfa, beta)

prediction = predict(bundle, trip, mode, alfa, beta)

if valid:
    st.success("🟢 " + message)
else:
    st.warning(
        "🟠 EXTRAPOLATION / OUT-OF-DOMAIN\n\n" + message +
        "\n\nHasil prediksi di luar domain eksperimen harus divalidasi "
        "lebih lanjut dan tidak diperlakukan sebagai data eksperimen."
    )

st.subheader("Prediction")

cols = st.columns(6)
for col, target in zip(cols, TARGETS):
    col.metric(target, f"{prediction[target]:.6f}")

cl_cd = (
    prediction["CL"] / prediction["CD"]
    if prediction["CD"] > 0 else np.nan
)

st.metric("CL / CD (L/D)", f"{cl_cd:.4f}")

# ------------------------- Graph -------------------------

st.subheader("Experimental vs ML")

curve = (
    make_alpha_curve(bundle, data, trip)
    if mode == "ALFA"
    else make_beta_curve(bundle, data, trip)
)

if curve is not None:
    exp, ml = curve
    variable = "ALFA" if mode == "ALFA" else "BETA"

    fig_cl = go.Figure()
    fig_cl.add_trace(go.Scatter(
        x=exp[variable],
        y=exp["CL"],
        mode="markers",
        name="Eksperimen CL",
    ))
    fig_cl.add_trace(go.Scatter(
        x=ml[variable],
        y=ml["CL"],
        mode="lines",
        name="ML CL",
    ))
    fig_cl.update_layout(
        title=f"CL vs {variable} — TRIP {trip}",
        xaxis_title=f"{variable} (°)",
        yaxis_title="CL",
    )
    st.plotly_chart(fig_cl, use_container_width=True)

    fig_cd = go.Figure()
    fig_cd.add_trace(go.Scatter(
        x=exp[variable],
        y=exp["CD"],
        mode="markers",
        name="Eksperimen CD",
    ))
    fig_cd.add_trace(go.Scatter(
        x=ml[variable],
        y=ml["CD"],
        mode="lines",
        name="ML CD",
    ))
    fig_cd.update_layout(
        title=f"CD vs {variable} — TRIP {trip}",
        xaxis_title=f"{variable} (°)",
        yaxis_title="CD",
    )
    st.plotly_chart(fig_cd, use_container_width=True)

    # Drag polar
    fig_polar = go.Figure()
    fig_polar.add_trace(go.Scatter(
        x=exp["CD"],
        y=exp["CL"],
        mode="markers",
        name="Eksperimen",
    ))
    fig_polar.add_trace(go.Scatter(
        x=ml["CD"],
        y=ml["CL"],
        mode="lines",
        name="ML",
    ))
    fig_polar.update_layout(
        title=f"Drag Polar — TRIP {trip}, MODE {mode}",
        xaxis_title="CD",
        yaxis_title="CL",
    )
    st.plotly_chart(fig_polar, use_container_width=True)

# ------------------------- Optimization -------------------------

st.subheader("Optimasi CL/CD Dalam Domain Eksperimen")

results = []
for m in ["ALFA", "BETA"]:
    for t in [0, 6, 12, 17]:
        result = optimize_in_domain(bundle, data, m, t)
        if result:
            results.append(result)

opt_df = pd.DataFrame(results)

if not opt_df.empty:
    display_cols = [
        "MODE", "TRIP", "VARIABLE",
        "ML_OPT", "ML_CL", "ML_CD", "ML_CL_CD",
        "EXP_OPT", "EXP_CL", "EXP_CD", "EXP_CL_CD",
        "BOUNDARY",
    ]
    st.dataframe(
        opt_df[display_cols].round(5),
        use_container_width=True,
        hide_index=True,
    )

    selected = opt_df[
        (opt_df["MODE"] == mode) &
        (opt_df["TRIP"] == trip)
    ]

    if not selected.empty:
        r = selected.iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric(
            f"Optimal {r['VARIABLE']}",
            f"{r['ML_OPT']:.2f}°",
        )
        c2.metric(
            "Predicted CL/CD",
            f"{r['ML_CL_CD']:.3f}",
        )
        c3.metric(
            "Experimental CL/CD",
            f"{r['EXP_CL_CD']:.3f}",
        )

        if r["BOUNDARY"]:
            st.info(
                "Titik optimum ML berada pada batas domain eksperimen. "
                "Artinya hasil ini adalah optimum dalam rentang data yang "
                "tersedia, bukan optimum absolut di luar rentang tersebut."
            )

# ------------------------- Model metrics -------------------------

st.subheader("Model Validation")

if not metrics.empty:
    st.dataframe(
        metrics[
            ["Model", "R2_Overall", "RMSE_Overall", "MAE_Overall"]
        ].round(6),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Catatan: model ML digunakan untuk estimasi dan pencarian kandidat "
    "optimum dalam domain data eksperimen. Validasi akhir kandidat optimum "
    "dilakukan menggunakan simulasi/eksperimen lanjutan, misalnya ANSYS."
)
