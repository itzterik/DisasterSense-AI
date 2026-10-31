
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="DisasterSense AI",
    page_icon="🌊",
    layout="wide"
)

# -------------------------------
# LOAD TRAINED MODELS
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    water_model = joblib.load(
        os.path.join(BASE_DIR, "water_level_model_corrected.pkl")
    )

    rain_model = joblib.load(
        os.path.join(BASE_DIR, "rainfall_model_corrected.pkl")
    )

    return water_model, rain_model

try:
    water_model, rain_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_message = str(e)

# -------------------------------
# CUSTOM DESIGN
# -------------------------------

st.markdown("""
<style>
.main {
    background-color: #0b1220;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------

st.title("🌊 DisasterSense AI")

st.markdown(
    "### Experimental River Water-Level Prediction System"
)

st.caption(
    "Location: Champua Station | Baitarani River | Kendujhar, Odisha"
)

st.divider()

st.warning(
    "Research prototype only. Predictions are experimental, "
    "not official flood warnings. Do not use them for evacuation "
    "or safety decisions."
)

# -------------------------------
# MODEL STATUS
# -------------------------------

if not models_loaded:
    st.error("Models could not be loaded.")
    st.code(error_message)
    st.info(
        "Make sure the DisasterSense_AI folder containing both "
        "model files is in the same directory as app.py."
    )
    st.stop()

st.success("Both trained machine-learning models loaded successfully.")

# -------------------------------
# INPUT SECTION
# -------------------------------

st.header("📊 Enter River Conditions")

st.write(
    "Enter water-level readings and rainfall measurements. "
    "Use actual historical measurements for meaningful experiments."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Water-Level Information")

    current_level = st.number_input(
        "Current water level (metres)",
        min_value=0.0,
        max_value=1000.0,
        value=372.0,
        step=0.01
    )

    level_1h = st.number_input(
        "Water level 1 hour ago (metres)",
        min_value=0.0,
        max_value=1000.0,
        value=371.98,
        step=0.01
    )

    level_3h = st.number_input(
        "Water level 3 hours ago (metres)",
        min_value=0.0,
        max_value=1000.0,
        value=371.95,
        step=0.01
    )

    level_6h = st.number_input(
        "Water level 6 hours ago (metres)",
        min_value=0.0,
        max_value=1000.0,
        value=371.90,
        step=0.01
    )

with col2:
    st.subheader("Rainfall Information")

    rain_1d = st.number_input(
        "Previous day's rainfall (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=5.0,
        step=0.1
    )

    rain_3d = st.number_input(
        "Previous 3 days' total rainfall (mm)",
        min_value=0.0,
        max_value=3000.0,
        value=15.0,
        step=0.1
    )

    rain_7d = st.number_input(
        "Previous 7 days' total rainfall (mm)",
        min_value=0.0,
        max_value=7000.0,
        value=30.0,
        step=0.1
    )

st.divider()

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

change_1h = current_level - level_1h
change_3h = current_level - level_3h
change_6h = current_level - level_6h

water_features = pd.DataFrame([{
    "level": current_level,
    "level_1h_ago": level_1h,
    "level_3h_ago": level_3h,
    "level_6h_ago": level_6h,
    "change_1h": change_1h,
    "change_3h": change_3h,
    "change_6h": change_6h
}])

rain_features = water_features.copy()

rain_features["rain_1d_ago"] = rain_1d
rain_features["rain_3d_total"] = rain_3d
rain_features["rain_7d_total"] = rain_7d

# Keep the exact training feature order
water_features = water_features[
    [
        "level",
        "level_1h_ago",
        "level_3h_ago",
        "level_6h_ago",
        "change_1h",
        "change_3h",
        "change_6h"
    ]
]

rain_features = rain_features[
    [
        "level",
        "level_1h_ago",
        "level_3h_ago",
        "level_6h_ago",
        "change_1h",
        "change_3h",
        "change_6h",
        "rain_1d_ago",
        "rain_3d_total",
        "rain_7d_total"
    ]
]

# -------------------------------
# PREDICTION
# -------------------------------

if st.button("🔍 Analyse River Conditions", type="primary"):

    water_probability = water_model.predict_proba(
        water_features
    )[0][1]

    rain_probability = rain_model.predict_proba(
        rain_features
    )[0][1]

    water_prediction = int(water_probability >= 0.70)
    rain_prediction = int(rain_probability >= 0.70)

    st.divider()
    st.header("🧠 Prediction Results")

    result1, result2 = st.columns(2)

    with result1:
        st.subheader("Water-Level-Only Model")

        st.metric(
            "Estimated event probability",
            f"{water_probability * 100:.2f}%"
        )

        if water_prediction:
            st.warning(
                "The model predicts a rise of at least 0.10 m "
                "over the next 3 hours."
            )
        else:
            st.success(
                "The model does not predict a rise of at least "
                "0.10 m over the next 3 hours."
            )

    with result2:
        st.subheader("Rainfall-Enhanced Model")

        st.metric(
            "Estimated event probability",
            f"{rain_probability * 100:.2f}%"
        )

        if rain_prediction:
            st.warning(
                "The model predicts a rise of at least 0.10 m "
                "over the next 3 hours."
            )
        else:
            st.success(
                "The model does not predict a rise of at least "
                "0.10 m over the next 3 hours."
            )

    st.caption(
        "These are uncalibrated model scores, not guaranteed "
        "probabilities or confidence levels."
    )

    st.divider()

    st.subheader("📈 Recent Water-Level Changes")

    chart_data = pd.DataFrame({
        "Time": ["6 hours ago", "3 hours ago", "1 hour ago", "Now"],
        "Water level (m)": [
            level_6h,
            level_3h,
            level_1h,
            current_level
        ]
    }).set_index("Time")

    st.line_chart(chart_data)

    st.subheader("🌧️ Rainfall Summary")

    rainfall_col1, rainfall_col2, rainfall_col3 = st.columns(3)

    rainfall_col1.metric("Previous day", f"{rain_1d:.1f} mm")
    rainfall_col2.metric("Previous 3 days", f"{rain_3d:.1f} mm")
    rainfall_col3.metric("Previous 7 days", f"{rain_7d:.1f} mm")

# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.markdown("""
**DisasterSense AI**

An experimental machine-learning project developed for
studying river water-level patterns at Champua station.

Models: Random Forest Classifier

Prediction horizon: 3 hours

Event definition: Water-level rise of at least 0.10 metres.

Data sources: Central Water Commission and NASA POWER.

*This project is not an operational flood forecasting or
public warning system.*
""")

# ==========================================
# MODEL PERFORMANCE ANALYSIS
# ==========================================

st.divider()
st.title("Model Performance Analysis")

st.write("Comparison of the two experimental Random Forest models.")

st.info(
    "The test period was inspected during model development. "
    "These results are not a pristine independent final evaluation."
)

metric_chart = pd.DataFrame({
    "Water-Level-Only": [0.500, 0.511, 0.505],
    "Rainfall-Enhanced": [0.597, 0.394, 0.474]
}, index=["Precision", "Recall", "F1 Score"])

st.subheader("Evaluation Metrics")
st.dataframe(metric_chart, use_container_width=True)
st.bar_chart(metric_chart)

event_chart = pd.DataFrame({
    "Water-Level-Only": [48, 46, 48],
    "Rainfall-Enhanced": [37, 57, 25]
}, index=["Correct detections", "Missed events", "False alarms"])

st.subheader("Event Detection Comparison")
st.dataframe(event_chart, use_container_width=True)
st.bar_chart(event_chart)

st.caption(
    "An event is a rise of at least 0.10 metres over three hours. "
    "This is a project-defined threshold, not an official warning threshold."
)

st.subheader("Feature Importance")

water_features = [
    "level", "level_1h_ago", "level_3h_ago", "level_6h_ago",
    "change_1h", "change_3h", "change_6h"
]

rain_features = water_features + [
    "rain_1d_ago", "rain_3d_total", "rain_7d_total"
]

water_importance = pd.DataFrame(
    {"Importance": water_model.feature_importances_},
    index=water_features
).sort_values("Importance", ascending=False)

rain_importance = pd.DataFrame(
    {"Importance": rain_model.feature_importances_},
    index=rain_features
).sort_values("Importance", ascending=False)

col1, col2 = st.columns(2)

with col1:
    st.write("Water-Level-Only Model")
    st.bar_chart(water_importance)

with col2:
    st.write("Rainfall-Enhanced Model")
    st.bar_chart(rain_importance)

st.caption(
    "Feature importance shows model reliance, not causation."
)

st.success(
    "The water-only model had higher recall. The rainfall-enhanced "
    "model had higher precision and fewer false alarms."
)


# ==========================================
# HISTORICAL DATA ANALYSIS
# ==========================================

st.divider()
st.title("Historical Water Level and Rainfall Analysis")

water_file = os.path.join(BASE_DIR, "water_level_dataset_corrected.csv")
rain_file = os.path.join(BASE_DIR, "rainfall_features.csv")

if os.path.exists(water_file) and os.path.exists(rain_file):

    historical_water = pd.read_csv(water_file)
    historical_rain = pd.read_csv(rain_file)

    historical_water["Data Acquisition Time"] = pd.to_datetime(
        historical_water["Data Acquisition Time"],
        errors="coerce"
    )

    historical_rain["date"] = pd.to_datetime(
        historical_rain["date"],
        errors="coerce"
    )

    st.write(
        "Explore historical observations from Champua station "
        "on the Baitarani River."
    )

    min_date = historical_water["Data Acquisition Time"].min().date()
    max_date = historical_water["Data Acquisition Time"].max().date()

    date_range = st.date_input(
        "Select historical date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:

        start_date, end_date = date_range

        water_filtered = historical_water[
            historical_water["Data Acquisition Time"].dt.date.between(
                start_date, end_date
            )
        ]

        rain_filtered = historical_rain[
            historical_rain["date"].dt.date.between(
                start_date, end_date
            )
        ]

        if not water_filtered.empty:

            st.subheader("River Water-Level History")

            daily_water = (
                water_filtered
                .set_index("Data Acquisition Time")["level"]
                .resample("D")
                .mean()
                .dropna()
            )

            st.line_chart(daily_water)

            c1, c2, c3 = st.columns(3)

            c1.metric("Average Level", f"{water_filtered['level'].mean():.2f} m")
            c2.metric("Maximum Level", f"{water_filtered['level'].max():.2f} m")
            c3.metric("Minimum Level", f"{water_filtered['level'].min():.2f} m")

            st.caption(
                "Water-level chart displays daily averages of available "
                "hourly observations. Missing observations may occur."
            )

        else:
            st.warning("No water-level records for the selected dates.")

        if not rain_filtered.empty:

            st.subheader("Historical Rainfall")

            daily_rain = rain_filtered.set_index("date")["rainfall"].dropna()

            st.bar_chart(daily_rain)

            c1, c2 = st.columns(2)

            c1.metric("Total Rainfall", f"{daily_rain.sum():.2f} mm")
            c2.metric("Average Daily Rainfall", f"{daily_rain.mean():.2f} mm")

            st.caption(
                "NASA POWER precipitation is reported in mm/day using "
                "Local Solar Time (LST), which differs from Indian "
                "Standard Time (IST). These dates are not exact IST-day totals."
            )

        else:
            st.warning("No rainfall records for the selected dates.")

else:
    st.info(
        "Historical datasets are not included. The prediction dashboard "
        "can still operate using manually entered values."
    )

