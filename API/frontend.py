import streamlit as st
import requests
import pandas as pd
from datetime import date

st.set_page_config(
    page_title="Weekly Sales Demand Forecast",
    page_icon="📦",
    layout="centered"
)

API_URL = "http://localhost:8000"
CATEGORIES = ["long sleeves", "short sleeves", "lowers"]

st.title("📦 Weekly Sales Demand Forecast")
st.caption("XGBoost-based demand forecasting and inventory recommendation")

with st.sidebar:
    st.subheader("API Settings")
    api_url = st.text_input("API Base URL", value=API_URL)
    st.caption("Point this at your running FastAPI server")

st.subheader("Input")

category = st.selectbox("Category", CATEGORIES)
forecast_date = st.date_input("Forecast Date", value=date.today())

st.write("Enter sales for the previous 7 weeks (week 1 = oldest, week 7 = most recent)")

col1, col2 = st.columns(2)

with col1:
    week_1 = st.number_input("Week 1", min_value=0.0, value=0.0, step=1.0)
    week_2 = st.number_input("Week 2", min_value=0.0, value=0.0, step=1.0)
    week_3 = st.number_input("Week 3", min_value=0.0, value=0.0, step=1.0)
    week_4 = st.number_input("Week 4", min_value=0.0, value=0.0, step=1.0)

with col2:
    week_5 = st.number_input("Week 5", min_value=0.0, value=0.0, step=1.0)
    week_6 = st.number_input("Week 6", min_value=0.0, value=0.0, step=1.0)
    week_7 = st.number_input("Week 7", min_value=0.0, value=0.0, step=1.0)

weekly_sales = [week_1, week_2, week_3, week_4, week_5, week_6, week_7]

# quick visual sanity check of what's being sent, before hitting the API
with st.expander("Preview last 7 weeks"):
    st.bar_chart(
        pd.DataFrame({"sales": weekly_sales}, index=[
                     f"Week {i + 1}" for i in range(7)])
    )

st.divider()

if st.button("Run Forecast", type="primary"):

    payload = {
        "category": category,
        "forecast_date": forecast_date.isoformat(),
        "week_1_sales": week_1,
        "week_2_sales": week_2,
        "week_3_sales": week_3,
        "week_4_sales": week_4,
        "week_5_sales": week_5,
        "week_6_sales": week_6,
        "week_7_sales": week_7
    }

    try:
        response = requests.post(
            f"{api_url}/predict", json=payload, timeout=30)
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to {api_url}. Is the API running?")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("Request timed out — the API may be stuck fetching Google Trends.")
        st.stop()

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"API Error ({response.status_code}): {detail}")
        st.stop()

    result = response.json()

    # predicted_demand / google_trends live under "prediction",
    # everything else is top-level
    prediction = result.get("prediction", {})
    inventory = result.get("inventory", {})
    model_features = result.get("model_features", {})

    st.success("Forecast complete")

    st.subheader("Demand Prediction")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Demand", prediction.get("predicted_demand", "—"))
    col2.metric("Google Trends", prediction.get("google_trends", "—"))

    st.subheader("Inventory Recommendation")
    col1, col2 = st.columns(2)
    col1.metric("Safety Stock", inventory.get("safety_stock", "—"))
    col2.metric("Reorder Point", inventory.get("reorder_point", "—"))

    st.metric("Tail Risk", inventory.get("tail_risk", "—"))

    with st.expander("Model Features"):
        st.json(model_features)

    with st.expander("Raw Response"):
        st.json(result)
