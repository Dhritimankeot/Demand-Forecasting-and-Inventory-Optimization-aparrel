from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, computed_field
from typing import Literal
from datetime import date, datetime, timedelta, timezone

import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import time

from pytrends.request import TrendReq


app = FastAPI(
    title="Weekly Sales Demand Forecasting API",
    description="XGBoost based weekly sales demand and inventory prediction",
    version="1.0.0"
)

# load the trained model + category encoder once at startup
model = xgb.XGBRegressor()
model.load_model("xgboost_model.json")

with open("category_mapping.pkl", "rb") as f:
    category_mapping = pickle.load(f)

# --- inventory model constants ---
Z_VALUE = 1.65   # ~95% service level
LEAD_TIME = 1    # weeks

# --- google trends config ---
GOOGLE_TRENDS_SEARCH_TERMS = {
    "long sleeves": "long sleeve shirt",
    "short sleeves": "short sleeve shirt",
    "lowers": "lowers"
}

GOOGLE_TRENDS_LOOKBACK_DAYS = 270  # ~9 months of weekly data

# in-memory cache so we're not hitting trends on every request
_gtrends_cache = {}


class PredictionInput(BaseModel):
    category: Literal["long sleeves", "short sleeves", "lowers"]
    forecast_date: date

# last 7 weeks of sales, week_1 = oldest, week_7 = most recent

    week_1_sales: float = Field(..., ge=0)
    week_2_sales: float = Field(..., ge=0)
    week_3_sales: float = Field(..., ge=0)
    week_4_sales: float = Field(..., ge=0)
    week_5_sales: float = Field(..., ge=0)
    week_6_sales: float = Field(..., ge=0)
    week_7_sales: float = Field(..., ge=0)

    @computed_field
    @property
    def lag_1(self) -> float:
        return self.week_7_sales

    @computed_field
    @property
    def lag_7(self) -> float:
        return self.week_1_sales

    @computed_field
    @property
    def rolling_7(self) -> float:
        return (
            self.week_1_sales + self.week_2_sales + self.week_3_sales
            + self.week_4_sales + self.week_5_sales + self.week_6_sales
            + self.week_7_sales
        ) / 7

    @computed_field
    @property
    def month(self) -> int:
        return self.forecast_date.month

    @computed_field
    @property
    def week(self) -> int:
        return self.forecast_date.isocalendar().week


@app.get("/")
def home():
    return {
        "message": "Weekly Sales Demand Forecasting API is running",
        "model": "XGBoost Regressor",
        "categories": ["long sleeves", "short sleeves", "lowers"]
    }


def get_google_trends(
    category: str,
    forecast_date: date,
    retries: int = 3,
    backoff_seconds: float = 2.0
) -> float:
    """
    Pull the Google Trends interest score for a category around
    a given date, using pytrends.

    Trends only gives us data up to "today", so if forecast_date
    is in the future we fall back to the latest known value.
    Results are cached per (category, date) since Trends rate
    limits fast.
    """

    cache_key = (category, forecast_date)
    if cache_key in _gtrends_cache:
        return _gtrends_cache[cache_key]

    search_term = GOOGLE_TRENDS_SEARCH_TERMS.get(category)
    if search_term is None:
        raise HTTPException(
            status_code=500,
            detail=f"No Google Trends search term configured for '{category}'."
        )

    today = datetime.now(timezone.utc).date()
    start_date = today - timedelta(days=GOOGLE_TRENDS_LOOKBACK_DAYS)
    timeframe = f"{start_date.isoformat()} {today.isoformat()}"

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            pytrends = TrendReq(hl="en-US", tz=0)
            pytrends.build_payload([search_term], timeframe=timeframe)
            trends_df = pytrends.interest_over_time()

            if trends_df.empty:
                raise HTTPException(
                    status_code=503,
                    detail=f"Google Trends returned no data for '{search_term}'."
                )

            series = trends_df[search_term]

# latest point at or before forecast_date, or the most
# recent value if forecast_date is out of range

            eligible = series[series.index.date <= forecast_date]
            gtrend_value = float(
                eligible.iloc[-1]) if not eligible.empty else float(series.iloc[-1])

            _gtrends_cache[cache_key] = gtrend_value
            return gtrend_value

        except HTTPException:
            raise

        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)

    raise HTTPException(
        status_code=503,
        detail=f"Google Trends could not be reached after {retries} attempts. Error: {last_error}"
    )


def calculate_tail_risk(category: str):
    """
    Pulls historical prediction errors for a category and works out
    sigma + a tail-risk figure to feed into the safety stock calc.

    residual = actual sales - predicted sales
    tail risk = 95th percentile of |residual|
    """

    try:
        df = pd.read_csv("inventory_simulation_data_model.csv")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="inventory_simulation_data_model.csv was not found."
        )

    required_columns = ["category", "sales", "y_pred"]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=500, detail=f"Missing columns: {missing_columns}")

    df["residual"] = df["sales"] - df["y_pred"]
    df["abs_residual"] = df["residual"].abs()

    category_df = df[df["category"].str.lower() == category].copy()
    if category_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No historical prediction data found for '{category}'."
        )

    return {
        "sigma": float(category_df["residual"].std()),
        "tail_risk": float(category_df["abs_residual"].quantile(0.95))
    }


@app.post("/predict")
def predict(data: PredictionInput):

    # look up the encoded category
    if data.category not in category_mapping:
        raise HTTPException(
            status_code=500,
            detail=f"Category '{
                data.category}' not found in category_mapping.pkl."
        )
    category_code = int(category_mapping[data.category])

    # pull current trend signal for this category
    gtrend = get_google_trends(data.category, data.forecast_date)

    # build the row the model expects
    model_input = pd.DataFrame([{
        "gtrend": gtrend,
        "lag_1": data.lag_1,
        "lag_7": data.lag_7,
        "rolling_7": data.rolling_7,
        "month": data.month,
        "week": data.week,
        "category_code": category_code
    }])

    prediction = model.predict(model_input)
    predicted_demand = max(0, float(prediction[0]))  # no negative demand

    # safety stock + reorder point, using the forecasted demand
    # rather than the historical average
    risk = calculate_tail_risk(data.category)
    sigma = risk["sigma"]
    tail_risk = risk["tail_risk"]

    ss_sigma = Z_VALUE * sigma * np.sqrt(LEAD_TIME)
    # use whichever is more conservative
    safety_stock = max(ss_sigma, tail_risk)
    reorder_point = (predicted_demand * LEAD_TIME) + safety_stock

    return {
        "category": data.category,
        "forecast_date": data.forecast_date.isoformat(),
        "prediction": {
            "predicted_demand": round(predicted_demand, 2),
            "google_trends": round(float(gtrend), 2)
        },
        "inventory": {
            "tail_risk": round(tail_risk, 2),
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(reorder_point, 2)
        },
        "model_features": {
            "lag_1": round(data.lag_1, 2),
            "lag_7": round(data.lag_7, 2),
            "rolling_7": round(data.rolling_7, 2),
            "month": data.month,
            "week": data.week,
            "category_code": category_code
        }
    }
