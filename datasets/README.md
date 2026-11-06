# Dataset Documentation

This folder contains the three main datasets used in the project **Apparel Demand Forecasting and Inventory Optimization Using Machine Learning**.

Each dataset serves a different purpose across the two phases of the project.

## 1. `model_dataset.csv`

**Phase:** Phase A — Average Product Price Forecasting  
**Source:** Visuelle 2.0

This is a constructed dataset developed by integrating data from the Visuelle 2.0 sales and restock files with Google Trends information.

The data preparation process involved combining product-level sales and restock information with product attributes and external search-interest signals.

The resulting dataset contains information such as:

- Product identifiers and retail information.
- Product categories, colours and fabric attributes.
- Product release dates and restock information.
- Product prices and date information.
- Google Trends features related to product categories, colours and fabrics.

The dataset was used to explore and benchmark average product price forecasting using:

- ARIMA
- Prophet
- XGBoost

**Purpose:** To investigate the performance of different forecasting models using a feature-rich apparel dataset.

## 2. `final_dataset.csv`

**Phase:** Phase B — Demand Forecasting and Inventory Parameter Estimation

This dataset was prepared by combining category-level retail sales information with Google Trends data.

Since the original retail data had limited temporal information, normalized Google Trends weights were used to distribute aggregate category-level sales across observation dates.

The dataset contains:

| Column | Description |
|---|---|
| `date` | Observation date |
| `category` | Apparel category |
| `sales` | Reconstructed sales quantity |
| `gtrend` | Google Trends interest |
| `time_weight` | Normalized temporal weight |

**Purpose:** Used for demand forecasting, feature engineering, residual analysis and inventory parameter estimation, including safety stock, reorder point and EOQ.

**Important:** The temporal demand values are reconstructed estimates based on aggregate sales and Google Trends weights, not directly observed weekly transactions.

## 3. `inventory_simulation_data_model.csv`

**Phase:** Phase B — Inventory Simulation

This is a separate prepared dataset used to evaluate inventory replenishment policies through simulation.

It contains actual demand, category information, engineered forecasting features and model predictions.

Important fields include:

- `date`
- `category`
- `sales`
- `gtrend`
- `lag_1`
- `lag_7`
- `rolling_7`
- `y_pred` — XGBoost predictions
- `y_naive` — Naive baseline predictions

**Purpose:** Used to compare XGBoost-based and Naive-based inventory policies by evaluating inventory levels, stockouts, holding costs, stockout costs, total costs and fill rates.

---

## Dataset Usage Summary

| Dataset | Project Phase | Main Application |
|---|---|---|
| `model_dataset.csv` | Phase A | Average product price forecasting |
| `final_dataset.csv` | Phase B | Demand forecasting and inventory parameter estimation |
| `inventory_simulation_data_model.csv` | Phase B | Inventory policy simulation |

## Notes

- The three datasets serve different purposes and are not interchangeable.
- `model_dataset.csv` was constructed using Visuelle 2.0 sales, restock and Google Trends data.
- `final_dataset.csv` contains reconstructed temporal demand estimates.
- `inventory_simulation_data_model.csv` is used separately for inventory simulation.
- The original source datasets and their associated usage or redistribution terms should be checked before sharing derived data publicly.