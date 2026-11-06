# Apparel Demand Forecasting and Inventory Optimization Using Machine Learning

**A data-driven approach to demand forecasting and inventory optimization for apparel retail.**

This project explores how machine learning, external consumer-interest signals and Operations Research techniques can be combined to improve demand forecasting and inventory planning in apparel retail.

It began as my M.Tech final-year thesis project in Operations Research at the **National Institute of Technology, Durgapur**. After completing the thesis, I continued developing the project by improving the inventory replenishment approach and building an interactive forecasting application using FastAPI and Streamlit.

The project is organized into two main research phases, followed by a post-thesis extension.

## Project Overview

The project addresses two related problems:

1. How can forecasting models be used to understand and predict patterns in apparel retail data?
2. How can demand forecasts and their associated uncertainty be translated into practical inventory decisions?

The work combines data preparation, time-series forecasting, residual analysis, inventory optimization and application development.

### Complete Project Workflow

![Complete Project Workflow](images/workflow_pipeline.png)

## Phase A — Average Product Price Forecasting

### 1. Dataset Preparation

For the first phase, I worked with the Visuelle 2.0 apparel dataset.

I constructed a feature-rich dataset, `model_dataset.csv`, by integrating information from:

- Sales data.
- Restock data.
- Google Trends data.

The resulting dataset contains product-level information, retail identifiers, product categories, product attributes, price information, release dates, restock information and external search-interest features.

The objective of this phase was to explore forecasting techniques using a feature-rich apparel dataset.

### 2. Forecasting Models

I compared three forecasting approaches:

- ARIMA
- Prophet
- XGBoost

For this experiment, product prices were aggregated by date to calculate average product price. The models were evaluated on their ability to forecast the average product price over time.

### 3. Model Evaluation

| Model | RMSE | MAE |
|---|---:|---:|
| **XGBoost** | **0.00615** | **0.00483** |
| Prophet | 0.00784 | 0.00644 |
| ARIMA | 0.01373 | 0.01082 |

XGBoost achieved the lowest RMSE and MAE among the three models, followed by Prophet and ARIMA.

### Actual vs Predicted Prices — Test Period

![Actual vs Predicted Average Prices](results/phase1_actual_vs_predicted_prices.png)

**Phase A outcome:** This experiment provided a comparative understanding of forecasting models using a feature-rich apparel dataset.

*Note: Phase A focuses on average product price forecasting, not demand forecasting.*

---

## Phase B — Demand Forecasting and Inventory Optimization

The second phase focused on the main research problem: forecasting apparel demand and translating those forecasts into inventory decisions under limited historical data.

### 1. Retail Data Preparation

The primary retail dataset had limited temporal information and inconsistent product descriptions.

I performed data cleaning and preparation, including:

- Standardizing column names.
- Removing missing and invalid records.
- Cleaning product descriptions.
- Converting sales quantities into numeric values.
- Mapping products into four categories:
  - Long Sleeves
  - Short Sleeves
  - Lowers
  - Printed Shirt
- Aggregating sales quantities at category level.

### 2. Google Trends Integration and Temporal Demand Reconstruction

One of the main challenges was the absence of a complete time-indexed sales history.

To address this, I integrated Google Trends data as an external consumer-interest signal.

The Google Trends data was transformed into a date-category format. I then calculated normalized temporal weights:

\[
w_{c,t}=\frac{GT_{c,t}}{\sum_t GT_{c,t}}
\]

These weights were used to distribute aggregate category-level sales across observation dates, creating a reconstructed temporal demand dataset.

This approach allowed me to develop a time-series forecasting pipeline despite the limited temporal information in the original retail data.

**Important:** The resulting temporal demand values are reconstructed estimates based on aggregate sales and Google Trends weights, rather than directly observed weekly transactions.

### 3. Dataset Usage

Phase B uses two separate prepared datasets for different purposes.

**`final_dataset.csv` — Demand Forecasting and Inventory Parameter Estimation**

This dataset contains:

| Column | Description |
|---|---|
| `date` | Observation date |
| `category` | Apparel category |
| `sales` | Reconstructed sales quantity |
| `gtrend` | Google Trends interest |
| `time_weight` | Normalized temporal weight |

It was used for demand forecasting, residual analysis and inventory parameter calculations.

**`inventory_simulation_data_model.csv` — Inventory Simulation**

This separate dataset contains actual demand, category information, engineered features and model predictions.

Important fields include:

- `sales`
- `gtrend`
- `lag_1`
- `lag_7`
- `rolling_7`
- `category_code`
- `y_pred` — XGBoost predictions
- `y_naive` — Naive baseline predictions

It was used to compare inventory replenishment policies through simulation.

For detailed dataset descriptions, refer to [`data/README.md`](data/README.md).

### 4. Feature Engineering

I engineered features to capture historical demand patterns, seasonality and external consumer-interest signals.

| Feature | Description |
|---|---|
| `lag_1` | Previous observation's demand |
| `lag_7` | Demand seven observations earlier |
| `rolling_7` | Seven-observation rolling statistic |
| `gtrend` | Google Trends interest |
| `month` | Calendar month |
| `week` | Week of the year |
| `category_code` | Encoded apparel category |

These features were used to train the XGBoost demand forecasting model.

### 5. XGBoost Demand Forecasting

I trained an XGBoost model and compared its performance against three baseline forecasting methods:

- Moving Average (7)
- Naive Forecast (lag 1)
- Seasonal Naive (lag 7)

A chronological 80:20 train-test split was used for each category.

#### XGBoost Hyperparameters

| Parameter | Value |
|---|---:|
| `n_estimators` | 700 |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 1.0 |

#### Model Evaluation

| Model | RMSE | MAE |
|---|---:|---:|
| **XGBoost** | **48.08** | **20.10** |
| Moving Average (7) | 303.18 | 166.23 |
| Naive (lag 1) | 404.46 | 201.89 |
| Seasonal Naive (lag 7) | 414.61 | 243.76 |

XGBoost achieved substantially lower forecasting errors than the baseline methods on the evaluated test set.

*Evaluation note: The test evaluation used a rolling one-step-ahead approach, where lag features could use preceding actual observations. It was not a fully recursive multi-week forecast.*

### 6. Forecast Residual Analysis

Forecasting accuracy alone does not provide enough information for inventory planning. Inventory decisions must also account for uncertainty in the predictions.

I therefore performed residual analysis.

Residuals were calculated as:

\[
e_t=Actual_t-Predicted_t
\]

The analysis included:

- Residual distribution.
- Category-wise residual behaviour.
- Residuals over time.
- Aggregate residual patterns.

These residuals were subsequently used to estimate safety stock and reorder points.

#### Residual Distribution

![Residual Distribution](results/residual_distribution.png)

#### Category-wise Residuals Over Time

The following visualization compares XGBoost and Naive residuals across the four apparel categories.

![Category-wise Residual Analysis](results/category_wise_residuals.png)

#### Aggregate Residuals Over Time

![Aggregate Residuals Over Time](results/aggregate_residuals_over_time.png)

### 7. Inventory Parameter Estimation

After forecasting demand, I applied Operations Research techniques to calculate inventory parameters.

The main parameters were:

- Safety Stock (SS)
- Reorder Point (ROP)
- Economic Order Quantity (EOQ)

#### Safety Stock

I considered both residual standard deviation and tail risk to estimate safety stock.

The statistical safety-stock component was:

\[
SS_{\sigma}=z\sigma_{\text{residual}}\sqrt{L}
\]

The tail-risk measure was calculated using the 95th percentile of absolute residuals:

\[
TailRisk=Q_{0.95}(|e|)
\]

The final safety stock was:

\[
SS=\max(SS_{\sigma},TailRisk)
\]

Where:

- \(z\) represents the service-level factor.
- \(\sigma_{\text{residual}}\) represents residual standard deviation.
- \(L\) represents lead time in observation periods.

The calculation used a service-level factor of 1.65 and a lead time of one observation period.

#### Reorder Point

\[
ROP=\mu_{LT}+SS
\]

Where \(\mu_{LT}\) is expected demand during lead time.

#### Economic Order Quantity

\[
EOQ=\sqrt{\frac{2DS}{H}}
\]

Where:

- \(D\) = Total demand over the planning horizon.
- \(S\) = Ordering cost per order.
- \(H\) = Holding cost per unit.

#### Estimated Inventory Parameters

| Category | Average Demand | Sigma | Tail Risk | Safety Stock | Reorder Point | EOQ |
|---|---:|---:|---:|---:|---:|---:|
| Short Sleeves | 3245.94 | 86.38 | 83.11 | 142.53 | 3388.48 | 8353.91 |
| Lowers | 1088.83 | 32.17 | 65.73 | 65.73 | 1154.56 | 4838.36 |
| Long Sleeves | 931.39 | 24.28 | 42.54 | 42.54 | 973.93 | 4474.91 |
| Printed Shirt | 70.23 | 1.35 | 3.78 | 3.78 | 74.01 | 1228.80 |

The results show that different apparel categories have different demand levels and forecasting uncertainty, which supports the use of category-specific inventory parameters.

### 8. Inventory Simulation — Original Thesis Experiment

To evaluate how forecasting methods influence inventory decisions, I developed a base-stock inventory simulation.

The simulation compared two replenishment policies:

1. XGBoost-based forecasting.
2. Naive-based forecasting.

The original thesis simulation used the following parameters:

| Parameter | Value |
|---|---:|
| Initial inventory | 300 units |
| Lead time | 7 days |
| Holding cost per unit | 2 |
| Stockout cost per unit | 5 |
| Fixed safety stock | 150 units |

The replenishment target was calculated as:

\[
TargetStock=ForecastDemandDuringLeadTime+SafetyStock
\]

The simulation accounted for inventory position, outstanding orders, demand fulfilment, holding costs and stockout costs.

#### Overall Simulation Results

| Metric | XGBoost | Naive |
|---|---:|---:|
| Total Holding Cost | 70,373.60 | 66,457.63 |
| Total Stockout Cost | 71,694.32 | 59,072.05 |
| Total Stockouts | 14,338.86 | 11,814.41 |
| **Total Cost** | **142,067.91** | **125,529.68** |
| Fill Rate | 93.75% | 94.85% |

An interesting observation emerged from this experiment.

Although XGBoost achieved considerably better forecasting accuracy, the Naive-based inventory policy performed better in terms of total inventory cost and fill rate in the original simulation.

This highlighted an important distinction:

**Better forecasting accuracy does not necessarily translate into better inventory performance.**

#### Inventory Level Over Time

![Inventory Level Over Time](results/inventory_level_over_time.png)

---

## Post-Thesis Extension — Improving Inventory Decisions

After completing the thesis, I continued working on the project to improve the inventory policy and make the forecasting framework accessible through an application.

### 9. Replacing Fixed Safety Stock with Category-Specific Safety Stock

One limitation of the original thesis simulation was the use of a uniform safety-stock value of 150 units for every category.

This meant that categories with very different demand uncertainty were assigned the same inventory buffer.

As a post-thesis improvement, I replaced this uniform safety-stock approach with category-specific safety-stock estimates derived from residual variability and tail-risk estimates.

#### Original vs Updated Safety Stock

| Category | Original Fixed Safety Stock | Updated Category-Specific Safety Stock |
|---|---:|---:|
| Short Sleeves | 150 | 142.53 |
| Long Sleeves | 150 | 42.54 |
| Lowers | 150 | 65.73 |
| Printed Shirt | 150 | 3.78 |

The updated approach reduces the inventory buffer for categories with lower estimated forecasting uncertainty.

I incorporated these category-specific values into the updated inventory simulation to move towards a more differentiated replenishment policy.

*The values above represent the updated safety-stock estimates. They should not be confused with the earlier category-specific simulation run, which used previous safety-stock estimates.*

### 10. Building a REST API with FastAPI

I extended the project by developing a REST API using FastAPI to make the trained forecasting model accessible through an application.

The backend performs the following operations:

- Loads the trained XGBoost model.
- Loads the category encoder.
- Accepts the apparel category and forecast date.
- Takes sales information for the previous seven weeks.
- Creates lag-based and rolling features.
- Retrieves Google Trends information using PyTrends.
- Generates weekly demand predictions.
- Calculates safety-stock and reorder-point recommendations.
- Returns the results in JSON format.

#### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API health check |
| POST | `/predict` | Demand prediction and inventory recommendations |

The API also provides interactive documentation through Swagger UI.

#### FastAPI Swagger Interface

![FastAPI Swagger UI](images/fastapi_swagger.png)

### 11. Building an Interactive Frontend with Streamlit

To make the forecasting system easier to use, I developed a Streamlit frontend that communicates with the FastAPI backend.

The application allows users to:

- Select an apparel category.
- Choose a forecast date.
- Enter sales information for the previous seven weeks.
- Preview historical weekly sales.
- Submit forecasting requests.
- View predicted weekly demand.
- View Google Trends information.
- View safety stock and reorder point recommendations.
- Inspect model features and the raw API response.

#### Streamlit Application

![Streamlit Frontend](images/streamlit_frontend.png)

#### Application Workflow

1. The user enters the category, forecast date and recent weekly sales.
2. Streamlit sends the input to the FastAPI prediction endpoint.
3. FastAPI prepares the required features and retrieves Google Trends information.
4. The trained XGBoost model generates the demand prediction.
5. Inventory logic calculates safety stock and reorder point recommendations.
6. The API returns the results to Streamlit for display.

This extension helped turn the original research workflow into an application-oriented forecasting and inventory decision-support system.

---

## Technologies and Tools

| Area | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning | XGBoost, Scikit-learn |
| Time-Series Forecasting | ARIMA, Prophet |
| External Data Integration | Google Trends, PyTrends |
| API Development | FastAPI |
| API Documentation | Swagger UI |
| Frontend | Streamlit |
| HTTP Requests | Requests |
| Version Control | Git, GitHub |
| Development Environment | Jupyter Notebook, VS Code |

## Key Learnings

Working on this project gave me experience across the complete data science and Operations Research workflow.

Some of the key areas I explored were:

- Integrating multiple data sources to construct a feature-rich dataset.
- Cleaning and preparing apparel retail data.
- Using Google Trends as an external consumer-interest signal.
- Reconstructing temporal demand under limited historical data.
- Comparing time-series forecasting approaches.
- Engineering lag-based and rolling statistical features.
- Evaluating forecasting models using RMSE and MAE.
- Understanding forecasting uncertainty through residual analysis.
- Applying safety stock, reorder point and EOQ calculations.
- Evaluating replenishment policies through inventory simulation.
- Understanding the relationship between forecasting accuracy and inventory performance.
- Building REST APIs using FastAPI.
- Connecting machine learning models to an interactive frontend using Streamlit.

## Future Improvements

There are several areas where I would like to extend this project further:

- Improve temporal demand reconstruction using transaction-level sales data.
- Explore additional forecasting architectures.
- Incorporate variable lead times and dynamic ordering costs.
- Evaluate more advanced inventory replenishment policies.
- Introduce seasonal demand spikes and other scenario-based simulations.
- Implement model monitoring and performance tracking.
- Deploy the application to a cloud platform.

## Acknowledgements

I would like to express my sincere gratitude to **Prof. Gautam Panighahi, Department of Mathematics, National Institute of Technology Durgapur**, for his valuable guidance, support and supervision throughout my M.Tech thesis project.

His guidance and feedback were instrumental in carrying out this research on apparel demand forecasting and inventory optimization.

## Academic Background

This project was developed as my M.Tech final-year thesis in **Operations Research at the National Institute of Technology, Durgapur**.

The research focused on combining machine learning and Operations Research techniques for apparel demand forecasting and inventory optimization.

The post-thesis extension focused on improving category-specific inventory policies and developing an interactive forecasting application.

## Author
**Dhritiman Keot**
**M.Tech — Operations Research**
**Department of Computer Science and Engineering, Mathematics and Management Studies**
**National Institute of Technology, Durgapur**