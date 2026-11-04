# Rice Yield Prediction using Government Crop Data (APY) 

Predicting the rice yield (tonnes per hectare) of Indian districts from **state, season, cultivated area and year**, and comparing six machine learning models. Built for a Predictive Analytics course at Lovely Professional University.

**Full write-up:** [`Rice_Yield_Report.pdf`](Rice_Yield_Report.pdf) (6-page IEEE-style report)

## Overview ##

- **Data:** Area, Production and Yield (APY) crop statistics published by the Government of India (data.gov.in). Only this one dataset is used.
- **Raw size:** 345,336 records, 8 columns, 1997-2020.
- **After cleaning:** 21,340 rice records, 36 states and union territories, 673 districts, 1997-2019.
- **Task:** regression, predict `Yield` (t/ha).
- **Evaluation:** time-based split. Train on 1997-2014 (15,984 records), test on 2015-2019 (5,356 records), so the scores measure prediction of unseen later years.

## Pipeline ##

| Step | What happens | Rows left |
|---|---|---|
| Raw file | Load `APY.csv`, strip extra spaces in headers and text | 345,336 |
| Remove missing values | `dropna()` | 340,383 |
| Remove zero area/production | Yield cannot be computed meaningfully | 338,918 |
| Remove duplicates | None found | 338,918 |
| Remove year 2020 | Incomplete year | 338,599 |
| Keep rice only | Yields of different crops are not comparable | 21,550 |
| Remove top 1% yields | Likely data-entry or unit errors (above 4.70 t/ha) | 21,340 |

**Inputs (44 columns):** `Area`, `Crop_Year`, one-hot `State` (36), one-hot `Season` (6).
`Production` is **not** used as an input, because Yield = Production / Area would leak the answer.

## Models

| # | Model | scikit-learn call |
|---|---|---|
| 1 | Linear Regression | `LinearRegression()` |
| 2 | Polynomial Regression | `LinearRegression()` + `Area²`, `Year²` columns |
| 3 | Decision Tree | `DecisionTreeRegressor(max_depth=8)` |
| 4 | Random Forest | `RandomForestRegressor(n_estimators=200)` |
| 5 | Gradient Boosting | `GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)` |
| 6 | MLP Neural Network | `MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000)` |

All use `random_state=42`. Linear, Polynomial and MLP use standardised inputs (scaler fitted on training years only). No hyper-parameter tuning was done.

## Results (test years 2015-2019)

| Model | R² | MAE | MSE | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.478 | 0.508 | 0.450 | 0.671 |
| Polynomial Regression | 0.445 | 0.522 | 0.478 | 0.692 |
| Decision Tree | 0.242 | 0.650 | 0.654 | 0.809 |
| Random Forest | 0.504 | **0.487** | 0.427 | 0.654 |
| **Gradient Boosting** | **0.513** | 0.501 | **0.420** | **0.648** |
| MLP Neural Network | 0.371 | 0.558 | 0.542 | 0.736 |

MAE and RMSE are in tonnes per hectare.

## Key findings ##

- **Gradient Boosting is best**, but simple Linear Regression is only 0.035 R² behind. More complexity did not pay much.
- **Random Forest overfits:** R² of 0.949 on training data versus 0.504 on test data.
- **Area and year are the most important inputs** (random-forest importance 0.278 and 0.127).
- **Errors are mostly inside states:** the average prediction per state is close to the actual average, so the remaining error is district-to-district and year-to-year variation that these four inputs cannot see.
- **About half of the variance is unexplained**, because the dataset has no weather or irrigation information.

## Limitations

- No weather, irrigation, soil or fertiliser variables.
- One time split, no cross-validation, no hyper-parameter tuning, so small gaps between top models are not reliable rankings.
- District is not used as an input.
- The 99th-percentile outlier limit was computed on all years before the split.
- Tree-based models cannot extrapolate the rising yield level of the later years.

## Future work

- Add monsoon rainfall (for example IMD sub-division data) and irrigation information.
- Add district-level inputs.
- Time-aware cross-validation and hyper-parameter tuning.

## How to run

1. Get `APY.csv` from [data.gov.in](https://data.gov.in) and put it in the same folder as the notebook. (The file is about 21 MB, so it is not stored in this repository.)
2. Install the requirements:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Open the notebook and run all cells. Make sure the file path in the loading cell is just `"APY.csv"`:

   ```python
   df = pd.read_csv("APY.csv")
   ```

## Repository contents

```
.
├── pred_ca2.ipynb          # notebook: cleaning, plots, six models, evaluation
├── Rice_Yield_Report.pdf   # 6-page IEEE-style report
└── README.md
```

## Author

**Ruthvik Raj B R**, School of Computer Science and Engineering, Lovely Professional University, Phagwara, Punjab.

## Data source

Directorate of Economics and Statistics, Ministry of Agriculture and Farmers Welfare, Government of India, crop area, production and yield (APY) statistics, via the Open Government Data (OGD) Platform India.
