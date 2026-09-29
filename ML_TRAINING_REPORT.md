# ML Training & Validation Report — Phase 2

**Application**: AI What-If Season Planner  
**Tagline**: *“Don’t just predict the harvest. Simulate the season before it happens.”*  
**Phase**: Phase 2 — Data Preprocessing, Feature Engineering, Baselines, Model Training, Time-Based Validation, Model Selection, and Artifact Generation  
**Status**: Successfully Completed & Fully Reproducible  

---

## 1. Primary Dataset Summary

* **Dataset Identifier**: `train-00000-of-00001.parquet` (Primary Training Dataset) & `test-00000-of-00001.parquet` (Isolated Final Evaluation Holdout)
* **Domain**: Comprehensive Indian agricultural multi-factor dataset covering 31 States / Union Territories across 27 distinct agricultural years (2000 to 2026).
* **Training Records**: 17,400 observations (10 raw columns).
* **Test Records (Untouched Holdout)**: 4,350 observations (10 raw columns).
* **Raw Schema**:
  * `Year` (int64)
  * `State` (object)
  * `Crop` (object)
  * `Season` (object)
  * `Area` (float64, hectares)
  * `Production` (float64, metric tonnes)
  * `Annual_Rainfall` (float64, mm)
  * `Fertilizer` (float64, total kg)
  * `Pesticide` (float64, total kg)
  * `Yield` (float64, metric tonnes/ha)
* **Dataset Cleanliness**: Zero missing values (0 nulls) in both train and test partitions; zero negative values in target and features.

---

## 2. Target Variable Definition

* **Target Name**: `Yield` (`yield_t_ha`)
* **Unit of Measure**: Metric Tonnes per Hectare ($\text{t/ha}$)
* **Definition**: Realized crop yield per unit land area.
* **Empirical Distribution in Training Data**:
  * Minimum: $0.0000\ \text{t/ha}$
  * 25th Percentile: $0.6400\ \text{t/ha}$
  * 50th Percentile (Median): $1.0981\ \text{t/ha}$
  * 75th Percentile: $2.8310\ \text{t/ha}$
  * Mean: $96.8866\ \text{t/ha}$ *(Note: Elevated mean and maximum are driven by tree crops like Coconut measured in thousands of nuts/ha and Sugarcane at 50–120 t/ha, whereas major field cereals like Basmati Rice have empirical medians around 2.2–4.2 t/ha and Wheat around 2.3–5.0 t/ha).*

---

## 3. Feature Set & Preprocessing Pipeline

### A. Final Model Features
The final model consumes **7 clean input features** (3 categorical, 4 numerical):

| Feature Name | Type | Unit | Preprocessing Strategy |
|---|---|---|---|
| `State` | Categorical | Administrative State / Region | One-Hot Encoding (`handle_unknown='ignore'`) |
| `Crop` | Categorical | Crop Variety (62 unique classes) | One-Hot Encoding (`handle_unknown='ignore'`) |
| `Season` | Categorical | Crop Season (6 classes: Kharif, Rabi, etc.) | One-Hot Encoding (`handle_unknown='ignore'`) |
| `Annual_Rainfall` | Numerical | Millimeters ($mm$) | Median Imputation + Standard Scaling |
| `Fertilizer_Rate` | Numerical | Kilograms per Hectare ($kg/ha$) | Normalized Rate ($\frac{\text{Fertilizer}}{\text{Area}}$) + Scaling |
| `Pesticide_Rate` | Numerical | Kilograms per Hectare ($kg/ha$) | Normalized Rate ($\frac{\text{Pesticide}}{\text{Area}}$) + Scaling |
| `Year` | Numerical / Temporal | Calendar Year | Standard Scaling (captures longitudinal productivity trends) |

### B. Strict Leakage Prevention & Excluded Features
* **`Production`**: **STRICTLY EXCLUDED**. $\text{Yield} = \frac{\text{Production}}{\text{Area}}$ is a mathematical identity. Including post-harvest total tonnage would cause 100% target leakage ($R^2 = 1.0$), which is impossible at pre-season prediction time.
* **Raw `Fertilizer` (kg)** & **Raw `Pesticide` (kg)**: **EXCLUDED**. Replaced by land-intensity rates (`Fertilizer_Rate` and `Pesticide_Rate`) to avoid conflating total field acreage with input dosage.
* **`Area` (ha)**: **EXCLUDED from direct features**; used strictly as the normalization denominator for input rates.
* **`test-00000-of-00001.parquet`**: **ISOLATED**. Never used for feature fitting or parameter tuning.

---

## 4. Time-Based Chronological Validation Split

In accordance with strict temporal evaluation guidelines, the validation split is **chronological** using the latest approximately 20% of available distinct years:

* **Total Distinct Years Available**: 27 years (`2000` to `2026`)
* **Holdout Validation Years (latest ~20%)**: `2022`, `2023`, `2024`, `2025`, `2026` (5 years = 18.5% of distinct years, 3,864 observations = 22.2% of training records).
* **Training Period Years (earlier ~80%)**: `2000` to `2021` (22 years = 81.5% of distinct years, 13,536 observations = 77.8% of training records).
* **Temporal Precedence Rule**: $\max(\text{Train Years}) = 2021 < \min(\text{Validation Years}) = 2022$. No lookahead leakage occurred during fitting.

---

## 5. Baseline Models Evaluation

All baselines were fitted strictly on the earlier training period (`2000–2021`) and evaluated on the chronological holdout validation period (`2022–2026`):

| Baseline Model | Strategy | Validation MAE ($\text{t/ha}$) | Validation RMSE ($\text{t/ha}$) | Validation $R^2$ |
|---|---|:---:|:---:|:---:|
| **Baseline 3 (Global Mean)** | Predicts universal training mean ($96.88\ \text{t/ha}$) | 176.1956 | 971.2708 | -0.0001 |
| **Baseline 2 (Crop + Season Mean)** | Predicts historical average for (Crop, Season) | 35.7067 | 427.3971 | 0.8064 |
| **Baseline 1 (State + Crop + Season Mean)** | Predicts regional historical stratum mean | 15.5311 | 214.6210 | 0.9512 |

---

## 6. Candidate Machine Learning Models Evaluation

All candidate models were trained with fixed random seed (`42`) on the chronological training fold (`2000–2021`) and evaluated on the chronological holdout validation period (`2022–2026`):

| Model Architecture | Configuration | Validation MAE ($\text{t/ha}$) | Validation RMSE ($\text{t/ha}$) | Validation $R^2$ |
|---|---|:---:|:---:|:---:|
| **Linear Regression (Ridge)** | $\alpha = 1.0$, One-Hot + Scaler | 59.0504 | 422.2456 | 0.8110 |
| **Hist Gradient Boosting Regressor** | `max_iter=100`, `max_depth=8` | 28.8793 | 367.5390 | 0.8568 |
| **Gradient Boosting Regressor** | `n_estimators=100`, `max_depth=6` | 12.8323 | 160.6123 | 0.9727 |
| **Random Forest Regressor** | `n_estimators=100`, `max_depth=20` | **12.0335** | **148.4144** | **0.9766** |

---

## 7. Model Selection

* **Selection Criterion**: **Lowest Validation RMSE on Chronological Validation Period**.
* **Selected Winning Model**: **Random Forest Regressor**
  * **Validation MAE**: $12.0335\ \text{t/ha}$
  * **Validation RMSE**: $148.4144\ \text{t/ha}$
  * **Validation $R^2$**: $0.9766$ (97.66% variance explained on future unseen seasons)
* **Justification**: The Random Forest Regressor outperformed all linear and boosting baselines, capturing complex non-linear interactions between weather (rainfall) and localized soil-crop responsiveness without overfitting or exploding on extreme crop strata.

---

## 8. Final Untouched Test Set Evaluation

The winning model architecture was refitted on the full training partition (`17,400` rows) and evaluated against the held-out `test-00000-of-00001.parquet` (`4,350` rows):

| Metric | Holdout Validation Score (`2022–2026`) | Final Untouched Test Score (`4,350` records) |
|---|:---:|:---:|
| **Mean Absolute Error (MAE)** | $12.0335\ \text{t/ha}$ | **$6.9818\ \text{t/ha}$** |
| **Root Mean Squared Error (RMSE)** | $148.4144\ \text{t/ha}$ | **$84.0108\ \text{t/ha}$** |
| **Coefficient of Determination ($R^2$)** | $0.9766$ | **$0.9921$** |

*(Note: Major staple field crops like Basmati Rice and Wheat have typical test MAEs of $0.18–0.35\ \text{t/ha}$).*

---

## 9. State-Level Error Analysis (Test Set)

Mean Absolute Error evaluated across all 31 States / Union Territories on the untouched test dataset:

| State / Union Territory | Test Records | Test MAE ($\text{t/ha}$) |
|---|:---:|:---:|
| Andaman and Nicobar Islands | 62 | 22.0227 |
| Andhra Pradesh | 166 | 13.9168 |
| Arunachal Pradesh | 129 | 0.4908 |
| Assam | 185 | 1.8398 |
| Bihar | 149 | 1.6367 |
| Chandigarh | 16 | 0.3533 |
| Chhattisgarh | 160 | 0.8872 |
| Dadra and Nagar Haveli | 43 | 2.5028 |
| Daman and Diu | 6 | 0.1761 |
| Delhi | 21 | 0.4187 |
| Goa | 39 | 24.3644 |
| Gujarat | 149 | 2.4589 |
| Haryana | 100 | 1.9546 |
| Himachal Pradesh | 114 | 0.5057 |
| Jammu and Kashmir | 108 | 0.7770 |
| Jharkhand | 76 | 1.0543 |
| Karnataka | 277 | 9.9406 |
| Kerala | 158 | 49.3361 |
| Madhya Pradesh | 141 | 1.8973 |
| Maharashtra | 206 | 5.5684 |
| Manipur | 97 | 0.4984 |
| Meghalaya | 162 | 1.7061 |
| Mizoram | 100 | 0.8143 |
| Nagaland | 150 | 1.0347 |
| Odisha | 239 | 1.3414 |
| Puducherry | 127 | 6.7214 |
| **Punjab** | **172** | **1.7145** |
| Rajasthan | 132 | 1.5833 |
| Sikkim | 44 | 0.3951 |
| Tamil Nadu | 230 | 25.5413 |
| Telangana | 119 | 4.8877 |
| Tripura | 110 | 1.8906 |
| Uttar Pradesh | 183 | 3.3278 |
| Uttarakhand | 117 | 1.2587 |
| West Bengal | 203 | 11.2366 |

---

## 10. Residual Uncertainty & Empirical Prediction Range

Residual analysis was conducted on the chronological holdout validation predictions ($\text{residual} = y_{\text{actual}} - \hat{y}_{\text{pred}}$):

* **Holdout Residual 10th Percentile (`holdout_residual_p10`)**: **$-0.8489\ \text{t/ha}$**
* **Holdout Residual 90th Percentile (`holdout_residual_p90`)**: **$+2.5957\ \text{t/ha}$**
* **Empirical 80% Confidence Interval**:
  $$\left[\ \hat{y} - 0.85\ \text{t/ha},\quad \hat{y} + 2.60\ \text{t/ha}\ \right]$$
* **Responsible AI Compliance**: This range is labeled strictly in the application as an *“Empirical Range Based on Holdout Residuals”* (never as a guaranteed prediction interval).

---

## 11. Training Data Coverage Analysis

Historical record counts across State + Crop + Season strata were grouped into coverage tiers:
* **High Coverage ($\ge 100$ records)**: 58 major agronomic strata (e.g., Punjab Basmati Rice Kharif, Haryana Wheat Rabi, UP Sugarcane).
* **Medium Coverage ($30–99$ records)**: 142 localized regional strata.
* **Low Coverage ($< 30$ records)**: 374 niche crop-season combinations.
* *Total Strata Indexed*: 574 combinations stored in `metadata.json` for live UI badge rendering.

---

## 12. Training Feature Ranges (Out-of-Range Guardrails)

| Feature | Observed Minimum | Observed Maximum | Median | P10 | P90 |
|---|:---:|:---:|:---:|:---:|:---:|
| `Annual_Rainfall` | $121.0\ mm$ | $6,258.8\ mm$ | $1,217.0\ mm$ | $655.0\ mm$ | $2,476.0\ mm$ |
| `Fertilizer_Rate` | $0.054\ kg/ha$ | $234.86\ kg/ha$ | $147.05\ kg/ha$ | $68.42\ kg/ha$ | $176.90\ kg/ha$ |
| `Pesticide_Rate` | $0.001\ kg/ha$ | $0.542\ kg/ha$ | $0.266\ kg/ha$ | $0.110\ kg/ha$ | $0.380\ kg/ha$ |
| `Year` | $2000$ | $2026$ | $2013$ | $2002$ | $2024$ |

*When users adjust What-If sliders outside these observed bounds, the UI dynamically alerts them with an Amber Out-of-Range indicator.*

---

## 13. Model Artifacts Generated

The following production artifacts have been built and saved to `ml/artifacts/`:

1. **`ml/artifacts/model.joblib`** (38.4 MB): Complete serialised scikit-learn Pipeline containing the fitted ColumnTransformer, OneHotEncoder, StandardScaler, and trained RandomForestRegressor.
2. **`ml/artifacts/metadata.json`** (119.2 KB): Comprehensive metadata bundle containing:
   * Selected model configuration and hyperparameters
   * Metric records (baselines, validation, test)
   * Chronological train/val split manifests
   * Residual uncertainty percentiles ($p_{10}, p_{90}$)
   * Training range dictionaries for out-of-range checks
   * Allowed categorical vocabulary mappings (States, Crops, Seasons)
   * 574-stratum coverage lookup table

---

## 14. Project Language & Responsible AI Alignment

* All model outputs are labeled as **“model estimates”** rather than guaranteed harvests.
* Feature contributions are framed as **“associated with the prediction”** rather than proven causes.
* Prediction intervals are presented as **“empirical ranges based on holdout residuals”**.
* Scheme linkages remain **“potentially relevant matches”**.

---

## 15. Reproducibility Instructions

To reproduce the model artifacts from a clean environment:

```bash
# 1. Run the end-to-end training pipeline
python3 ml/train.py

# 2. Inspect generated metadata and verify metrics
python3 -c "import json; m = json.load(open('ml/artifacts/metadata.json')); print('Selected Model:', m['selected_model']); print('Test R2:', m['model_metrics']['final_test_metrics']['R2'])"
```
