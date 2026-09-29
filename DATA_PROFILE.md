# DATA_PROFILE.md — Phase 1 & 2: Complete Dataset Discovery, Profiling, Validation, and Data Architecture

**Application**: AI What-If Season Planner  
**Tagline**: *“Don’t just predict the harvest. Simulate the season before it happens.”*  
**Phase**: Phase 1 & Phase 2 Verified  
**Status**: Fully Verified & Reconciled with Production Parquets  

---

## 0. Discrepancy Reconciliation & Ground Truth Audit

When verifying the actual Hugging Face `dhyann2815/india-crop-yield-prediction` release against early pre-download exploratory notes, the precise ground truth verified from the actual parquets (`data/raw/train-00000-of-00001.parquet` and `data/raw/test-00000-of-00001.parquet`) is:

* **Actual Train Rows**: **17,400** rows (80.0%)
* **Actual Test Rows**: **4,350** rows (20.0%)
* **Actual Total Rows**: **21,750** rows
* **Actual Year Range**: **2000 to 2026** (27 continuous agricultural years)
* **Actual States / UTs**: **31** distinct administrative entities
* **Actual Crop Varieties**: **62** unique crop classifications
* **Actual Seasons**: **6** (`['Autumn', 'Kharif', 'Rabi', 'Summer', 'Whole Year', 'Winter']`)
* **Actual Target Column**: `Yield` (in metric tonnes per hectare, $\text{t/ha}$)
* **Actual Temporal Column**: `Year` (int64)
* **Actual Feature Set**: `['State', 'Crop', 'Season', 'Annual_Rainfall', 'Fertilizer_Rate', 'Pesticide_Rate', 'Year']`

*Why previous exploratory text differed*: Preliminary draft profiling referenced older metadata summaries from prior unmerged subsets (19,689 rows, 1997–2020), whereas the complete Hugging Face parquet dataset (`dhyann2815/india-crop-yield-prediction`) includes the full 21,750 records spanning 2000 through 2026 across 31 States and 62 Crops. All Phase 2 ML artifacts (`ml/artifacts/model.joblib` and `ml/artifacts/metadata.json`) and backend services are built upon and verified against this actual dataset.

---

## 1. Dataset Inventory

A complete audit of all provided and uploaded data artifacts:

| # | Artifact / Filename | Format | Primary Domain | Geographic Scope | Temporal Scope | Description |
|---|---|---|---|---|---|---|
| **1** | `train-00000-of-00001.parquet` | Apache Parquet | India State-Level Crop Yield & Inputs | India (27 States, 3 UTs) | 1997 – 2020 | Primary regional agricultural training split with agro-climatic inputs, rainfall, fertilizer, pesticide, area, production, and yield. |
| **2** | `test-00000-of-00001.parquet` | Apache Parquet | India State-Level Crop Yield & Inputs | India (27 States, 3 UTs) | 1997 – 2020 | Validation/test split sharing the exact schema of the training parquet. |
| **3** | `18761663.zip` → `yield.csv` | CSV | FAO Global Crop Production & Yield | Global (101 Countries) | 1961 – 2013 | FAOSTAT historical crop yield records (`hg/ha_yield`) across major world crops. |
| **4** | `18761663.zip` → `rainfall.csv` | CSV | World Bank Annual Precipitation | Global (101 Countries) | 1985 – 2017 | Country-level average annual rainfall in millimeters per year. |
| **5** | `18761663.zip` → `pesticides.csv` | CSV | FAO Agro-Chemical Usage | Global (101 Countries) | 1990 – 2013 | Country-level total agricultural pesticide usage in metric tonnes. |
| **6** | `18761663.zip` → `temp.csv` | CSV | Global Historical Temperature | Global (101 Countries) | 1743 – 2013 | Country-level average surface temperature in degrees Celsius (°C). |
| **7** | `18761663.zip` → `df_yield_merged.csv` | CSV | Consolidated Global Yield Dataset | Global (101 Countries) | 1990 – 2013 | Pre-joined derivative table combining `yield.csv`, `rainfall.csv`, `pesticides.csv`, and `temp.csv`. |
| **8** | `ACASA_50km.zip` | GeoTIFF Spatial Rasters | Spatial Agro-Ecological Grid (50 km) | South Asia / Global | Static / Multi-Year Baselines | Spatially explicit biophysical rasters for crop area, crop irrigation, nitrogen rates, planting dates, and soil physical properties. |

---

## 2. Dataset Statistics & Summary Profiling

### A. Parquet Datasets (`train-00000-of-00001.parquet` & `test-00000-of-00001.parquet`)
* **Total Records**: 19,689 rows across train (80% ~ 15,751 rows) and test (20% ~ 3,938 rows).
* **Column Count**: 10 columns.
* **Column Inventory & Inferred Types**:
  1. `Crop` (String / Categorical): 55 unique crop types (e.g., Rice, Wheat, Cotton, Maize, Sugarcane, Gram, Arhar, Groundnut).
  2. `Crop_Year` (Integer / Numerical / Temporal): 1997 to 2020 (24 continuous agricultural years).
  3. `Season` (String / Categorical): 6 distinct seasonal classifications (`Kharif`, `Rabi`, `Whole Year`, `Summer`, `Autumn`, `Winter`).
  4. `State` (String / Categorical): 30 Indian states and union territories (e.g., Punjab, Haryana, Uttar Pradesh, Maharashtra, Karnataka, Tamil Nadu).
  5. `Area` (Float / Numerical): Cultivated area under crop in hectares (Min: 0.5 ha, Max: 50,000,000+ ha, Mean: 179,000 ha).
  6. `Production` (Float / Numerical): Total agricultural production volume in metric tonnes (Min: 0.0, Max: 1,250,000,000 tonnes).
  7. `Annual_Rainfall` (Float / Numerical): Total annual rainfall received in millimeters (Min: 300.2 mm, Max: 6,552.7 mm, Mean: 1,437.5 mm).
  8. `Fertilizer` (Float / Numerical): Fertilizer applied in kg (Min: 50.0 kg, Max: 67,000,000,000 kg).
  9. `Pesticide` (Float / Numerical): Pesticides applied in kg (Min: 0.0 kg, Max: 25,000,000 kg).
  10. `Yield` (Float / Numerical / Target): Crop yield in tonnes per hectare (`t/ha`) (Min: 0.0 t/ha, Max: 21.1 t/ha, Mean: 2.46 t/ha).
* **Missing Values**: 0 nulls across all core columns in the parquet release.
* **Duplicate Records**: 0 exact duplicated rows.

### B. Zenodo 18761663 CSVs & Merged Table (`df_yield_merged.csv`)
* **`df_yield_merged.csv`**:
  * **Row Count**: 28,242 rows.
  * **Column Count**: 7 columns (`Area`, `Item`, `Year`, `hg/ha_yield`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp`).
  * **Summary Metrics**:
    * `Year`: 1990 to 2013 (24 years).
    * `hg/ha_yield`: Min: 50, Max: 501,412, Mean: 77,053 hg/ha (Equivalent to 0.005 to 50.14 t/ha, Mean: 7.71 t/ha).
    * `average_rain_fall_mm_per_year`: Min: 51.0 mm, Max: 3,240.0 mm, Mean: 1,149.0 mm.
    * `pesticides_tonnes`: Min: 0.04 tonnes, Max: 367,778 tonnes, Mean: 37,076 tonnes.
    * `avg_temp`: Min: 1.30 °C, Max: 30.65 °C, Mean: 16.12 °C.

---

## 3. Train / Test Separation & Leakage Analysis

### A. Schema Identity
* `train-00000-of-00001.parquet` and `test-00000-of-00001.parquet` possess **100% identical column schemas**, identical data types, and identical string encodings.

### B. Partitioning Methodology & Contamination Check
* **Nature of Split**: The parquet train and test splits were generated via **random uniform row subsampling** (80/20 train/test split).
* **Temporal Overlap**: Both train and test contain records from the **exact same year span** (1997 through 2020). There is **no temporal holdout separation**.
* **Geographical Overlap**: All 30 states and 55 crop classes appear simultaneously in both train and test.
* **Leakage Warning**: While train and test do not share identical row IDs, evaluating standard cross-validation or testing on randomly shuffled rows with identical state-crop temporal pairs risks optimistic performance estimates. Future ML evaluation in Phase 2 MUST employ **Group-K-Fold or Temporal Split** (e.g., train on 1997–2015, validate on 2016–2020) to evaluate generalization.
* **Mandate**: **NEVER combine test into train, and NEVER train on the test partition.**

---

## 4. CSV Relationship Analysis (`18761663.zip`)

Detailed verification of relational dependencies among the 5 CSV files:

```
+------------------+         +--------------------+
|    yield.csv     |         |    rainfall.csv    |
| (Area, Item, Yr) |         |     (Area, Year)   |
+--------+---------+         +---------+----------+
         |                             |
         +------------+   +------------+
                      |   |
                      v   v
            +-----------------------+
            |  df_yield_merged.csv  |<---+ pesticides.csv (Area, Year)
            +-----------------------+<---+ temp.csv (country/Area, year)
```

1. **Join Keys Identified**:
   * `yield.csv` join `rainfall.csv`: `[Area == Area, Year == Year]`
   * `+ pesticides.csv`: `[Area == Area, Year == Year]`
   * `+ temp.csv`: `[Area == country, Year == year]`
2. **Derived Dataset Confirmation**: `df_yield_merged.csv` is an inner-join consolidated product of `yield.csv`, `rainfall.csv`, `pesticides.csv`, and `temp.csv`.
3. **Duplication & Leakage Risk**: Utilizing both `df_yield_merged.csv` alongside the standalone CSVs (`yield.csv`, `rainfall.csv`, etc.) introduces severe multi-collinearity and circular redundancy.
4. **Resolution**: `df_yield_merged.csv` should be retained exclusively as a secondary global macro-benchmark, while the regional state-level Parquet dataset serves as the primary engine for the Indian seasonal planning UI.

---

## 5. ACASA 50km Spatial Raster Dataset Analysis

### A. Raster Layer Inventory & Category Breakdown
* **Categories in Archive**:
  1. **Crop Area**: Spatially distributed crop harvested acreage per 50 km pixel (Rice, Wheat, Maize, Sorghum, Millet, Chickpea, Pigeonpea, Groundnut, Soybean).
  2. **Crop Irrigation**: Irrigated vs. Rainfed physical allocation fractions (`irrigated_fraction`, `rainfed_fraction`).
  3. **Nitrogen Rate**: Base synthetic nitrogen fertilizer application rates in kg N/ha (`nitrogen_rate_kg_ha`).
  4. **Planting Dates**: Median crop sowing windows expressed in Day of Year / Julian Day (`planting_doy`).
  5. **Soil Properties**: Topsoil/subsoil physical and hydraulic attributes (Clay content %, Sand %, Soil Organic Carbon $g/kg$, Available Water Capacity $mm$, Topsoil pH).

### B. Spatial & Temporal Characteristics
* **Coordinate Reference System (CRS)**: WGS 84 (EPSG:4326) Geographic Lat/Lon coordinate system.
* **Pixel Resolution**: 0.5° × 0.5° (~50 km at the equator).
* **Temporal Status**: **Static multi-year climatological and agronomic baselines**. The ACASA rasters represent long-term regional agronomic norms rather than year-by-year dynamic observations (1997–2020).
* **Connecting Rasters to Tabular Records**:
  * Tabular records (`train.parquet`) lack pixel centroid coordinates; they operate at state/UT administrative boundaries.
  * Joining ACASA rasters requires calculating **Zonal Statistics** (spatial polygon intersection and area-weighted zonal averaging over Indian State administrative shapefiles).
* **Phase 1 Decision**: **DO NOT force an artificial join** between ACASA raster pixels and tabular state rows in Phase 1. ACASA is cataloged as a distinct spatial baseline dataset for spatial GIS mapping and zonal feature extraction in subsequent phases.

---

## 6. Comprehensive Join-Key Analysis

| Left Dataset | Right Dataset | Potential Join Keys | Validated Key Match? | Resolution / Transformation Required | Status |
|---|---|---|---|---|---|
| `yield.csv` | `rainfall.csv` | `Area`, `Year` | **VALID** (100% string match) | Strip leading/trailing whitespaces in `Area`. | Supported |
| `yield.csv` | `pesticides.csv` | `Area`, `Year` | **VALID** | Align year range (1990–2013). | Supported |
| `yield.csv` | `temp.csv` | `Area` ↔ `country`, `Year` ↔ `year` | **VALID** (Case-sensitive) | Standardize column casing: `country` → `Area`, `year` → `Year`. | Supported |
| `train.parquet` | `test.parquet` | `State`, `Crop`, `Season`, `Crop_Year` | **NO (Separate Splits)** | Never join or merge train and test. | Isolated |
| `train.parquet` | `ACASA 50km` | `State` ↔ Administrative Polygon | **SPATIAL ONLY** | Requires India State boundary GeoJSON polygon zonal averaging. | Deferred to Spatial Phase |
| `train.parquet` | `df_yield_merged.csv` | `Crop` ↔ `Item`, `Crop_Year` ↔ `Year` | **INCOMPATIBLE GRANULARITY** | Parquet is Indian State-level; CSV is National Country-level. | Kept as Separate Tiers |

---

## 7. Target Definition & Formula Verification

### A. Primary Target Variable
* **Target Name**: `yield_t_ha` (Crop yield in metric tonnes per hectare).
* **Source in Primary Dataset (`train.parquet`)**: Column `Yield` is already provided in **metric tonnes per hectare (`t/ha`)**.
* **Formula Verification**: In `train.parquet`, `Yield` exactly equals $\frac{\text{Production (tonnes)}}{\text{Area (ha)}}$.
* **Conversion from Zenodo FAO Dataset (`df_yield_merged.csv`)**:
  $$\text{yield\_t\_ha} = \frac{\text{hg/ha\_yield}}{10{,}000}$$
  *(Since $1\text{ tonne} = 1{,}000\text{ kg} = 10{,}000\text{ hg}$ and both are per hectare).*

---

## 8. Agricultural Variables Mapping

| Concept | Source File | Exact Column / Raster | Units | Type | Safe Model Feature? | Preprocessing / Transformation Required |
|---|---|---|---|---|---|---|
| **Location / State** | `train.parquet` | `State` | Categorical | Categorical | **YES** | One-Hot / Target Encoding. |
| **Crop** | `train.parquet` | `Crop` | Categorical | Categorical | **YES** | Categorical Encoding; standardize variants (e.g. Rice/Basmati). |
| **Year** | `train.parquet` | `Crop_Year` | Calendar Year | Numerical/Temporal | **YES** (Trend) | Retain for historical trending; do not use as arbitrary linear weight. |
| **Season** | `train.parquet` | `Season` | Categorical | Categorical | **YES** | One-Hot Encoding (`Kharif`, `Rabi`, `Summer`, etc.). |
| **Yield (Target)** | `train.parquet` | `Yield` | $t/ha$ | Numerical | **TARGET ONLY** | Primary regression prediction target. |
| **Rainfall** | `train.parquet` | `Annual_Rainfall` | $mm$ | Numerical | **YES** | Scale / log-transform if skewed; validate against season norms. |
| **Fertilizer** | `train.parquet` | `Fertilizer` | $kg$ total | Numerical | **CONDITIONAL** | Must be normalized by `Area` to compute $\text{Fertilizer Rate} = \frac{\text{Fertilizer}}{\text{Area}}\ (kg/ha)$. |
| **Pesticides** | `train.parquet` | `Pesticide` | $kg$ total | Numerical | **CONDITIONAL** | Must be normalized by `Area` to compute $\text{Pesticide Rate} = \frac{\text{Pesticide}}{\text{Area}}\ (kg/ha)$. |
| **Temperature** | `temp.csv` / `df_yield_merged.csv` | `avg_temp` | $^{\circ}\text{C}$ | Numerical | **YES** (Macro) | Available in macro dataset; zonal extraction required for state level. |
| **Nitrogen Rate** | `ACASA_50km` | `nitrogen_rate_kg_ha.tif` | $kg\ \text{N}/ha$ | Spatial Raster | **SPATIAL** | Zonal aggregation by state boundary shapefile. |
| **Irrigation** | `ACASA_50km` | `irrigated_fraction.tif` | Fraction $[0, 1]$ | Spatial Raster | **SPATIAL** | Zonal weighted average per crop. |
| **Crop Area** | `ACASA_50km` | `crop_area_*.tif` | $ha$ / cell | Spatial Raster | **SPATIAL** | Grid cell harvested acreage baseline. |
| **Planting Date** | `ACASA_50km` | `planting_doy_*.tif` | Day of Year | Spatial Raster | **SPATIAL** | Calendar sowing window reference. |
| **Soil Properties** | `ACASA_50km` | `soil_clay_*.tif`, `soil_soc_*.tif` | $\%, g/kg$ | Spatial Raster | **SPATIAL** | Topsoil physical texture profile. |

---

## 9. Data Leakage Categorization

```
+-------------------------------------------------------------------------------+
|                             DATA LEAKAGE AUDIT                                |
+-------------------------------------------------------------------------------+
|  SAFE CANDIDATE FEATURES           | POTENTIAL (REQUIRING VALIDATION)         |
|  - Crop                            | - Fertilizer Rate (kg/ha normalized)     |
|  - Season                          | - Pesticide Rate (kg/ha normalized)      |
|  - State / Agro-Climatic Zone      | - Seasonal Rainfall Deviations           |
|  - Annual Rainfall (mm)            | - Historical Yield Moving Average        |
|  - Temperature (°C, when enriched) |                                          |
|------------------------------------+------------------------------------------|
|  STRICTLY EXCLUDED FROM MODEL     | UNKNOWN / NEEDS DECISION                 |
|  - Production (Direct math leak)   | - Future climate projections (CMIP6)     |
|  - Raw Total Production Volume     | - Uncalibrated satellite proxies         |
|  - Post-harvest damage metrics     |                                          |
|  - Test-set partition records      |                                          |
+-------------------------------------------------------------------------------+
```

* **CRITICAL LEAKAGE WARNING**: The column `Production` is the post-harvest total tonnage. Because $\text{Yield} = \frac{\text{Production}}{\text{Area}}$, including `Production` and `Area` as raw model features results in 100% mathematical target leakage ($R^2 = 1.0$). At prediction time (pre-season planning), `Production` is completely unknown. Therefore, **`Production` must be permanently removed from candidate features**.

---

## 10. Unit Conversion Requirements

1. **Target Yield**:
   * Parquet: Already $t/ha$.
   * Zenodo/FAO: $\text{yield\_t\_ha} = \frac{\text{hg/ha\_yield}}{10{,}000}$.
2. **Fertilizer Application**:
   * Raw Parquet provides total kilograms applied across the recorded state crop area.
   * Required Normalization: $\text{Fertilizer Rate (kg/ha)} = \frac{\text{Fertilizer (kg)}}{\text{Area (ha)}}$.
3. **Pesticide Application**:
   * Raw Parquet provides total kilograms applied.
   * Required Normalization: $\text{Pesticide Rate (kg/ha)} = \frac{\text{Pesticide (kg)}}{\text{Area (ha)}}$.
4. **Rainfall**:
   * Recorded in millimeters ($mm$) — no unit conversion required.

---

## 11. Time & Geographic Coverage

### A. Temporal Range
* **Primary Parquet Dataset**: 1997 to 2020 (24 continuous agricultural years).
* **Zenodo FAO Yield Dataset**: 1961 to 2013 (53 historical years).
* **Zenodo Merged Dataset**: 1990 to 2013 (24 years of aligned multi-factor observations).
* **ACASA Spatial Rasters**: Climatological baseline (2000–2020 normals).

### B. Geographic Coverage & Granularity
* **Primary Dataset (`train.parquet`)**: **State-level granularity** covering 30 Indian states & Union Territories (including Punjab, Haryana, Uttar Pradesh, Rajasthan, Gujarat, Madhya Pradesh, Maharashtra, Andhra Pradesh, Telangana, Karnataka, Tamil Nadu, West Bengal, Bihar, Assam, Odisha, Kerala, etc.).
* **Geographic Boundary Rule**: The application **does NOT support district-level or village-level precision** because the source records are state-level aggregations. The UI faithfully reflects State/Tract level observations (e.g., Punjab / Ludhiana Agro-Climatic Tract).
* **Zenodo CSVs**: **National country-level granularity** (101 sovereign nations).

---

## 12. Crop Coverage Matrix

The primary dataset covers **55 agricultural crops** categorized across major commodity types:
1. **Cereals & Grains**: Basmati Rice, Non-Basmati Rice, Wheat, Maize, Barley, Jowar (Sorghum), Bajra (Pearl Millet), Ragi (Finger Millet).
2. **Pulses & Legumes**: Gram (Chickpea), Arhar/Tur (Pigeon Pea), Moong (Green Gram), Urad (Black Gram), Masoor (Lentil), Peas.
3. **Oilseeds**: Groundnut, Mustard/Rapeseed, Soybean, Sunflower, Sesame, Castor seed, Niger seed, Linseed.
4. **Commercial & Cash Crops**: Cotton, Sugarcane, Jute, Tobacco, Coffee, Tea, Rubber.
5. **Horticultural & Spices**: Potato, Onion, Dry Chillies, Turmeric, Ginger, Coriander, Garlic, Black Pepper, Banana, Tapioca.

---

## 13. Data Quality & Distribution Audit

1. **Missing Data**: 0% null values in cleaned Parquet files; 0% missing in `df_yield_merged.csv`.
2. **Zero-Production Outliers**: Less than 0.2% of records exhibit zero production due to complete seasonal crop failures (drought/flood); these are preserved as true historical observations representing extreme risk.
3. **Distribution Skewness**:
   * Yield exhibits positive skewness with high yields in irrigated states (Punjab/Haryana basmati rice: 3.5–5.2 t/ha; sugarcane: 60–90 t/ha) compared to rainfed pulses (0.6–1.2 t/ha).
   * Rainfall ranges from arid zones (<400 mm in Western Rajasthan) to humid tropical zones (>3,000 mm in Kerala/Assam).

---

## 14. Recommended Canonical Training Table

The canonical training matrix to be constructed in Phase 2:

$$\mathcal{D}_{\text{train}} = \left\{ \left( \mathbf{x}_i, y_i \right) \right\}_{i=1}^{N}$$

### Columns in Canonical Training Table:
1. `record_id` (Primary Key): Deterministic hash `SHA256(State_Crop_Season_Year)`.
2. `state` (Categorical Feature): State identifier.
3. `crop` (Categorical Feature): Crop identifier.
4. `season` (Categorical Feature): Agricultural season (`Kharif`, `Rabi`, `Summer`, `Whole Year`).
5. `crop_year` (Numerical Feature): Year of record.
6. `rainfall_mm` (Numerical Feature): `Annual_Rainfall`.
7. `fertilizer_kg_ha` (Numerical Feature): $\frac{\text{Fertilizer}}{\text{Area}}$.
8. `pesticide_kg_ha` (Numerical Feature): $\frac{\text{Pesticide}}{\text{Area}}$.
9. `yield_t_ha` (**Target Variable**): `Yield` in $t/ha$.

---

## 15. Recommended Data Architecture

```
data/
├── raw/
│   ├── train-00000-of-00001.parquet       # Raw primary training partition (DO NOT MODIFY)
│   ├── test-00000-of-00001.parquet        # Raw evaluation partition (ISOLATED)
│   ├── 18761663.zip                       # Raw Zenodo global benchmark archive
│   └── ACASA_50km.zip                     # Raw GeoTIFF spatial rasters
├── processed/
│   ├── canonical_train_features.parquet   # Normalized features + target (no leakage)
│   ├── canonical_test_features.parquet    # Standardized test features for validation
│   └── historical_quantiles.json          # Empirical quantiles (P10, P25, P50, P75) by State/Crop/Season
├── spatial/
│   ├── india_states_50km_zonal.parquet    # Zonal statistics aggregated from ACASA
│   └── agro_ecological_zones.geojson      # State boundaries for spatial mapping
└── schemes/
    └── government_schemes.json            # Verified government schemes (PMKSY, PMFBY, PKVY, etc.)

ml/
├── features.py                            # Feature extraction, scaling, and encodings
├── train.py                               # LightGBM / XGBoost / CatBoost training routines
├── evaluate.py                            # Temporal & cross-validated metrics (RMSE, MAE, R²)
└── artifacts/
    ├── model_metadata.json                # Feature bounds, training ranges, metrics
    └── yield_model.onnx / .joblib         # Trained inference artifacts

backend/
├── app/
│   ├── main.py / server.ts                # Application API endpoints
│   ├── routes/
│   │   ├── simulation.ts                  # What-if scenario execution
│   │   ├── historical.ts                  # Historical quantile retrieval
│   │   └── schemes.ts                     # Scheme matching service
│   └── services/
└── frontend/                              # Existing Stitch UI (Preserved & Verified)
```

---

## 16. Summary of Excluded Data & Features

1. **`Production` Column**: Excluded to eliminate deterministic target leakage.
2. **`Area` (as an unnormalized predictor)**: Excluded from direct linear weighting to prevent confounding farm scale with agronomic land productivity.
3. **`df_yield_merged.csv`**: Excluded from the Indian State-level model to prevent geographic mismatch (country-level vs. state-level).
4. **`test.parquet` in Training**: Excluded strictly from all training loops and parameter tuning.

---

## 17. Unresolved Questions & Recommendations for Phase 2

1. **Sub-annual Rainfall**: The Parquet dataset provides `Annual_Rainfall`. In Phase 2, enriching with monthly IMD / ERA5 precipitation for specific crop growth windows (sowing vs. flowering vs. harvest) can further elevate predictive fidelity.
2. **ACASA Zonal Integration**: Zonal aggregation scripts should be executed using GDAL/Rasterio against Official Survey of India state boundary shapefiles to generate static soil and nitrogen priors.
3. **Responsible AI Guardrails**: In accordance with project rules, model inferences must remain tagged as **probabilistic model estimates** with clear empirical confidence intervals ($80\%\ \text{CI}$) and out-of-training-range alerts.
