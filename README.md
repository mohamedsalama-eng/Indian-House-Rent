#  Indian House Rent Prediction

An end-to-end machine learning project to predict monthly rent (₹) for residential properties across 6 major Indian cities.

---

##  Project Overview

| | |
|---|---|
| **Type** | Supervised Regression |
| **Target** | Monthly Rent (₹) |
| **Best Model** | Random Forest (tuned) |
| **CV RMSE** | 0.3772 (log scale) |
| **Test RMSE** | ₹8,994 |
| **R²** | 0.789 |
| **Dataset size** | ~4,700 rows |

---

##  Project Structure

```
├── Indian_House_Rent_Dataset.csv       # Raw dataset
├── Indian_House_Rent_FINAL_v2.ipynb    # Fully annotated notebook
├── indian_house_rent_model.pkl         # Saved final pipeline
└── README.md                           # This file
```

---

##  Dataset

| Column | Description | Type |
|---|---|---|
| `bhk` | Bedrooms, Hall, Kitchen count | Numeric (noisy) |
| `rent` | Monthly rent in ₹ — **target** | Numeric |
| `size` | Property size in sq ft | Numeric (noisy) |
| `floor` | Floor info e.g. "3 out of 10" | String (parsed) |
| `area_type` | Super Area / Carpet Area / Built Area | Categorical |
| `area_locality` | Neighbourhood name | High cardinality (2012 unique) |
| `city` | City name | Categorical (noisy) |
| `furnishing_status` | Furnished / Semi-Furnished / Unfurnished | Ordinal |
| `tenant_preferred` | Bachelors / Family / Bachelors&Family | Categorical (noisy) |
| `bathroom` | Number of bathrooms | Numeric (noisy) |
| `point_of_contact` | Contact Agent / Contact Owner | Categorical |

---

##  Pipeline Overview

```
Raw Data
   │
   ├── 1. Data Cleaning
   │      ├── Strip noisy characters (bhk, size, bathroom)
   │      ├── Parse floor string → floor_number + total_floors
   │      ├── Fuzzy match city names (thefuzz library)
   │      ├── Fuzzy match tenant_preferred categories
   │      └── Cast to appropriate dtypes (category, numeric, datetime)
   │
   ├── 2. EDA
   │      ├── Distribution analysis (rent heavily right-skewed)
   │      ├── Correlation heatmap (before and after feature engineering)
   │      └── Mean rent by categorical features
   │
   ├── 3. Feature Engineering
   │      ├── post_month, days_since_posted (from posted_on)
   │      ├── size_per_bhk (comfort score)
   │      ├── bathroom_to_bhk (luxury ratio)
   │      ├── total_floors (building height proxy)
   │      └── floor_ratio = floor_number / total_floors
   │
   ├── 4. Train / Test Split (80/20, random_state=42)
   │      └── Split on RAW rent — before capping and log transform
   │
   ├── 5. Outlier Capping (AFTER split — no leakage)
   │      ├── IQR method on size  → X_train / X_test
   │      ├── IQR method on rent  → y_train_raw / y_test_raw (raw rupees)
   │      └── Log transform AFTER capping → y_train / y_test
   │
   ├── 6. Preprocessing Pipeline (sklearn ColumnTransformer)
   │      ├── Numerical  → SimpleImputer(median) + StandardScaler
   │      ├── Ordinal    → SimpleImputer(most_frequent) + OrdinalEncoder
   │      ├── Nominal    → SimpleImputer(most_frequent) + OneHotEncoder(drop='first')
   │      └── Locality   → SimpleImputer(most_frequent) + TargetEncoder(smooth='auto')
   │
   ├── 7. Model Comparison (CV only — test set NOT touched here)
   │      ├── Linear Regression
   │      ├── Decision Tree
   │      ├── Random Forest ← winner
   │      ├── SVR
   │      └── XGBoost
   │
   ├── 8. Hyperparameter Tuning (GridSearchCV on full pipeline)
   │      └── Best: max_depth=20, max_features='sqrt',
   │                min_samples_split=5, n_estimators=200
   │
   └── 9. Final Evaluation (test set used exactly once — here)
          ├── RMSE, MAE, R² in actual rupees
          └── Top 10 worst predictions analysis
```

---

##  Key Design Decisions

### Why split BEFORE capping and log transform?
The correct order is **Split → Cap (raw rupees) → Log**. If you log first then cap, the IQR of log values is completely different from the IQR of rupees — extreme values like ₹3,500,000 become `log(3,500,000) = 15.07` which looks like a slightly high log value and never gets clipped. This caused Test RMSE to explode to ₹130,000 in an earlier version. Capping must always happen in the original units before any transformation.

### Why IQR capping AFTER the train/test split?
IQR boundaries (Q1, Q3) are computed from data. If computed on the full dataset before splitting, test set values influence the thresholds — this is **data leakage**. The fix: compute thresholds from training data only and apply the same bounds to the test set.

### Why log transform on `rent` only?
`rent` has skewness of 3.6 — log1p reduces it to 0.84. `size` has skewness of 1.42 — log1p overcorrects to -1.07 (flips the skew direction). So we kept `size` as-is and only transformed `rent`. Predictions are back-transformed with `np.expm1()` for evaluation in rupees.

### Why TargetEncoder for `area_locality`?
2012 unique localities — OneHotEncoding would create 2000+ sparse columns. TargetEncoder replaces each locality with its mean rent, compressing to 1 meaningful numeric column. `smooth='auto'` blends rare localities toward the global mean to prevent overfitting. Fitted only on training folds inside CV — no leakage.

### Why SimpleImputer over custom ProbabilityImputer?
A `ProbabilityImputer` (fills NaN by sampling from the column's frequency distribution) was explored but replaced because:
- Introduces randomness → results not reproducible across runs
- Lacks `get_feature_names_out()` → breaks sklearn pipeline compatibility
- `ColumnTransformer` passes numpy arrays — column names are lost, breaking `X[col]` access
- For low-cardinality columns (3–5 values), barely differs from mode imputation in practice

The custom class is kept in the notebook (commented out) as a learning reference.

### Why `~(df['bhk'] > 5)` instead of `df['bhk'] <= 5`?
`df['bhk'] <= 5` returns `False` for NaN → silently drops 1,241 NaN rows (26% of data). `~(df['bhk'] > 5)` returns `True` for NaN → keeps them for the pipeline imputer to handle. This recovered 1,241 rows.

### Why is the test set used only once?
During model comparison and hyperparameter tuning, only CV RMSE is used to make decisions. The test set is touched exactly once — in the final evaluation cell. Using test set metrics to select models or tune hyperparameters causes optimistic bias: you end up selecting the model that fits that specific test split, not the one that generalises best.

### Encoding choices

| Column | Encoder | Why |
|---|---|---|
| `furnishing_status` | OrdinalEncoder | Natural order: Unfurnished < Semi-Furnished < Furnished |
| `city`, `area_type`, `tenant_preferred`, `point_of_contact` | OneHotEncoder | Nominal — no natural order between categories |
| `area_locality` | TargetEncoder | Too high cardinality for OHE |

---

##  Results

### Model Comparison (CV only — no test set)

| Model | CV RMSE (log) | CV STD |
|---|---|---|
| Linear Regression | 0.4235 | 0.0110 |
| Decision Tree | 0.5469 | 0.0021 |
| **Random Forest** | **0.3918** | **0.0117** |
| SVR | 0.3968 | 0.0116 |
| XGBoost | 0.4056 | 0.0108 |

### Final Test Set Evaluation — Tuned Random Forest

| Metric | Value |
|---|---|
| **RMSE** | **₹8,994** |
| **MAE** | **₹5,888** |
| **R²** | **0.789** |
| Best params | `max_depth=20`, `max_features='sqrt'`, `min_samples_split=5`, `n_estimators=200` |

> R² of 0.789 means the model explains 78.9% of rent variance — solid for a noisy
> real estate dataset spanning 6 cities with median rent ₹16,000 and std ₹78,000.

---

##  Business Value

**For a real estate app (MagicBricks, 99acres, NoBroker):**

- **Landlords** get an instant suggested rent when listing a new property instead of guessing
- **Tenants** see a fair price indicator (🟢 Fair / 🔴 Overpriced / 🔵 Great Deal) on every listing
- **Platform trust** increases by automatically flagging suspicious listings before they go live
- **Investors** can estimate expected rental yield on a property before purchasing

> On average the model predicts rent within ₹5,888 (MAE) — useful enough for
> pricing guidance and fair-value detection across the majority of the market.

---

##  Feature Importances (Top 10)

| Feature | Importance | Insight |
|---|---|---|
| `bathroom` | 0.145 | Strongest predictor — stronger than BHK |
| `city_mumbai` | 0.123 | Mumbai is a completely separate market |
| `point_of_contact_Contact Owner` | 0.120 | Agents = luxury, owners = budget |
| `bhk` | 0.118 | More bedrooms = higher rent |
| `total_floors` | 0.107 | Taller building = newer/luxury |
| `size` | 0.106 | Property size |
| `area_locality` | 0.100 | TargetEncoder successfully captured geographic signal |
| `size_per_bhk` | 0.032 | Engineered feature contributed ✅ |
| `days_since_posted` | 0.028 | Listing urgency |
| `floor_ratio` | 0.022 | Relative floor position |

---

##  Known Limitations

- **IQR clipping wall at ₹65K** — model cannot predict above this threshold. Properties originally worth ₹150K–₹300K are clipped to ₹65K in training, causing large errors for the luxury segment
- **Missing features** — building age, metro distance, amenities, and exact neighbourhood tier would significantly improve accuracy
- **Small dataset** — ~4,700 rows with ₹78K std is fundamentally challenging for any model
- **City segmentation** — a separate model per city (especially Mumbai) would likely improve results significantly

---

##  Usage

```python
import joblib
import numpy as np
import pandas as pd

# Load the saved pipeline
model = joblib.load('indian_house_rent_model.pkl')

# Prepare a new property (raw — no preprocessing needed)
new_property = pd.DataFrame([{
    'size': 1200,
    'bhk': 3,
    'bathroom': 2,
    'days_since_posted': 15,
    'post_month': 5,
    'size_per_bhk': 400,        # size / bhk
    'bathroom_to_bhk': 0.67,    # bathroom / bhk
    'total_floors': 10,
    'floor_ratio': 0.6,         # floor_number / total_floors
    'furnishing_status': 'Semi-Furnished',
    'city': 'mumbai',
    'area_type': 'super area',
    'tenant_preferred': 'Family',
    'point_of_contact': 'Contact Agent',
    'area_locality': 'bandra west'
}])

# Predict — back-transform from log scale to actual rupees
predicted_rent = np.expm1(model.predict(new_property))
print(f"Predicted monthly rent: ₹{predicted_rent[0]:,.0f}")
```

---

##  Requirements

```
pandas
numpy
scikit-learn >= 1.3   # TargetEncoder added in 1.3
xgboost
thefuzz
python-Levenshtein
seaborn
matplotlib
joblib
```

Install:
```bash
pip install pandas numpy scikit-learn xgboost thefuzz python-Levenshtein seaborn matplotlib joblib
```
