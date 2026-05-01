#  Indian House Rent Prediction

An end-to-end machine learning project to predict monthly rent (₹) for residential properties across 6 major Indian cities.

---

##  Project Overview

| | |
|---|---|
| **Type** | Supervised Regression |
| **Target** | Monthly Rent (₹) |
| **Best Model** | Random Forest (tuned) |
| **Test MAE** | ~₹11,000 |
| **R²** | ~0.66 |
| **Dataset size** | ~4,700 rows |

---

##  Project Structure

```
├── Indian_House_Rent_Dataset.csv     # Raw dataset
├── Indian_House_Rent_FINAL.ipynb     # Fully annotated notebook
├── indian_house_rent_model.pkl       # Saved final pipeline
└── README.md                         # This file
```

---

##  Dataset

The dataset contains Indian residential property listings with the following columns:

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
   │      ├── Correlation heatmap
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
   │
   ├── 5. Outlier Capping (AFTER split — no leakage)
   │      └── IQR method: clip at Q1-1.5*IQR / Q3+1.5*IQR
   │          using train statistics applied to both sets
   │
   ├── 6. Preprocessing Pipeline (sklearn ColumnTransformer)
   │      ├── Numerical  → SimpleImputer(median) + StandardScaler
   │      ├── Ordinal    → SimpleImputer(most_frequent) + OrdinalEncoder
   │      ├── Nominal    → SimpleImputer(most_frequent) + OneHotEncoder(drop='first')
   │      └── Locality   → SimpleImputer(most_frequent) + TargetEncoder(smooth='auto')
   │
   ├── 7. Model Comparison (with full pipeline CV)
   │      ├── Linear Regression
   │      ├── Decision Tree
   │      ├── Random Forest ← winner
   │      ├── SVR
   │      └── XGBoost
   │
   └── 8. Hyperparameter Tuning (GridSearchCV on full pipeline)
          └── Best: max_depth=20, max_features='sqrt',
                    min_samples_split=5, n_estimators=200
```

---

##  Key Design Decisions

### Why IQR capping AFTER the train/test split?
IQR boundaries (Q1, Q3) are computed from data. If computed on the full dataset before splitting, test set values influence the thresholds — this is **data leakage**. The fix: compute from `X_train` only and apply the same bounds to `X_test`.

### Why log transform on `rent` only?
`rent` has skewness of 3.6 — log1p reduces it to 0.84. `size` has skewness of 1.42 — log1p overcorrects to -1.07 (flips direction). So we kept `size` as-is and only transformed `rent`, which becomes `rent_log` (the target `y`). Predictions are back-transformed with `np.expm1()` for evaluation in rupees.

### Why TargetEncoder for `area_locality`?
2012 unique localities — OneHotEncoding would create 2000+ sparse columns. TargetEncoder replaces each locality with its mean rent, compressing to 1 meaningful numeric column. `smooth='auto'` blends rare localities (1-2 rows) toward the global mean to prevent overfitting.

### Why SimpleImputer over a custom ProbabilityImputer?
A `ProbabilityImputer` (fills NaN by sampling from the column's frequency distribution) was explored but replaced because:
- Introduces randomness → results not reproducible across runs
- Lacks `get_feature_names_out()` → breaks sklearn pipeline compatibility
- For low-cardinality columns, barely differs from mode imputation in practice

### Why `~(df['bhk'] > 5)` instead of `df['bhk'] <= 5`?
`df['bhk'] <= 5` returns `False` for NaN → silently drops 1,241 NaN rows (26% of data). `~(df['bhk'] > 5)` returns `True` for NaN → keeps them for the pipeline imputer to handle. This recovered 1,241 rows.

### Encoding choices
| Column | Encoder | Why |
|---|---|---|
| `furnishing_status` | OrdinalEncoder | Natural order: Unfurnished < Semi-Furnished < Furnished |
| `city`, `area_type`, `tenant_preferred`, `point_of_contact` | OneHotEncoder | Nominal — no natural order between categories |
| `area_locality` | TargetEncoder | Too high cardinality for OHE |

---

##  Results

| Model | CV RMSE (log) | CV STD | Test RMSE (₹) | MAE (₹) | R² |
|---|---|---|---|---|---|
| Linear Regression | 0.4749 | 0.0223 | ₹33,367 | ₹12,830 | 0.541 |
| Decision Tree | 0.6488 | 0.0249 | ₹37,302 | ₹16,340 | 0.427 |
| **Random Forest** | **0.4641** | **0.0174** | **₹27,194** | **₹11,071** | **0.695** |
| SVR | 0.4629 | 0.0225 | ₹30,381 | ₹11,875 | 0.620 |
| XGBoost | 0.4750 | 0.0179 | ₹30,013 | ₹12,126 | 0.629 |

**Tuned Random Forest (GridSearchCV) — Final Test Results:**
- RMSE: ₹28,713
- MAE: ₹11,025
- R²: 0.6604
- Best params: `max_depth=20`, `max_features='sqrt'`, `min_samples_split=5`, `n_estimators=200`

> The high variance in Indian rental prices — median ₹16,000 vs properties
> exceeding ₹3,000,000 across 6 cities with very different market dynamics —
> makes this a fundamentally challenging regression problem. The model explains
> 66% of rent variance and predicts within ₹11,025 on average.
  
---

##  Business Value

**For a real estate app (MagicBricks, 99acres, NoBroker):**

- **Landlords** get an instant suggested rent when listing a property instead of guessing
- **Tenants** see a fair price indicator ( Fair /  Overpriced /  Great Deal) on every listing
- **Platform trust** increases by automatically flagging suspicious listings before they go live
- **Investors** can estimate expected rental yield on a property before purchasing

> On average the model predicts rent within ₹11,025 (MAE) — useful enough for
> pricing guidance and fair-value detection across the majority of the market.

##  Known Limitations

- **IQR clipping wall at ₹65K** — model cannot predict above this threshold. Properties clipped from ₹150K–₹300K appear as ₹65K in training, creating ~42K errors for luxury segment
- **Missing features** — building age, metro distance, amenities, exact neighbourhood tier would significantly improve accuracy
- **Small dataset** — ~4,700 rows with ₹78K std is fundamentally challenging for any model
- **Mumbai dominance** — a separate model per city would likely improve results significantly

---

##  Usage

```python
import numpy as np
import pandas as pd


# Prepare a new property (raw — no preprocessing needed)
new_property = pd.DataFrame([{
    'size': 1200,
    'bhk': 3,
    'bathroom': 2,
    'days_since_posted': 15,
    'post_month': 5,
    'size_per_bhk': 400,
    'bathroom_to_bhk': 0.67,
    'total_floors': 10,
    'floor_ratio': 0.6,
    'furnishing_status': 'Semi-Furnished',
    'city': 'mumbai',
    'area_type': 'super area',
    'tenant_preferred': 'Family',
    'point_of_contact': 'Contact Agent',
    'area_locality': 'bandra west'
}])

# Predict (back-transform from log scale)
predicted_rent = np.expm1(model.predict(new_property))
print(f"Predicted monthly rent: ₹{predicted_rent[0]:,.0f}")
```

---

##  Requirements

```
pandas
numpy
scikit-learn >= 1.3  # TargetEncoder added in 1.3
xgboost
thefuzz
python-Levenshtein
seaborn
matplotlib
```
