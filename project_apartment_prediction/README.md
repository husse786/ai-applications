# Model Iterations Documentation
## Task: Apartment Price Prediction (Regression)

---

## Summary of Iterative Process

| Iteration | Objective | Key Changes | Models Used | CV Mean RMSE (CHF) | CV Mean R² | Change in Performance | Fit Diagnosis |
|-----------|-----------|-------------|-------------|---------------------|------------|-----------------------|---------------|
| **1** | Build baseline model | - Removed missing values<br>- Removed duplicates<br>- Removed price outliers (< 750 or > 8000 CHF)<br>- Selected 13 features from weeks 1–2<br>- No feature scaling<br>- 5-fold CV | Linear Regression (default)<br>Random Forest (n_estimators=100) | 696.4 (LR)<br>699.5 (RF) | 0.4947 (LR)<br>0.5184 (RF) | Baseline | ☑ Overfitting ☐ Underfitting ☐ Good Fit |
| **2** | Improve generalization + add new feature | - All steps from Iteration 1 retained<br>- Added `dist_to_zhb` (haversine distance to Zürich HB)<br>- Added `avg_price_postal_rooms_area` (local market anchor)<br>- StandardScaler in Pipeline for MLP<br>- Hyperparameter tuning<br>- 5-fold CV | Tuned Random Forest (n_estimators=500, max_depth=15)<br>MLP Neural Network (layers=64,32, relu, adam) | 416.5 (RF)<br>617.3 (MLP) | 0.8244 (RF)<br>0.4856 (MLP) | -283 CHF / +0.31 R² improvement | ☐ Overfitting ☐ Underfitting ☑ Good Fit |

---

## Preprocessing Steps

- Removed rows with missing values (`dropna`)
- Removed duplicate rows
- Removed price outliers: apartments below 750 CHF/month and above 8000 CHF/month excluded
- Selected relevant numeric and binary features
- Applied `StandardScaler` inside a `Pipeline` for the MLP model only (Random Forest does not require scaling)

---

## Models Used

| Model | Iteration | Hyperparameters |
|-------|-----------|-----------------|
| Linear Regression | 1 | Default (sklearn) |
| Random Forest | 1 | n_estimators=100, random_state=42 |
| Random Forest (tuned) | 2 | n_estimators=500, max_depth=15, random_state=42 |
| MLP Neural Network | 2 | hidden_layer_sizes=(64, 32), activation=relu, solver=adam, max_iter=500 |

---

## Evaluation Method

**5-fold cross-validation** was used for all models across both iterations.

- The dataset (804 apartments) is split into 5 equal folds
- Each fold is used once as validation, the remaining 4 for training
- Metrics reported: **Mean RMSE (CHF)** and **Mean R²** across all 5 folds
- This avoids overfitting to a single train/test split and gives a reliable estimate of real-world performance

---

## Notes

**Created Features:**
- `room_per_m2` — rooms divided by area (spaciousness per room)
- `luxurious` — binary flag from listing description keywords (LOFT, POOL, ATTIKA, etc.)
- `temporary` — binary flag for temporary rentals
- `furnished` — binary flag for furnished apartments
- `area_cat_ecoded` — area category encoded: small / medium / large
- `zurich_city` — binary: 1 if apartment is in the city of Zürich
- `avg_price_postal_rooms_area` — mean rent for same postal code + room count + area bracket
- `dist_to_zhb` *(new)* — haversine distance in km from apartment to Zürich Hauptbahnhof (47.3782°N, 8.5403°E)

**Final Selected Features:**
- `rooms`, `area`
- `pop`, `pop_dens`, `frg_pct`, `emp`, `tax_income`
- `room_per_m2`, `luxurious`, `temporary`, `furnished`
- `area_cat_ecoded`, `zurich_city`
- `avg_price_postal_rooms_area`
- `dist_to_zhb`

**Final Selected Model:**
Tuned Random Forest (Iteration 2) — `n_estimators=500`, `max_depth=15`
- CV Mean RMSE: **416.5 CHF**
- CV Mean R²: **0.8244**
- Train R²: 0.9709

**Reason for Selection:**
Chosen based on lowest CV RMSE and highest R² across all iterations. The addition of `dist_to_zhb` and `avg_price_postal_rooms_area` in Iteration 2 drove the largest performance gains — R² improved from 0.52 to 0.82 and RMSE dropped by 283 CHF compared to the baseline.
