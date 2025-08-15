# Pattern Recognition Strategy (ML)
## Supervised “is this a good entry now?”

**Goal of this approach:** turn patterns into a **label** and train a classifier/regressor.

- **Label ideas:**
    - **Classification:** `1` if **forward return** over next *H* days ≥ threshold (e.g., +1.5% in 5 days), else `0`.
    - **Regression:** predict **forward return** over next *H* days, later threshold it for signals.
- **Models:** Start simple → **LogisticRegression**, **RandomForest**, **GradientBoosting** (or XGBoost if allowed).
- **Pipeline:** `ColumnSelector → Imputer → Scaler → Model` (scikit-learn Pipeline).
- **CV:** `TimeSeriesSplit(n_splits=5)` with **purged** windows (no overlapping leakage when labels use horizons).
- **Model selection:** compare simple baselines to avoid overfitting; tune only a few hyperparams.
- **Interpretation:** permutation importance; partial dependence profiles for top features.

**Outputs I'm using:** probability/score per day → **convert to trades** with simple rules (e.g., score > 0.6 enter long; stop/TP rules; max concurrent positions = 1 for MVP).


# Thsi project fetaures teh following ML techniques:

### Supervised Learning:
- Creating labels (classification or regression) based on future returns and train models to predict them.

### Classification Models:

- Logistic Regression, Random Forest, Gradient Boosting (and optionally XGBoost) to predict “good entry” points.

### Feature Engineering:

- Generating technical indicators (moving averages, RSI, etc.) as input features for the models.

## Down-the-line Additions:
### Pipeline Processing:

- Using scikit-learn Pipelines for preprocessing (imputation, scaling) and modeling.

### Time Series Cross-Validation:

- Using TimeSeriesSplit to evaluate model performance without data leakage.

### Model Interpretation:

- Techniques like permutation importance and partial dependence profiles to understand feature impact.