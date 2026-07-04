# AIMS-Kaggle-comp-2026
# AIMS Kaggle 2K26 - Top 50 Solution 

This repository contains my solution for the [AIMS Kaggle 2k26 Competition](https://www.kaggle.com/competitions/aims-kaggle-2k26/overview). The model achieved **Rank 46** on the private leaderboard.

## 📊 Leaderboard Performance
* **Private Leaderboard Score:** `0.88890` (Rank 46)
* **Public Leaderboard Score:** `0.89255`
* *(For context, the 1st place solution by Paras Pandita scored `0.92287`)*

## 🧠 Approach & Architecture

The final submission relies on a robust Level-2 Stacking Ensemble that mitigates both bias and variance by combining several gradient-boosting frameworks and bagging algorithms. 

### 1. Data Preprocessing
* Handled missing categorical data via `SimpleImputer` (most frequent).
* Handled missing numerical data via `KNNImputer` (k=7, distance-weighted) fit strictly on the training set to prevent data leakage.
* Harmonized one-hot encoding across both train and test splits to ensure consistent feature matrix dimensions.

### 2. Feature Engineering
I generated synthetic features based on domain logic (planetary physics) and statistical separation:
* **Physics Proxies:** Created composite metrics like `Habitability_Index`, `Atmosphere_Retention`, `Radiation_Exposure`, and `Kepler_Ratio` by mapping non-linear relationships between gravity, mass, stellar luminosity, and orbital periods.
* **Statistical Separators:** Identified classes that heavily overlapped (e.g., Class 4 vs 9, Class 3 vs 8) and created custom weighted features based on the highest variance means between those specific classes to help tree-based models split them more efficiently.

### 3. The Ensemble Model
I evaluated both soft-voting and stacking approaches. Stacking proved more effective at minimizing log loss and improving F1 scores across minority classes.

**Base Learners (Level 1):**
* `CatBoostClassifier` 
* `LGBMClassifier` (LightGBM)
* `XGBClassifier` (XGBoost)
* `GradientBoostingClassifier`
* `RandomForestClassifier`

**Meta-Learner (Level 2):**
* `LogisticRegression` acting as the final estimator, using a 3-fold cross-validation strategy on the base predictions.

