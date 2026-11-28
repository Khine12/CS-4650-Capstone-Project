# Capstone report outline (matches the Canvas instructions)

## 1. Team (very important)
Full names of all four members. One Canvas submission for the group.

## 2. Dataset / competition
Kaggle Playground Series S6E10, Predicting Airline Satisfaction. Synthetic data generated from a real airline survey dataset.
699,635 training rows, 299,844 test rows, 21 features. Binary target, scored by ROC AUC. Ran Sep 30 – Oct 31, 2026; we used Late Submission.

## 3. Approach and cleaning  (Teammate A)
- Filled 292 missing arrival delays with departure delay (correlation 0.84)
- log1p on delays and flight distance (heavily skewed)
- One-hot encoded Gender, Customer Type, Type of Travel, Class
- Ratings 0–5: treated 0 as "not applicable" by one-hot encoding (logistic AUC 0.921 → 0.948)
- MapReduce job: satisfaction rate per Class × Type of Travel
- Fixed validation split for fair comparison

## 4. Results and visualizations  (Teammates B + C)
- Table of models and validation AUC
- Confusion matrix, precision, recall for the best model
- Chart: our scores vs leaderboard (`figures/baseline_vs_leaderboard.png`)
- Clustering: K-means passenger groups, silhouette score, what each group looks like
- Feature importance / strongest features: `figures/feature_importance_v2.png` (permutation importance; top 5: Type of Travel, Inflight wifi, digital-experience average, Customer Type, Online boarding)

## 5. Comparison with other submissions  (Khine)
- Late-submission scores from Kaggle (screenshots)
- First late submission (gradient boosting): public 0.95761, private 0.95908
- Would rank about 2,860 of 3,715 teams on the final private leaderboard (best 0.96318, median 0.96081)
- Validation AUC matched public score within 0.00003
- Best late submission (v3 ensemble: 0.7 LightGBM + 0.2 CatBoost + 0.1 HistGradientBoosting): public 0.95841, private 0.95957
- v3 would rank about 2,190 of 3,715 (up about 670 places from v1); validation 0.95824 matched public within 0.00017
- Screenshots of the Kaggle Submissions page: `figures/kaggle_submissions_2026-11-27.png` (v1), `figures/kaggle_submissions_v3_2026-11-27.png` (v1 + v3)

## 6. What worked, what didn't  (everyone, one paragraph each)
- Khine, didn't work: extra features (rating mean/min/spread, count of 0 ratings, total delay) and tuning gradient boosting lowered validation AUC slightly (best 0.95745 vs baseline 0.95758). Boosting already finds this information on its own (`notebooks/01_khine_features_boosting.ipynb`)
- Khine, didn't work: blending boosting with logistic regression gave no gain (best weight was boosting only)
- Khine, worked: switching model instead of tuning. LightGBM alone 0.95819 (+0.00061); ensemble of 0.7 LightGBM + 0.2 CatBoost + 0.1 HistGradientBoosting (rank average) 0.95824 (+0.00066) (`notebooks/02_khine_catboost_lightgbm.ipynb`)
- Khine, didn't work: CatBoost alone tied the baseline (0.95755) and was slow (about 25 min to train, 14 min to refit); ratings as CatBoost categories was slower still and scored lower on a 20k-row test

## AI assistance (check course policy)
Starter code (cleaning module, baseline notebook, MapReduce job) was created with AI assistance; the team reviewed, ran, and extended it.
