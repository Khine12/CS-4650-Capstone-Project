# Capstone report outline (matches the Canvas instructions)

## 1. Team (very important)
One Canvas submission for the group; the professor copies the score to everyone named here.
- Khine Zar Hein
- Devin Davis
- Wesley Trinh
- Jacqueline Ninh

Add one line per person on what they did (see the README team table).

## 2. Dataset / competition
Kaggle Playground Series S6E10, Predicting Airline Satisfaction. Synthetic data generated from a real airline survey dataset.
699,635 training rows, 299,844 test rows, 21 features. Binary target, scored by ROC AUC. Ran Sep 30 – Oct 31, 2026; we used Late Submission.
Participation (Kaggle Overview page): 3,734 teams, 3,901 participants, 44,249 submissions. Our ranks use the downloaded private leaderboard (3,735 rows), minus 20 broken entries scoring 0.5 or below: 3,715 teams.

## 3. Approach and cleaning  (write-up: Jacqueline)
- Filled 292 missing arrival delays with departure delay (correlation 0.84)
- log1p on delays and flight distance (heavily skewed)
- One-hot encoded Gender, Customer Type, Type of Travel, Class
- Ratings 0–5: treated 0 as "not applicable" by one-hot encoding (logistic AUC 0.921 → 0.948)
- MapReduce job: satisfaction rate per Class × Type of Travel, with [sum, count] partial results so the combiner is correct. Result: Business class + business travel 73.3% satisfied (338,272 passengers); Eco + personal travel 9.7% (185,792); every personal-travel group is under 12%
- Fixed validation split for fair comparison

## 4. Results and visualizations  (Khine; linear regression: Devin; logistic tuning: Wesley)
- Table of models and validation AUC
- Linear regression (Devin): arrival delay from departure delay, slope / R² / RMSE, scatter plot with fitted line
- Logistic regression (Wesley): C and L1/L2 table, top coefficients chart
- Confusion matrix, precision, recall for the best model
- Chart: our scores vs leaderboard (`figures/baseline_vs_leaderboard.png` = validation, v1; `figures/kaggle_scores_vs_leaderboard.png` = private scores, v1 + v3)
- Clustering (`notebooks/03_khine_clustering.ipynb`): K-means on 13 ratings + age, distance, delay, class, travel type, loyalty (scaled; satisfaction not used). k = 7, silhouette 0.160 (low: groups overlap)
  - Two satisfied clusters (39% of passengers, ~90% satisfied): business travellers in Business class with good on-board ratings, even when wifi/booking/gate ratings are low
  - Other five clusters 11–22% satisfied: personal travellers in Economy (unsatisfied even with good ratings), disloyal young business travellers, passengers with poor seat/food/entertainment/cleanliness
  - Charts: `figures/kmeans_choose_k.png`, `figures/kmeans_cluster_ratings.png`, `figures/kmeans_clusters_satisfaction.png`
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
The starter code (cleaning module, baseline notebook, MapReduce job) and notebooks 01–03 (feature/boosting experiment, LightGBM/CatBoost ensemble, clustering) were written with AI assistance (Claude). Khine directed the work, ran the notebooks, checked the results and made the Kaggle submissions. Each teammate adds a line here for any AI help on their own part.
