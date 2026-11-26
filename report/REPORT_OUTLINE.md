# Capstone report outline (matches the Canvas instructions)

## 1. Team (very important)
Full names of all four members. One Canvas submission for the group.

## 2. Dataset / competition
Kaggle Playground Series S6E10, Predicting Airline Satisfaction. Synthetic data generated from a real airline survey dataset.
699,635 training rows, 299,844 test rows, 22 features. Binary target, scored by ROC AUC. Ran Sep 30 – Oct 31, 2026; we used Late Submission.

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
- Feature importance / strongest features

## 5. Comparison with other submissions  (Khine)
- Late-submission scores from Kaggle (screenshots)
- Rank we would have had out of 3,715 teams; best 0.96318, median 0.96081

## 6. What worked, what didn't  (everyone, one paragraph each)
