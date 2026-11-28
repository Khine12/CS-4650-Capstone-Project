# CS4650 Capstone: Predicting Airline Satisfaction

Kaggle Playground Series S6E10: https://www.kaggle.com/competitions/playground-series-s6e10

**Task:** predict the probability that a passenger is satisfied (`satisfaction` = True/False).
**Score:** ROC AUC (higher is better). Submit **probabilities**, not 0/1.
**Due on Canvas:** Fri Dec 11, 2026, 11:59 pm. Our target: submit Thu Dec 10.

## Team

| Name | GitHub | Role |
|---|---|---|
| Khine Zar Hein | | Lead: repo, MapReduce, Kaggle submissions, report |
| _Teammate A_ | | Cleaning + feature ideas (`src/clean.py`) |
| _Teammate B_ | | Logistic regression + random forest / boosting |
| _Teammate C_ | | Clustering + charts |

## Setup (once)

1. Install Miniconda, then from this folder:
   ```bash
   conda env create -f environment.yml
   conda activate capstone
   ```
2. On Kaggle, open the competition, accept the **Rules**, and download `train.csv`, `test.csv`, `sample_submission.csv` from the **Data** tab into `data/`. The data is not in git.
3. Run `notebooks/00_baseline.ipynb` top to bottom to check everything works.

## Rules

- **One notebook per person** (`notebooks/NN_name_topic.ipynb`). Never edit someone else's notebook.
- **Use the shared code** so scores are comparable:
  ```python
  from src.clean import load_data, make_features, make_split, save_submission
  ```
- **Always use `make_split`** (80/20, `random_state=42`, stratified) for validation AUC.
- **Branch → pull request → one teammate approves → merge.** Never push to `main` directly.
- Changing `src/clean.py` changes everyone's results. Say so in the group chat first.
- Clear big outputs before committing notebooks.

## Data facts (from train.csv)

- 699,635 training rows, 299,844 test rows, 21 features, no duplicate rows.
- 44.4% satisfied. Only missing values: 292 in `Arrival Delay in Minutes` (filled with departure delay; the two correlate at 0.84).
- 13 service ratings on 0–5, where **0 = not applicable**. Strongest single signal: `Online boarding` (correlation 0.56).
- Satisfaction by group: Business class 72.5% vs Eco 16.8%; business travel 58.4% vs personal 9.7%; loyal customers 49.5% vs disloyal 20.1%.

## Results log

Validation AUC uses `make_split`. "Would rank" compares with the final private leaderboard (3,715 valid teams; best 0.96318, median 0.96081).

| Date | Who | Model | Validation AUC | Kaggle late-submission score | Would rank | File |
|---|---|---|---|---|---|---|
| 2026-11-26 | Khine | Logistic regression, ratings as numbers | 0.92064 | | 3,669 | not submitted |
| 2026-11-26 | Khine | Logistic regression, ratings one-hot | 0.94767 | | 3,535 | `submissions/baseline_logreg_onehot.csv` |
| 2026-11-26 | Khine | Gradient boosting (HistGradientBoosting) | 0.95758 | public 0.95761 / private 0.95908 | ~2,860 / 3,715 | `submissions/baseline_gradient_boosting.csv` |
| 2026-11-27 | Khine | Gradient boosting + new features + tuning (`01_khine_features_boosting`) | 0.95745 | | 3,371 | not submitted (worse than baseline) |
| 2026-11-27 | Khine | CatBoost alone (`02_khine_catboost_lightgbm`) | 0.95755 | | 3,367 | not submitted |
| 2026-11-27 | Khine | LightGBM alone (`02_khine_catboost_lightgbm`) | 0.95819 | | 3,279 | not submitted |
| 2026-11-27 | Khine | v3 ensemble: 0.7 LightGBM + 0.2 CatBoost + 0.1 HistGB, rank average | 0.95824 | public 0.95841 / private 0.95957 | ~2,190 / 3,715 | `submissions/v3_ensemble.csv` |

Validation AUC (0.95758) matched the Kaggle public score (0.95761) within 0.00003, so our `make_split` validation is reliable. Use it to test ideas before spending Kaggle submissions. Compare the **private** score with `report/leaderboard_private_2026-11-26.csv` (that file is the private leaderboard).

v3 ensemble: validation 0.95824 vs public 0.95841 (within 0.00017). Private 0.95957 moves us from ~2,860 to ~2,190 of 3,715 (top 59%).

## Kaggle submission descriptions

Every Kaggle submission description must follow this pattern so we can match scores to models later:

`vN - model, key settings. What changed. Validation AUC X.`

Example: `v2 - Logistic regression, ratings one-hot, scaled. Validation AUC 0.94767.`

## MapReduce (Topics 4 + 9)

```bash
python src/mr_satisfaction_rate.py -r local data/train.csv
```
Satisfaction rate per Class × Type of Travel, using [sum, count] so the combiner stays correct. Output: `report/mapreduce_satisfaction_by_class_travel.txt`.

## To do

- [ ] Everyone: create a Kaggle account, accept the competition Rules, download the data into `data/`, run `00_baseline.ipynb`
- [x] Khine: submit gradient boosting baseline (private 0.95908)
- [ ] Khine: submit logistic regression baseline (`baseline_logreg_onehot.csv`)
- [ ] Khine: email professor to confirm a closed competition with late submissions is OK
- [ ] A: try new features in `src/clean.py`. Already tried in `01_khine_features_boosting` with no gain: total delay, average/min rating, count of 0 ratings. Still open: age groups, other ideas
- [x] Khine: submit v3 ensemble (private 0.95957, ~2,190 / 3,715)
- [ ] B: tune logistic regression (C value); random forest. Boosting tuning, CatBoost and LightGBM already done in notebooks 01 and 02; LightGBM is the strongest single model so far
- [ ] C: K-means on passengers (use a 20k-row sample for silhouette), describe each cluster; charts for the report
- [ ] All: report sections (see `report/REPORT_OUTLINE.md`)
