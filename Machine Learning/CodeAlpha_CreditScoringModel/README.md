# Credit Scoring Model

**CodeAlpha Machine Learning Internship - Task 1**
Author: Farhan Shaikh

Predict how likely a customer is to **default** on credit, using their past financial history, and turn
that probability into a familiar 300-850 credit score.

## Dataset

[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
(UCI Machine Learning Repository, Yeh & Lien, 2009, CC BY 4.0) - a 24,000-client sample, bundled in
`data/credit_default.csv`.

| Group | Columns |
|---|---|
| Credit limit | `LIMIT_BAL` |
| Demographics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Payment history (6 months) | `PAY_1` ... `PAY_6` (-2/-1 = paid duly, 0 = revolving, 1-9 = months late) |
| Bills (6 months) | `BILL_AMT1` ... `BILL_AMT6` |
| Payments (6 months) | `PAY_AMT1` ... `PAY_AMT6` |
| Target | `default` - 1 if the client defaulted the next month (22.1 % of clients) |

## Approach

```
raw customer data -> feature engineering -> impute / scale / one-hot -> classifier -> probability of default -> credit score
```

Everything up to the classifier is one scikit-learn `Pipeline`, so cross-validation never leaks test
information and the saved model accepts raw customer data.

1. **Feature engineering** (`features.py`) - behaviour signals built from the financial history:

   | Feature | Meaning |
   |---|---|
   | `utilization_latest / _avg / _peak` | bill divided by credit limit - how much of the credit line is used |
   | `payment_ratio` | total paid divided by total billed over 6 months |
   | `max_delay`, `avg_delay`, `months_late` | how late and how often the customer pays late |
   | `recent_delay` | delays in the last 3 months only |
   | `delay_trend` | `PAY_1 - PAY_6`: is repayment behaviour getting worse? |
   | `bill_trend` | change in the bill over 6 months relative to the limit |
   | `avg_bill`, `avg_payment` | typical monthly bill and payment |

2. **Preprocessing** - median-impute + standardise numbers; mode-impute + one-hot encode categories.
3. **Models** (`train.py`) - Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, each tuned with
   5-fold stratified `GridSearchCV` on ROC-AUC. The best model is chosen on cross-validation, **never on the test set**.
4. **Decision threshold** - only ~22 % of customers default, so the usual 0.5 cut-off misses most defaulters. The
   threshold that maximises F1 for the "default" class is found on out-of-fold training predictions, then applied to
   the untouched 20 % test set.
5. **Credit score** (`scorecard.py`) - the standard "points to double the odds" scaling: 600 points = 30:1 odds of
   repaying, +40 points doubles the odds, clipped to 300-850 (illustrative constants).

## Results

Held-out test set: 4,800 customers, 1,061 of whom defaulted. Positive class = default.

| Model | CV ROC-AUC | Test ROC-AUC | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.760 | 0.756 | 0.503 | 0.563 | 0.531 | 0.29 |
| Decision Tree | 0.767 | 0.767 | 0.509 | 0.576 | 0.540 | 0.30 |
| Random Forest | 0.785 | 0.780 | 0.552 | 0.557 | 0.554 | 0.30 |
| **Gradient Boosting** (best CV) | **0.787** | 0.777 | 0.531 | 0.571 | 0.550 | 0.27 |

Best model at its tuned threshold (0.27): accuracy 0.794, precision 0.531, recall 0.571, F1 0.550, PR-AUC 0.562.
Confusion matrix: 3,203 true negatives, 536 false positives, 455 false negatives, 606 true positives.
At the default 0.5 threshold the same model catches only 37 % of defaulters (recall 0.373, precision 0.701), which is why the
threshold is tuned.

The most influential features are `PAY_1` (last month's repayment status), `max_delay`, `recent_delay`, `BILL_AMT1`,
`PAY_AMT3` and `utilization_peak` - recent late payments dominate, and several of them are the engineered features.

These numbers come from one run with `random_state=42`; they can differ slightly with other library versions. Plots are in
`outputs/`: `roc_curves.png`, `confusion_matrix.png`, `feature_importance.png`, `score_distribution.png`;
the full tables are in `outputs/model_comparison.csv` and `outputs/metrics.json`.

## Run it

```bash
pip install -r requirements.txt

python train.py            # full run (several minutes, depends on your CPU)
python train.py --quick    # smaller grids, 3-fold CV (a couple of minutes)

python predict.py --sample 8                                   # score 8 random customers
python predict.py --csv new_customers.csv                      # score a CSV with the raw columns
python predict.py --json '{"LIMIT_BAL": 50000, "AGE": 35, "PAY_1": 2, "PAY_2": 2, "BILL_AMT1": 48000, "PAY_AMT1": 0}'
```

`predict.py` prints each customer's default probability, credit score and a policy decision
(APPROVE / MANUAL REVIEW / DECLINE). Columns you leave out are imputed.

### Use your own data (for example Kaggle "Give Me Some Credit")

```bash
python train.py --source csv --data cs-training.csv --target SeriousDlqin2yrs
python train.py --source csv --data my_loans.csv --target status --positive-label bad   # text labels
```

Income, debt-ratio and late-payment features for that file are engineered automatically.

## Fairness note

Lenders in many countries may not use sex or marital status in credit decisions (for example ECOA in the US), so
`SEX` and `MARRIAGE` are **excluded by default**. `python train.py --keep-protected` includes them so you can see how little
they add - a good talking point for the video. A real credit model would also be audited for bias.

## Limitations

- One dataset from one country and period (Taiwan, 2005); a production model needs local, recent data.
- Probabilities are not recalibrated, so the score scale is illustrative, not a bureau score.
- Only people who were *given* credit appear in the data (selection bias).

## Project structure

```
data_loader.py   load the dataset (or your CSV)
features.py      feature engineering + preprocessing pipeline
scorecard.py     probability -> 300-850 score, approve / review / decline policy
train.py         tune, compare and evaluate models, save plots and the best model
predict.py       score new customers
data/            credit_default.csv
outputs/         sample results from my run
```
