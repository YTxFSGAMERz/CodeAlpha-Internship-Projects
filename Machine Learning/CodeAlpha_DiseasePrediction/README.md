# Disease Prediction from Medical Data

**CodeAlpha Machine Learning Internship - Task 4**
Author: Farhan Shaikh

Predict the possibility of a disease from structured patient data with four classification algorithms -
**Logistic Regression, SVM, Random Forest and XGBoost** - on three UCI Machine Learning Repository datasets.

> Educational project. It is **not** a medical device and gives **no medical advice**.

## Datasets

| Dataset | Source | Patients | Features | Disease prevalence |
|---|---|---|---|---|
| `heart` - heart disease | UCI *Statlog (Heart)* | 270 | 13 (age, sex, chest-pain type, blood pressure, cholesterol, ECG, max heart rate, ST depression, ...) | 44.4 % |
| `diabetes` - diabetes | Pima Indians Diabetes (NIDDK / UCI) | 768 | 8 (glucose, BMI, blood pressure, insulin, age, pregnancies, ...) | 34.9 % |
| `breast_cancer` - breast cancer | Wisconsin Diagnostic Breast Cancer (UCI, bundled with scikit-learn) | 569 | 30 cell-nucleus measurements | 37.3 % malignant |

The heart and diabetes CSV files are bundled in `data/` (they are re-downloaded automatically if missing).
In every dataset the target is `1 = disease present`.

## Approach

1. **Split** - 80 % train / 20 % test, *stratified* so both parts keep the disease ratio.
2. **Preprocessing inside a `Pipeline`** - median-impute + standardise numeric columns; mode-impute + one-hot encode
   categorical columns (chest-pain type, thal, ...). Because it is part of the pipeline, test data never influences
   the imputation or scaling.
3. **Data cleaning** - in the diabetes data a value of 0 for glucose, blood pressure, skin thickness, insulin or BMI means
   "not measured", so those zeros are treated as missing values and imputed.
4. **Hyper-parameter tuning** - 5-fold stratified `GridSearchCV` (scored on ROC-AUC) for every model. Class imbalance is
   handled with class weights (`scale_pos_weight` for XGBoost).
5. **Model choice** - the best model is picked on **cross-validation**, never on the test set.
6. **Evaluation** - accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC, confusion matrix, ROC curves and
   permutation feature importance.

Why more than accuracy? In medicine a **false negative** (a sick patient labelled healthy) is usually worse than a false
positive, so **recall / sensitivity** matters most, and ROC-AUC shows the trade-off across all thresholds.

## Results

Test sets are small (54, 154 and 114 patients), so a single split is noisy - one patient moves the heart-disease
numbers by about two percentage points. The cross-validated ROC-AUC (on the training data) is the steadier number.
Full tables: `outputs/results_summary.md`, `outputs/<dataset>/model_comparison.csv`.

### Heart disease

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.912 | 0.870 | 0.815 | 0.917 | 0.863 | 0.911 |
| **SVM** (best CV) | 0.913 | 0.852 | 0.808 | 0.875 | 0.840 | 0.907 |
| Random Forest | 0.913 | 0.815 | 0.769 | 0.833 | 0.800 | 0.886 |
| XGBoost | 0.901 | 0.833 | 0.800 | 0.833 | 0.816 | 0.887 |

Most important features (best model): `ca`, `thal`, `cp`, `oldpeak`, `thalach`.

### Diabetes

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.845 | 0.708 | 0.571 | 0.667 | 0.615 | 0.810 |
| **SVM** (best CV) | 0.849 | 0.708 | 0.600 | 0.500 | 0.545 | 0.810 |
| Random Forest | 0.842 | 0.747 | 0.615 | 0.741 | 0.672 | 0.821 |
| XGBoost | 0.841 | 0.760 | 0.631 | 0.759 | 0.689 | 0.818 |

Most important features: `Glucose`, `BMI`, `Pregnancies`, `DiabetesPedigreeFunction`, `Age`.
The four models are nearly tied on ROC-AUC, but the SVM that wins on cross-validation finds only half of the diabetic
patients (recall 0.50) while Random Forest and XGBoost find about three quarters. For screening, run
`python train.py --dataset diabetes --scoring recall` to tune and select for sensitivity instead.

### Breast cancer

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.995 | 0.974 | 0.976 | 0.952 | 0.964 | 0.995 |
| **SVM** (best CV) | 0.996 | 0.983 | 1.000 | 0.952 | 0.976 | 0.997 |
| Random Forest | 0.989 | 0.974 | 1.000 | 0.929 | 0.963 | 0.996 |
| XGBoost | 0.995 | 0.974 | 1.000 | 0.929 | 0.963 | 0.991 |

The best model has no false alarms and misses 2 of 42 malignant cases in the test set. Most important features:
`worst_texture`, `worst_concavity`, `mean_concave_points`, `mean_concavity`, `worst_smoothness`.

Numbers come from one run with `random_state=42` and may differ slightly with other library versions.
Plots for each dataset (`roc_curves.png`, `confusion_matrix.png`, `feature_importance.png`) are in `outputs/<dataset>/`.

## Run it

```bash
pip install -r requirements.txt

python train.py --dataset all                 # heart + diabetes + breast cancer (several minutes)
python train.py --dataset heart               # one dataset
python train.py --dataset diabetes --scoring recall
python train.py --dataset heart --quick       # smaller grids, finishes in seconds

python predict.py --dataset heart --describe  # what each input means
python predict.py --dataset heart --example   # score a real patient from the dataset
python predict.py --dataset heart --json '{"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0, "restecg": 2, "thalach": 120, "exang": 1, "oldpeak": 2.0, "slope": 2, "ca": 2, "thal": 7}'
```

Training writes `outputs/<dataset>/` (tables, plots, metrics) and `models/<dataset>_model.joblib`.

## Project structure

```
data_loader.py   loads the three datasets and does dataset-specific cleaning
train.py         tunes four models per dataset, evaluates them, saves plots and the best model
predict.py       risk prediction for one patient
data/            heart_disease.csv, diabetes.csv
outputs/         sample results from my run
```

## Limitations

- Small datasets: results vary between splits; cross-validation is more trustworthy than one test set.
- The datasets are old and from specific populations; the models are not validated for real patients.
- The default decision threshold is 0.5; a clinical screening tool would choose it from the cost of missed cases.
