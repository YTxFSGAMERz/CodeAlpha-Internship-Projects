"""
Dataset loaders for the Disease Prediction project.

Three classic UCI Machine Learning Repository datasets are supported:

    heart          UCI Statlog (Heart)                  270 patients, 13 clinical features
    diabetes       Pima Indians Diabetes (NIDDK / UCI)  768 women,     8 clinical features
    breast_cancer  Wisconsin Diagnostic Breast Cancer   569 samples,  30 cell-nucleus features

Every loader returns  (X, y, info):
    X     pandas DataFrame with the raw features
    y     pandas Series, 1 = disease present, 0 = disease absent
    info  dict with a display title and the numeric / categorical column lists

The two small CSV files live in ./data. If they are missing they are downloaded once
from a public mirror and cached there, so the project also works from a fresh clone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

HEART_URL = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/heart_disease.csv"
DIABETES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

HEART_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
DIABETES_COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# In the Pima data a value of 0 is used for "not measured" in these columns
# (a living person cannot have 0 blood pressure or 0 BMI). They are treated as missing.
DIABETES_ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURE_HELP = {
    "heart": {
        "age": "age in years",
        "sex": "1 = male, 0 = female",
        "cp": "chest pain type: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic",
        "trestbps": "resting blood pressure (mm Hg)",
        "chol": "serum cholesterol (mg/dl)",
        "fbs": "fasting blood sugar > 120 mg/dl (1 = yes, 0 = no)",
        "restecg": "resting ECG: 0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy",
        "thalach": "maximum heart rate achieved",
        "exang": "exercise-induced angina (1 = yes, 0 = no)",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "slope": "slope of the peak exercise ST segment: 1 up, 2 flat, 3 down",
        "ca": "number of major vessels (0-3) coloured by fluoroscopy",
        "thal": "thalassemia: 3 normal, 6 fixed defect, 7 reversible defect",
    },
    "diabetes": {
        "Pregnancies": "number of times pregnant",
        "Glucose": "plasma glucose, 2 h oral glucose tolerance test (mg/dl)",
        "BloodPressure": "diastolic blood pressure (mm Hg)",
        "SkinThickness": "triceps skin-fold thickness (mm)",
        "Insulin": "2-hour serum insulin (mu U/ml)",
        "BMI": "body mass index (kg/m^2)",
        "DiabetesPedigreeFunction": "diabetes pedigree function (family-history score)",
        "Age": "age in years",
    },
}


def _load_heart():
    path = DATA_DIR / "heart_disease.csv"
    if not path.exists():
        raw = pd.read_csv(HEART_URL)
        raw.columns = HEART_COLUMNS + ["target"]          # the mirror uses long column names
        DATA_DIR.mkdir(exist_ok=True)
        raw.to_csv(path, index=False)
    df = pd.read_csv(path)
    y = df.pop("target").astype(int)                       # 1 = heart disease present
    info = {
        "title": "Heart disease (UCI Statlog Heart)",
        "numeric": ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"],
        "categorical": ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
    }
    return df, y, info


def _load_diabetes():
    path = DATA_DIR / "diabetes.csv"
    if not path.exists():
        raw = pd.read_csv(DIABETES_URL, header=None, names=DIABETES_COLUMNS + ["Outcome"])
        DATA_DIR.mkdir(exist_ok=True)
        raw.to_csv(path, index=False)
    df = pd.read_csv(path)
    y = df.pop("Outcome").astype(int)                      # 1 = diabetic
    info = {"title": "Diabetes (Pima Indians)", "numeric": list(df.columns), "categorical": []}
    return df, y, info


def _load_breast_cancer():
    from sklearn.datasets import load_breast_cancer       # bundled with scikit-learn
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    X.columns = [c.replace(" ", "_") for c in X.columns]
    y = (data.target == 0).astype(int)                     # sklearn: 0 = malignant -> make 1 = disease
    info = {"title": "Breast cancer (Wisconsin Diagnostic)", "numeric": list(X.columns), "categorical": []}
    return X, y, info


DATASETS = {
    "heart": _load_heart,
    "diabetes": _load_diabetes,
    "breast_cancer": _load_breast_cancer,
}


def clean_features(dataset: str, X: pd.DataFrame) -> pd.DataFrame:
    """Dataset-specific cleaning. Used at training time AND at prediction time."""
    X = X.copy()
    if dataset == "diabetes":
        for col in DIABETES_ZERO_IS_MISSING:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)
    return X


def load_dataset(name: str):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {', '.join(DATASETS)}")
    X, y, info = DATASETS[name]()
    X = clean_features(name, X)
    info["name"] = name
    return X, y.rename("disease"), info
