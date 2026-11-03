### Heart disease (UCI Statlog Heart)

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.912 | 0.870 | 0.815 | 0.917 | 0.863 | 0.911 |
| SVM | 0.913 | 0.852 | 0.808 | 0.875 | 0.840 | 0.907 |
| Random Forest | 0.913 | 0.833 | 0.778 | 0.875 | 0.824 | 0.893 |
| XGBoost | 0.899 | 0.833 | 0.800 | 0.833 | 0.816 | 0.890 |

Best model (by cross-validation): **SVM**

### Diabetes (Pima Indians)

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.845 | 0.708 | 0.571 | 0.667 | 0.615 | 0.810 |
| SVM | 0.849 | 0.708 | 0.600 | 0.500 | 0.545 | 0.810 |
| Random Forest | 0.837 | 0.740 | 0.600 | 0.778 | 0.677 | 0.811 |
| XGBoost | 0.840 | 0.773 | 0.642 | 0.796 | 0.711 | 0.818 |

Best model (by cross-validation): **SVM**

### Breast cancer (Wisconsin Diagnostic)

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.995 | 0.991 | 1.000 | 0.976 | 0.988 | 0.998 |
| SVM | 0.995 | 0.974 | 1.000 | 0.929 | 0.963 | 0.996 |
| Random Forest | 0.990 | 0.974 | 1.000 | 0.929 | 0.963 | 0.998 |
| XGBoost | 0.995 | 0.974 | 1.000 | 0.929 | 0.963 | 0.989 |

Best model (by cross-validation): **SVM**
