
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# 1. Chargement des données
data = pd.read_csv(
    "data/german.data",
    sep=r"\s+",
    header=None
)

columns = [
    "checking_account", "duration", "credit_history",
    "purpose", "credit_amount", "savings",
    "employment", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing",
    "existing_credits", "job", "dependents",
    "telephone", "foreign_worker", "target"
]

data.columns = columns

# 2. Séparation des variables
X = data.drop("target", axis=1)
y = (data["target"] == 1).astype(int)

categorical = X.select_dtypes(include="object").columns
numerical = X.select_dtypes(exclude="object").columns

# 3. Préparation des données
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

# 4. Création du modèle
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 5. Séparation entraînement / test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Entraînement
print("\nENTRAINEMENT DU MODELE...")
model.fit(X_train, y_train)

# 7. Prédictions sur les données de test
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

good_index = list(model.classes_).index(1)
good_probabilities = probabilities[:, good_index]

# 8. Evaluation
print("\n========== EVALUATION DU MODELE ==========")

print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
print("Precision :", round(precision_score(y_test, predictions, zero_division=0), 4))
print("Recall :", round(recall_score(y_test, predictions, zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, predictions, zero_division=0), 4))
print("ROC-AUC :", round(roc_auc_score(y_test, good_probabilities), 4))

print("\nRAPPORT DE CLASSIFICATION :")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Mauvais credit", "Bon credit"],
    zero_division=0
))

print("MATRICE DE CONFUSION :")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# 9. Sauvegarde du modèle
joblib.dump(model, "models/credit_model.pkl")
print("\nModele entraine et sauvegarde avec succes !")
