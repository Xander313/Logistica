#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para entrenar una regresión logística con 15 variables independientes
y obtener la ecuación final (modelo sigmoide) para predecir TenYearCHD.

Uso:
    python3 fit_logistic_framingham.py /home/xander/framingham.csv
"""

import sys
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# ======================================================
#          MAIN
# ======================================================
def main():
    # Verificación de argumentos
    if len(sys.argv) < 2:
        print("\n❌ ERROR: Debes indicar el archivo CSV\n")
        print("Ejemplo:")
        print("   python3 fit_logistic_framingham.py /home/xander/framingham.csv\n")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"\n❌ ERROR: No existe el archivo: {csv_path}")
        sys.exit(1)

    print("\n📥 Leyendo dataset...")
    df = pd.read_csv(csv_path)

    # Variables esperadas
    expected = [
        'male','age','education','currentSmoker','cigsPerDay','BPMeds',
        'prevalentStroke','prevalentHyp','diabetes','totChol','sysBP',
        'diaBP','BMI','heartRate','glucose','TenYearCHD'
    ]

    missing = [c for c in expected if c not in df.columns]
    if missing:
        print("\n❌ ERROR: Faltan las siguientes columnas en el dataset:")
        print(missing)
        sys.exit(1)

    # Limpieza básica
    df = df[expected].dropna().reset_index(drop=True)

    print(f"✅ Datos cargados: {len(df)} registros\n")

    # X = variables independientes, Y = etiqueta
    X = df[expected[:-1]]
    y = df['TenYearCHD'].astype(int)

    numeric_cols = [
        'male','age','currentSmoker','cigsPerDay','BPMeds','prevalentStroke',
        'prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose'
    ]
    categorical_cols = ['education']

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("logreg", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])

    # Búsqueda de hiperparámetro (regularización C)
    param_grid = {"logreg__C": [0.1, 0.5, 1, 2, 5]}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(model, param_grid, cv=cv, scoring="roc_auc", refit=True)
    grid.fit(X, y)

    best_model = grid.best_estimator_

    print("✅ Mejor parámetro C encontrado:", grid.best_params_["logreg__C"], "\n")

    # Predicciones
    y_pred = best_model.predict(X)
    y_prob = best_model.predict_proba(X)[:, 1]

    print("📊 Métricas del modelo:")
    print(classification_report(y, y_pred))
    print("AUC-ROC:", roc_auc_score(y, y_prob), "\n")

    # Extraer coeficientes
    ohe = best_model.named_steps["preprocess"].named_transformers_["cat"]
    ohe_names = list(ohe.get_feature_names_out(categorical_cols))
    feature_names = numeric_cols + ohe_names

    coef = best_model.named_steps["logreg"].coef_.ravel()
    intercept = best_model.named_steps["logreg"].intercept_[0]

    # Guardar salida
    out_txt = os.path.join(os.path.dirname(csv_path), "logistic_equation.txt")
    with open(out_txt, "w") as f:
        f.write("Ecuación logística (logit):\n")
        f.write(f"z = {intercept:.6f}")
        for name, b in zip(feature_names, coef):
            f.write(f" + ({b:.6f})*{name}")
        f.write("\n\nP(TenYearCHD=1) = 1 / (1 + exp(-z))\n")

    print("✅ Ecuación exportada en:")
    print(out_txt)
    print("\n🎯 ¡Modelo entrenado correctamente!\n")


if __name__ == "__main__":
    main()
