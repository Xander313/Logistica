from functools import lru_cache
from pathlib import Path

import pandas as pd
import numpy as np
from django.conf import settings
from django.shortcuts import render
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EXPECTED_COLUMNS = [
    "male",
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
    "TenYearCHD",
]

NUMERIC_FEATURES = [
    "male",
    "age",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]
TARGET_FEATURE = "TenYearCHD"

DEFAULT_FORM_VALUES = {
    "male": "0",
    "age": "",
    "education": "1",
    "currentSmoker": "0",
    "cigsPerDay": "",
    "BPMeds": "0",
    "prevalentStroke": "0",
    "prevalentHyp": "0",
    "diabetes": "0",
    "totChol": "",
    "sysBP": "",
    "diaBP": "",
    "BMI": "",
    "heartRate": "",
    "glucose": "",
}


def _probability_css(probability_pct):
    """Retorna la clase de Bootstrap según el rango de probabilidad."""
    if probability_pct <= 40:
        return "text-success"
    if probability_pct <= 70:
        return "text-warning"
    return "text-danger"


@lru_cache(maxsize=1)
def _get_pipeline():
    """Entrena el modelo una sola vez y lo guarda en caché."""
    csv_path = Path(settings.BASE_DIR) / "media" / "framingham.csv"
    data = pd.read_csv(csv_path, na_values=["NA", "NaN", ""])

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"El dataset no contiene las columnas esperadas: {missing_cols}"
        )

    data = data[EXPECTED_COLUMNS].dropna().reset_index(drop=True)
    X = data[EXPECTED_COLUMNS[:-1]]
    y = data[TARGET_FEATURE].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first"), ["education"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "logreg",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=5000,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    model.validation_score_ = float(model.score(X_valid, y_valid))
    return model


def _parse_payload(payload):
    """Convierte y valida los campos del formulario."""
    try:
        parsed = {
            "male": int(payload.get("male", 0)),
            "age": float(payload.get("age", 0)),
            "education": int(payload.get("education", 1)),
            "currentSmoker": int(payload.get("currentSmoker", 0)),
            "cigsPerDay": float(payload.get("cigsPerDay", 0)),
            "BPMeds": int(payload.get("BPMeds", 0)),
            "prevalentStroke": int(payload.get("prevalentStroke", 0)),
            "prevalentHyp": int(payload.get("prevalentHyp", 0)),
            "diabetes": int(payload.get("diabetes", 0)),
            "totChol": float(payload.get("totChol", 0)),
            "sysBP": float(payload.get("sysBP", 0)),
            "diaBP": float(payload.get("diaBP", 0)),
            "BMI": float(payload.get("BMI", 0)),
            "heartRate": float(payload.get("heartRate", 0)),
            "glucose": float(payload.get("glucose", 0)),
        }
    except ValueError as exc:
        raise ValueError("No se pudo convertir uno de los campos numéricos.") from exc

    if parsed["education"] not in {1, 2, 3, 4}:
        raise ValueError("El campo educación debe ser un valor entre 1 y 4.")

    return parsed


def home(request):
    context = {
        "probabilidad": None,
        "probabilidad_pct": None,
        "prediccion": None,
        "errores": None,
        "equation": None,
        "probabilidad_clase": None,
    }

    if request.method == "POST":
        try:
            features = _parse_payload(request.POST)
            pipeline = _get_pipeline()
            input_df = pd.DataFrame([features])
            probability = pipeline.predict_proba(input_df)[0, 1]
            prediction = int(probability >= 0.5)

            preprocess = pipeline.named_steps["preprocess"]
            logreg = pipeline.named_steps["logreg"]

            intercept = float(logreg.intercept_[0])
            coefficients = logreg.coef_.ravel()

            scaler = preprocess.named_transformers_["num"]
            ohe = preprocess.named_transformers_["cat"]
            cat_names = list(ohe.get_feature_names_out(["education"]))
            feature_names = NUMERIC_FEATURES + cat_names

            numeric_raw = np.array(
                [features[col] for col in NUMERIC_FEATURES], dtype=float
            )
            numeric_mean = getattr(scaler, "mean_", np.zeros_like(numeric_raw))
            numeric_scale = getattr(scaler, "scale_", np.ones_like(numeric_raw))
            numeric_scale = np.where(numeric_scale == 0, 1, numeric_scale)
            numeric_transformed = (numeric_raw - numeric_mean) / numeric_scale

            cat_transformed = ohe.transform([[features["education"]]]).toarray()[0]
            transformed_values = np.concatenate([numeric_transformed, cat_transformed])

            products = coefficients * transformed_values
            logit_value = float(intercept + products.sum())

            education_level = features["education"]
            raw_indicators = [
                int(education_level == float(name.split("_")[1]))
                for name in cat_names
            ]

            equation_terms = []
            for idx, name in enumerate(feature_names):
                term = {
                    "name": name,
                    "coefficient": float(coefficients[idx]),
                    "transformed": float(transformed_values[idx]),
                    "product": float(products[idx]),
                    "raw": None,
                    "mean": None,
                    "scale": None,
                }
                if name in NUMERIC_FEATURES:
                    num_index = NUMERIC_FEATURES.index(name)
                    term["raw"] = features[name]
                    term["mean"] = float(numeric_mean[num_index])
                    term["scale"] = float(numeric_scale[num_index])
                else:
                    cat_index = idx - len(NUMERIC_FEATURES)
                    term["raw"] = raw_indicators[cat_index]
                equation_terms.append(term)

            raw_pieces = [f"{intercept:.6f}"]
            for idx, name in enumerate(feature_names):
                if name in NUMERIC_FEATURES:
                    raw_value = features[name]
                else:
                    raw_value = raw_indicators[idx - len(NUMERIC_FEATURES)]
                raw_pieces.append(
                    f"({coefficients[idx]:.6f})*{raw_value}"
                )

            probability_pct = probability * 100
            probability_class = _probability_css(probability_pct)

            context.update(
                {
                    "probabilidad": probability,
                    "probabilidad_pct": probability_pct,
                    "prediccion": prediction,
                    "valores": {key: str(value) for key, value in features.items()},
                    "equation": {
                        "intercept": intercept,
                        "terms": equation_terms,
                        "logit": logit_value,
                        "probability": probability,
                        "raw_formula": "z = " + " + ".join(raw_pieces),
                    },
                    "probabilidad_clase": probability_class,
                }
            )
        except Exception as error:  # pylint: disable=broad-except
            context["errores"] = str(error)
            context["valores"] = DEFAULT_FORM_VALUES.copy()
            context["valores"].update(request.POST.dict())
    else:
        context["valores"] = DEFAULT_FORM_VALUES.copy()

    return render(request, "principal/index.html", context)
