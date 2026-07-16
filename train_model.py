from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "student_burnout_cleaned_full.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "burnout_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

FEATURES = [
    "age",
    "gender",
    "academic_year",
    "study_hours_per_day",
    "exam_pressure",
    "academic_performance",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "sleep_hours",
    "physical_activity",
    "social_support",
    "screen_time",
    "internet_usage",
    "financial_stress",
    "family_expectation",
]

TARGET = "risk_level"


def stratified_training_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Keep training practical while preserving all available high-risk rows."""
    limits = {"Low": 100_000, "Medium": 40_000, "High": 20_000}
    parts = []

    for label, limit in limits.items():
        group = df[df[TARGET] == label]
        parts.append(group.sample(n=min(limit, len(group)), random_state=42))

    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Place the full cleaned burnout CSV at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = set(FEATURES + [TARGET])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Do not include burnout_score, mental_health_index, dropout_risk, or
    # risk_level_encoded. These columns directly or indirectly reveal the target.
    training_df = stratified_training_sample(df)
    X = training_df[FEATURES]
    y = training_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    categorical_features = ["gender"]
    numeric_features = [column for column in FEATURES if column not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=13,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {MODEL_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
