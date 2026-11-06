"""Compare labor-contract classification before and after Weka feature selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "class"


def load_arff(path):
    records, _ = arff.loadarff(path)
    frame = pd.DataFrame(records)
    for column in frame.select_dtypes(include=["object"]).columns:
        frame[column] = frame[column].map(
            lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
        )
    return frame


def _validate_pair(original, selected):
    if TARGET not in original or TARGET not in selected:
        raise ValueError("Both datasets must contain a class target.")
    if not original.columns.is_unique or not selected.columns.is_unique:
        raise ValueError("ARFF attribute names must be unique.")
    if len(original) != len(selected) or not original[TARGET].equals(selected[TARGET]):
        raise ValueError("Original and selected rows must have matching labels in order.")
    unexpected = set(selected.columns) - set(original.columns)
    if unexpected:
        raise ValueError(f"Selected dataset has unknown attributes: {', '.join(sorted(unexpected))}")
    labels = original[TARGET].astype(str)
    if labels.isna().any() or labels.nunique() != 2:
        raise ValueError("Classification target must contain exactly two classes.")
    min_class = int(labels.value_counts().min())
    if min_class < 2:
        raise ValueError("At least two examples per class are required.")
    return labels, min(5, min_class)


def _evaluate(frame, labels, folds, seed):
    X = frame.drop(columns=TARGET)
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                     StandardScaler()), numeric)
        )
    if categorical:
        transformers.append(
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                OneHotEncoder(handle_unknown="ignore"),
            ), categorical)
        )
    pipeline = make_pipeline(
        ColumnTransformer(transformers),
        LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    classes = np.sort(labels.unique())
    probabilities_by_class = cross_val_predict(pipeline, X, labels, cv=cv, method="predict_proba")
    probabilities = probabilities_by_class[:, 1]
    predictions = classes[np.argmax(probabilities_by_class, axis=1)]
    return {
        "rows": len(frame),
        "predictors": X.shape[1],
        "roc_auc": float(roc_auc_score(labels == classes[1], probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def compare(original, selected, seed=42):
    labels, folds = _validate_pair(original, selected)
    return {
        "rows": len(original),
        "folds": folds,
        "class_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "majority_baseline_balanced_accuracy": float(
            1 / labels.nunique()
        ),
        "all_features": _evaluate(original, labels, folds, seed),
        "weka_selected_features": _evaluate(selected, labels, folds, seed),
        "selected_predictors": selected.columns.drop(TARGET).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("labor_data_before_feature_selection.arff"))
    parser.add_argument("--selected", type=Path, default=Path("labor_data_after_feature_selection.arff"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = compare(load_arff(args.original), load_arff(args.selected), args.seed)
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
