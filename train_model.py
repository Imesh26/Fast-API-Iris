import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names.tolist()
    feature_names = iris.feature_names

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build & train model
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save model & metadata
    joblib.dump(pipe, "model.pkl")
    info = {
        "model_type": "LogisticRegression (with StandardScaler in Pipeline)",
        "problem_type": "classification",
        "features": [
            "sepal_length", "sepal_width", "petal_length", "petal_width"
        ],
        "feature_order_note": "Inputs must be in the order above.",
        "target_names": target_names,
        "test_accuracy": acc,
    }
    Path("model_info.json").write_text(json.dumps(info, indent=2))
    print("\nSaved model.pkl and model_info.json")


if __name__ == "__main__":
    main()
