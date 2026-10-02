import os
import pickle

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def execute_pipeline():
    print("[MLOps Pipeline] Starting pipeline execution...")

    # Gather data payloads
    raw_data = load_iris(as_frame=True)
    df = raw_data.frame

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    classifier.fit(X_train, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)

    with open("models/iris_model.pkl", "wb") as f:
        pickle.dump(classifier, f)

    score = classifier.score(X_test, y_test)

    print(
        f"[MLOps Pipeline] Target model successfully cached. "
        f"Score: {score:.4f}"
    )


if __name__ == "__main__":
    execute_pipeline()