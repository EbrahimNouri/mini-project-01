import os

import joblib
import numpy as np
from pygame.examples.video import x_
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name: str):
    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(cm)
    print()
    print(f"True Negatives  (correctly predicted legit): {tn}")
    print(f"False Positives (predicted fraud, actually legit): {fp}")
    print(f"False Negatives (predicted legit, actually fraud): {fn}  ← DANGEROUS!")
    print(f"True Positives  (correctly predicted fraud): {tp}")
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def cross_validate_model(model, X, y, model_name: str, folds: int = 5, random_state: int = 42):
    print("\n" + "-" * 60)
    print(f"CROSS-VALIDATION: {model_name}")
    print("-" * 60)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    precision_scores = cross_val_score(model, X, y, cv=skf, scoring="precision")
    recall_scores = cross_val_score(model, X, y, cv=skf, scoring="recall")
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring="f1")

    for fold in range(folds):
        print(
            f"  Fold {fold + 1}: P={precision_scores[fold]:.4f}  "
            f"R={recall_scores[fold]:.4f}  F1={f1_scores[fold]:.4f}"
        )
    mean_p = precision_scores.mean()
    mean_r = recall_scores.mean()
    mean_f1 = f1_scores.mean()
    print(f"\n  Mean Precision: {mean_p:.4f} (±{precision_scores.std():.4f})")
    print(f"  Mean Recall   : {mean_r:.4f} (±{recall_scores.std():.4f})")
    print(f"  Mean F1       : {mean_f1:.4f} (±{f1_scores.std():.4f})")

    return {
        "model_name": model_name,
        "mean_precision": mean_p,
        "mean_recall": mean_r,
        "mean_f1": mean_f1,
    }


def experiment_scaling_effect(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: EFFECT OF FEATURE SCALING ON KNN")
    print("=" * 60)

    results = {}

    knn_no_scale = KNeighborsClassifier(n_neighbors=5)
    knn_no_scale.fit(X_train, y_train)
    y_pred_no_scale = knn_no_scale.predict(X_test)

    results["KNN_Without_Scaling"] = {
        "precision": precision_score(y_test, y_pred_no_scale),
        "recall": recall_score(y_test, y_pred_no_scale),
        "f1": f1_score(y_test, y_pred_no_scale),
    }

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = knn_scaled.predict(X_test_scaled)

    results["KNN_With_Scaling"] = {
        "precision": precision_score(y_test, y_pred_scaled),
        "recall": recall_score(y_test, y_pred_scaled),
        "f1": f1_score(y_test, y_pred_scaled),
    }

    # Print comparison table
    print("\n| Model                  | Scaling     | Precision | Recall  | F1     |")
    print("|------------------------|-------------|-----------|---------|--------|")
    for name, m in results.items():
        scaling = "Yes" if "With" in name else "No"
        print(
            f"| KNN                    | {scaling:>11} | {m['precision']:.4f}   | {m['recall']:.4f} | {m['f1']:.4f} |"
        )

    print("\nExplanation:")
    print("  - KNN is a distance-based algorithm. Features with larger numeric")
    print("    ranges dominate the distance calculation if not scaled.")
    print("  - StandardScaler normalizes all features to mean=0, std=1,")
    print("    giving each feature equal influence on the distance.")
    print("  - Decision Trees are NOT affected by scaling because they make")
    print("    axis-based splits on individual features.")

    return results


def experiment_hyperparameter(X_train, X_test, y_train, y_test, random_state: int = 42):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: HYPERPARAMETER ANALYSIS (Decision Tree max_depth)")
    print("=" * 60)

    depths = [2, 5, 10, None]
    results = []

    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
        dt.fit(X_train, y_train)
        y_train_pred = dt.predict(X_train)
        y_test_pred = dt.predict(X_test)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        results.append({
            "max_depth": str(depth),
            "train_f1": train_f1,
            "test_f1": test_f1,
            "precision": test_precision,
            "recall": test_recall,
        })
    print("\n| max_depth | Train F1 | Test F1 | Precision | Recall  |")
    print("|-----------|----------|---------|-----------|---------|")
    for r in results:
        print(
            f"| {r['max_depth']:>9} | {r['train_f1']:.4f}  | {r['test_f1']:.4f} | {r['precision']:.4f}   | {r['recall']:.4f} |"
        )

    print("\nAnalysis:")
    print("  - When max_depth=None, the tree perfectly memorizes training data")
    print("    (Train F1 ≈ 1.0) but performs poorly on test data → OVERFITTING.")
    print("  - Small max_depth (e.g., 2) underfits — too simple to capture patterns.")
    print("  - The best depth provides a good balance between train/test performance.")

    return results


def experiment_threshold(model, X_test, y_test):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: CLASSIFICATION THRESHOLD ANALYSIS")
    print("=" * 60)
    y_proba = model.predoba(X_test)[:, 1]
    thresholds = [0.3, 0.5, 0.7]
    results = []

    for threshold in thresholds:
        y_pred_custom = (y_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred_custom)
        recall = recall_score(y_test, y_pred_custom)
        f1 = f1_score(y_test, y_pred_custom)
        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    print("\n| Threshold | Precision | Recall  | F1     |")
    print("|-----------|-----------|---------|--------|")
    for r in results:
        print(
            f"| {r['threshold']:>9} | {r['precision']:.4f}   | {r['recall']:.4f} | {r['f1']:.4f} |"
        )

    print("\nExplanation:")
    print("  - Lowering the threshold (e.g., 0.3) → more transactions flagged as fraud")
    print("      → Recall increases (catch more fraud) but Precision decreases (more false alarms)")
    print("  - Raising the threshold (e.g., 0.7) → fewer transactions flagged as fraud")
    print("      → Precision increases (fewer false alarms) but Recall decreases (miss some fraud)")
    print("  - For fraud detection, a lower threshold is often preferred because")
    print("    the cost of missing fraud (FN) is higher than the cost of a false alarm (FP).")

    return results


def run_training_pipeline(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    X_full_scaled=None,
    y_full=None,
    random_state: int = 42
):
    print("\n" + "=" * 60)
    print("Run Training Pipeline")
    print("=" * 60)
    if X_full_scaled is None:
        X_full_scaled = np.vstack([X_train_scaled, X_test_scaled])
    if y_full is None:
        y_full = np.concatenate([y_train, y_test])

    results = []

    lr_model = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_result = train_and_evaluate(
        lr_model, X_train_scaled, X_test_scaled, y_train, y_test,
        "Logistic Regression"
    )
    results.append(lr_result)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_result = train_and_evaluate(
        knn_model, X_train_scaled, X_test_scaled, y_train, y_test,
        "K-Nearest Neighbors (K=5)"
    )
    results.append(knn_result)

    dt_model = DecisionTreeClassifier(max_depth=10, random_state=random_state)
    dt_result = train_and_evaluate(
        dt_model, X_train_scaled, X_test_scaled, y_train, y_test,
        "Decision Tree (max_depth=10)"
    )
    results.append(dt_result)

    cv_results = []
    cv_lr = cross_validate_model(
        LogisticRegression(max_iter=1000, random_state=random_state),
        X_full_scaled, y_full, "Logistic Regression"
    )
    cv_results.append(cv_lr)

    cv_knn = cross_validate_model(
        KNeighborsClassifier(n_neighbors=5),
        X_full_scaled, y_full, "K-Nearest Neighbors (K=5)"
    )
    cv_results.append(cv_knn)

    cv_dt = cross_validate_model(
        DecisionTreeClassifier(max_depth=10, random_state=random_state),
        X_full_scaled, y_full, "Decision Tree (max_depth=10)"
    )
    cv_results.append(cv_dt)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print("\n| Model                  | Mean Precision | Mean Recall | Mean F1  |")
    print("|------------------------|----------------|-------------|----------|")
    for cv in cv_results:
        print(
            f"| {cv['model_name']:<22} | {cv['mean_precision']:.4f}         | {cv['mean_recall']:.4f}      | {cv['mean_f1']:.4f}  |"
        )

    scaling_results = experiment_scaling_effect(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    hyperparam_results = experiment_hyperparameter(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    threshold_results = experiment_threshold(lr_model, X_test_scaled, y_test)

    print("\n" + "=" * 60)
    print("SAVING FINAL MODEL")
    print("=" * 60)

    lr_model.fit(X_train_scaled, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(lr_model, "models/model.pkl")
    print("Final model (Logistic Regression) saved to models/model.pkl")

    # Verify all three required artifacts exist
    required_artifacts = [
        "models/model.pkl",
        "models/encoder.pkl",
        "models/scaler.pkl",
    ]
    print("\nVerifying saved artifacts:")
    for artifact in required_artifacts:
        exists = os.path.exists(artifact)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {artifact}")

    if not all(os.path.exists(a) for a in required_artifacts):
        print("\nWARNING: Some artifacts are missing! Run data_prep.py first.")
    else:
        print("\nAll artifacts saved successfully in models/")

    return results, cv_results, scaling_results, hyperparam_results, threshold_results


# ---------------------------------------------------------------------------
# Allow running this module directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_prep import run_data_pipeline

    X_train_scaled, X_test_scaled, y_train, y_test = run_data_pipeline("../data/creditcard.csv")
    run_training_pipeline(X_train_scaled, X_test_scaled, y_train, y_test)
