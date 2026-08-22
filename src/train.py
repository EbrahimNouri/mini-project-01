from sklearn.neighbors import KNeighborsClassifier
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


