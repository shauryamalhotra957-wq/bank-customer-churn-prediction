import os
import glob
import json
import warnings
import subprocess
import sys

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "Exited"
np.random.seed(SEED)

def find_csvs():
    paths = []
    for pattern in ["/kaggle/input/**/*.csv", "/kaggle/working/**/*.csv", "./**/*.csv"]:
        paths.extend(glob.glob(pattern, recursive=True))
    return sorted(set(paths))

def download_fallback():
    try:
        import kagglehub
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
        import kagglehub

    csvs = []
    for dataset in ["shrutimechlearn/churn-modelling", "shantanudhakadd/bank-customer-churn-prediction"]:
        try:
            path = kagglehub.dataset_download(dataset)
            csvs.extend(glob.glob(os.path.join(path, "**/*.csv"), recursive=True))
        except Exception:
            pass
    return sorted(set(csvs))

def select_csv(csvs):
    required = {"CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"}
    ranked = []

    for file in csvs:
        try:
            sample = pd.read_csv(file, nrows=5)
            sample.columns = [str(c).strip() for c in sample.columns]
            score = len(required.intersection(set(sample.columns)))
            ranked.append((score, os.path.getsize(file), file))
        except Exception:
            continue

    if not ranked:
        raise FileNotFoundError("No readable CSV found.")

    ranked.sort(reverse=True)
    best = ranked[0][2]

    if ranked[0][0] < 8:
        raise ValueError(f"CSV found but it does not look like the expected churn dataset: {best}")

    return best

def load_data():
    csvs = find_csvs()

    if not csvs:
        csvs = download_fallback()

    selected = select_csv(csvs)
    df = pd.read_csv(selected)
    df.columns = [str(c).strip() for c in df.columns]
    return df, selected

def make_features(df):
    data = df.copy()

    required = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.drop_duplicates().reset_index(drop=True)
    data[TARGET] = data[TARGET].astype(int)

    data["BalanceSalaryRatio"] = data["Balance"] / data["EstimatedSalary"].replace(0, np.nan)
    data["BalanceSalaryRatio"] = data["BalanceSalaryRatio"].replace([np.inf, -np.inf], np.nan)

    data["AgeTenureRatio"] = data["Age"] / data["Tenure"].replace(0, np.nan)
    data["AgeTenureRatio"] = data["AgeTenureRatio"].replace([np.inf, -np.inf], np.nan)

    data["BalancePerProduct"] = data["Balance"] / data["NumOfProducts"].replace(0, np.nan)
    data["BalancePerProduct"] = data["BalancePerProduct"].replace([np.inf, -np.inf], np.nan)

    data["IsZeroBalance"] = (data["Balance"] == 0).astype(int)

    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[0, 30, 40, 50, 60, 120],
        labels=["Under_30", "30_40", "40_50", "50_60", "60_plus"],
        include_lowest=True
    )

    data["CreditScoreGroup"] = pd.cut(
        data["CreditScore"],
        bins=[0, 500, 600, 700, 800, 1000],
        labels=["Very_Low", "Low", "Medium", "High", "Very_High"],
        include_lowest=True
    )

    drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in data.columns]

    X = data.drop(columns=drop_cols + [TARGET], errors="ignore")
    y = data[TARGET].astype(int)

    return X, y, data, drop_cols

def encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def preprocessor_for(X):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

    return preprocessor, numeric, categorical

def best_threshold(y_true, probs):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_t = 0.50
    best_score = -1

    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t

def dice_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()

    d0 = 0.0 if ((2 * tn) + fp + fn) == 0 else (2 * tn) / ((2 * tn) + fp + fn)
    d1 = 0.0 if ((2 * tp) + fp + fn) == 0 else (2 * tp) / ((2 * tp) + fp + fn)

    return d0, d1, (d0 + d1) / 2

def evaluate(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    d0, d1, avg_dice = dice_from_cm(cm)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
        "average_precision": average_precision_score(y_true, probs),
        "dice_continued": d0,
        "dice_churned": d1,
        "average_dice": avg_dice,
        "confusion_matrix": cm,
        "predictions": preds
    }

def build_models(y_train):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    weight = neg / max(pos, 1)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=600, min_samples_split=4, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=700, min_samples_split=4, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=350, learning_rate=0.035, max_depth=3, subsample=0.90, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=450, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.05, random_state=SEED)
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=700,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.90,
            colsample_bytree=0.90,
            min_child_weight=2,
            reg_lambda=2.0,
            reg_alpha=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=weight,
            random_state=SEED,
            n_jobs=-1
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=24,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=2.0,
            reg_alpha=0.1,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1
        )
    except Exception:
        pass

    return models

def compare_candidate_models(models, preprocessor, X_train, y_train, X_valid, y_valid):
    """Rank model families using validation data only."""
    if not models:
        raise ValueError("At least one candidate model is required")

    rows = []
    thresholds = {}

    print(f"[2/4] Comparing {len(models)} candidate models...")
    for index, (name, model) in enumerate(models.items(), start=1):
        print(f"      {index}/{len(models)}  {name}")
        pipe = Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", clone(model))
        ])

        pipe.fit(X_train, y_train)
        valid_probs = pipe.predict_proba(X_valid)[:, 1]
        threshold = best_threshold(y_valid, valid_probs)
        metrics = evaluate(y_valid, valid_probs, threshold)
        thresholds[name] = threshold

        rows.append({
            "model": name,
            "threshold": threshold,
            "validation_accuracy": metrics["accuracy"],
            "validation_precision": metrics["precision"],
            "validation_recall": metrics["recall"],
            "validation_f1": metrics["f1"],
            "validation_roc_auc": metrics["roc_auc"],
            "validation_average_precision": metrics["average_precision"],
            "validation_dice_continued": metrics["dice_continued"],
            "validation_dice_churned": metrics["dice_churned"],
            "validation_average_dice": metrics["average_dice"]
        })

    leaderboard = pd.DataFrame(rows).sort_values(
        [
            "validation_roc_auc",
            "validation_f1",
            "validation_average_dice",
            "validation_accuracy"
        ],
        ascending=False
    ).reset_index(drop=True)
    return leaderboard, thresholds

def add_features_for_prediction(df):
    df = df.copy()

    df["BalanceSalaryRatio"] = df["Balance"] / df["EstimatedSalary"].replace(0, np.nan)
    df["BalanceSalaryRatio"] = df["BalanceSalaryRatio"].replace([np.inf, -np.inf], np.nan)

    df["AgeTenureRatio"] = df["Age"] / df["Tenure"].replace(0, np.nan)
    df["AgeTenureRatio"] = df["AgeTenureRatio"].replace([np.inf, -np.inf], np.nan)

    df["BalancePerProduct"] = df["Balance"] / df["NumOfProducts"].replace(0, np.nan)
    df["BalancePerProduct"] = df["BalancePerProduct"].replace([np.inf, -np.inf], np.nan)

    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)

    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 30, 40, 50, 60, 120], labels=["Under_30", "30_40", "40_50", "50_60", "60_plus"], include_lowest=True)
    df["CreditScoreGroup"] = pd.cut(df["CreditScore"], bins=[0, 500, 600, 700, 800, 1000], labels=["Very_Low", "Low", "Medium", "High", "Very_High"], include_lowest=True)

    return df

def train():
    print("[1/4] Discovering and validating the churn dataset...")
    df, selected_csv = load_data()
    X, y, data, drop_cols = make_features(df)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=SEED, stratify=y_train_full)

    preprocessor, numeric, categorical = preprocessor_for(X_train)
    models = build_models(y_train)
    leaderboard, thresholds = compare_candidate_models(
        models,
        preprocessor,
        X_train,
        y_train,
        X_valid,
        y_valid
    )
    best_name = leaderboard.loc[0, "model"]
    selected_threshold = thresholds[best_name]

    print(f"[3/4] Refitting the selected model: {best_name}")
    final_pipeline = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", clone(models[best_name]))
    ])

    final_pipeline.fit(X_train_full, y_train_full)

    final_probs = final_pipeline.predict_proba(X_test)[:, 1]
    final_metrics = evaluate(y_test, final_probs, selected_threshold)
    final_preds = final_metrics["predictions"]
    final_cm = final_metrics["confusion_matrix"]
    d0 = final_metrics["dice_continued"]
    d1 = final_metrics["dice_churned"]
    avg_dice = final_metrics["average_dice"]

    package = {
        "pipeline": final_pipeline,
        "threshold": float(selected_threshold),
        "model_name": best_name,
        "selection_split": "validation",
        "feature_columns": list(X.columns),
        "target_column": TARGET,
        "target_meaning": {"0": "CONTINUED_BANK", "1": "LEFT_BANK_CHURNED"},
        "dropped_columns": drop_cols,
        "numeric_features": numeric,
        "categorical_features": categorical
    }

    os.makedirs("outputs", exist_ok=True)

    joblib.dump(package, "outputs/bank_churn_model.joblib")
    leaderboard.to_csv("outputs/model_leaderboard.csv", index=False)

    test_output = X_test.copy()
    test_output["Actual_Exited"] = y_test.values
    test_output["Actual_Status"] = np.where(test_output["Actual_Exited"] == 1, "LEFT_BANK_CHURNED", "CONTINUED_BANK")
    test_output["Churn_Probability"] = final_probs
    test_output["Continue_Probability"] = 1 - final_probs
    test_output["Predicted_Exited"] = final_preds
    test_output["Predicted_Status"] = np.where(test_output["Predicted_Exited"] == 1, "LEFT_BANK_CHURNED", "CONTINUED_BANK")
    test_output.to_csv("outputs/test_predictions.csv", index=False)

    print("[4/4] Training complete. Review the decision summary below.")
    print("=" * 80)
    print("BANK CUSTOMER CHURN MODEL  |  DECISION SUMMARY")
    print("=" * 80)
    print("Dataset:", selected_csv)
    print("Best model:", best_name)
    print("Threshold:", round(selected_threshold, 3))
    print("Accuracy:", round(final_metrics["accuracy"], 4))
    print("Precision:", round(final_metrics["precision"], 4))
    print("Recall:", round(final_metrics["recall"], 4))
    print("F1:", round(final_metrics["f1"], 4))
    print("ROC-AUC:", round(final_metrics["roc_auc"], 4))
    print("Dice continued:", round(d0, 4))
    print("Dice churned:", round(d1, 4))
    print("Average dice:", round(avg_dice, 4))
    print("\nConfusion matrix:")
    print(pd.DataFrame(final_cm, index=["Actual Continued", "Actual Churned"], columns=["Predicted Continued", "Predicted Churned"]))
    print("\nClassification report:")
    print(classification_report(y_test, final_preds, target_names=["CONTINUED_BANK", "LEFT_BANK_CHURNED"], zero_division=0))
    print("\nArtifacts:")
    print("  MODEL        outputs/bank_churn_model.joblib")
    print("  LEADERBOARD  outputs/model_leaderboard.csv")
    print("  PREDICTIONS  outputs/test_predictions.csv")
    print("\nNext step: inspect false negatives before using scores in a retention workflow.")

    return package, leaderboard

def predict_customer(customer, model_path="outputs/bank_churn_model.joblib"):
    package = joblib.load(model_path)

    if isinstance(customer, dict):
        customer_df = pd.DataFrame([customer])
    else:
        customer_df = customer.copy()

    customer_df = add_features_for_prediction(customer_df)

    for col in package["feature_columns"]:
        if col not in customer_df.columns:
            customer_df[col] = np.nan

    customer_df = customer_df[package["feature_columns"]]

    probs = package["pipeline"].predict_proba(customer_df)[:, 1]
    preds = (probs >= package["threshold"]).astype(int)

    output = customer_df.copy()
    output["Churn_Probability"] = probs
    output["Continue_Probability"] = 1 - probs
    output["Predicted_Exited"] = preds
    output["Predicted_Status"] = np.where(preds == 1, "LEFT_BANK_CHURNED", "CONTINUED_BANK")

    return output

if __name__ == "__main__":
    train()

    sample = {
        "CreditScore": 650,
        "Geography": "Spain",
        "Gender": "Male",
        "Age": 42,
        "Tenure": 6,
        "Balance": 80000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 90000.0
    }

    sample_prediction = predict_customer(sample)
    print("\nSample prediction:")
    print(sample_prediction[["Churn_Probability", "Continue_Probability", "Predicted_Status"]].to_string(index=False))
