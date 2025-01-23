import pickle
import json
import hashlib
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Variables
general_features = ['Flow Duration', 'Protocol_6', 'Protocol_17', 'Pkt Len Min', 'Pkt Len Max']
statistical_features = ['Tot Fwd Pkts', 'TotLen Fwd Pkts', 'Flow IAT Mean', 'Fwd IAT Std', 'Bwd Pkt Len Mean']
behavioral_features = ['SYN Flag Cnt', 'ACK Flag Cnt', 'Fwd Pkts/s', 'Bwd Pkts/s', 'RST Flag Cnt', 'Active Mean', 'Idle Mean', 'Down/Up Ratio']

data = pd.read_csv("processed_dataset.csv")
csv_name = "processed_dataset.csv"

X_general = data[general_features]
X_statistical = data[statistical_features]
X_behavioral = data[behavioral_features]
y = data['Label']

splits = {
    "General": train_test_split(X_general, y, test_size=0.3, stratify=y, random_state=42),
    "Statistical": train_test_split(X_statistical, y, test_size=0.3, stratify=y, random_state=42),
    "Behavioral": train_test_split(X_behavioral, y, test_size=0.3, stratify=y, random_state=42)
}

model_configs = {
    "RandomForest_General": {
        "model": RandomForestClassifier(random_state=42),
        "param_grid": {"n_estimators": [50, 100], "max_depth": [10, 20, None], "min_samples_split": [5, 2]},
        "data_split": splits["General"]
    },
    "MLPClassifier_Statistical": {
        "model": MLPClassifier(max_iter=500, random_state=42),
        "param_grid": {"hidden_layer_sizes": [(50,), (100,)], "activation": ["relu", "tanh"], "solver": ["adam", "sgd"], "alpha": [0.0001, 0.001]},
        "data_split": splits["Statistical"]
    },
    "GradientBoosting_Behavioral": {
        "model": GradientBoostingClassifier(random_state=42),
        "param_grid": {"n_estimators": [100, 50], "learning_rate": [0.1, 0.01], "max_depth": [5, 3]},
        "data_split": splits["Behavioral"]
    }
}

cache_dir = Path("Cached_Models")
cache_dir.mkdir(exist_ok=True)
cache_metadata_path = cache_dir / "cache_metadata.json"

if cache_metadata_path.exists():
    with cache_metadata_path.open("r") as f:
        cache_metadata = json.load(f)
else:
    cache_metadata = {}

sub_model_predictions_train = {}
sub_model_predictions_test = {}
sub_model_metrics = {}

use_grid_search = False
display_data = False

# Methods
def verbose_log(message, level="INFO"):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}][{level}] {message}")

def save_cache_metadata():
    with cache_metadata_path.open("w") as f:
        json.dump(cache_metadata, f, indent=4)

def normalize_parameters(params):
    return {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}

def get_param_hash(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()

def calculate_rates(confusion_matrix):
    TN, FP = confusion_matrix[0]
    FN, TP = confusion_matrix[1]

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    return {"TPR": TPR, "FPR": FPR}

def visualize_model_classification(model, X, y, feature_names=None, use_pca=True, title="Model Classification"):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    if use_pca and X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        x1, x2 = X_reduced[:, 0], X_reduced[:, 1]
        xlabel, ylabel = "PCA Component 1", "PCA Component 2"
    elif X.shape[1] > 1:
        x1, x2 = X.iloc[:, 0], X.iloc[:, 1]
        xlabel, ylabel = feature_names[0] if feature_names else "Feature 1", feature_names[1] if feature_names else "Feature 2"
    else:
        raise ValueError("X must have at least 2 features or enable PCA for dimensionality reduction.")

    y_pred = model.predict(X)

    plt.figure(figsize=(10, 7))
    plt.scatter(x1, x2, c=y_pred, cmap="coolwarm", alpha=0.6, edgecolor="k", label="Predicted")
    misclassified = y != y_pred
    plt.scatter(x1[misclassified], x2[misclassified], facecolors="none", edgecolors="red", label="Misclassified")

    for i, (x, y_point) in enumerate(zip(x1[misclassified], x2[misclassified])):
        plt.text(x, y_point, str(y.iloc[i]), color="red", fontsize=8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)  # Briefly pause to display the plot without blocking

def evaluate_model(model, X_train, X_test, y_test, model_name):
    verbose_log(f"Evaluating model for {model_name}...")

    if hasattr(model, "predict_proba"):
        verbose_log(f"{model_name}: Generating probabilities using `predict_proba`...")
        train_probs = model.predict_proba(X_train)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
    else:
        verbose_log(f"{model_name}: Generating predictions using `predict`...")
        train_probs = model.predict(X_train)
        test_probs = model.predict(X_test)

    verbose_log(f"{model_name}: Storing predictions...")
    sub_model_predictions_train[model_name] = train_probs
    sub_model_predictions_test[model_name] = test_probs

    verbose_log(f"{model_name}: Calculating evaluation metrics...")
    preds = (test_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, preds)
    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, pos_label=1),
        "Recall": recall_score(y_test, preds, pos_label=1),
        "F1 Score": f1_score(y_test, preds, pos_label=1),
        "Confusion Matrix": cm.tolist()
    }
    sub_model_metrics[model_name] = metrics

    verbose_log(f"\n{model_name} Metrics:")
    for key, value in metrics.items():
        verbose_log(f"{key}: {value}")

    # Calculate TPR and TNR
    rates = calculate_rates(cm)
    verbose_log(f"{model_name} True Positive Rate (TPR): {rates['TPR']:.4f}")
    verbose_log(f"{model_name} False Positive Rate (FPR): {rates['FPR']:.4f}")

def train_and_get_predictions(model_name, model, param_grid, X_train, y_train, X_test, y_test):
    param_hash = get_param_hash(param_grid if use_grid_search else normalize_parameters({k: v[0] for k, v in param_grid.items()}))
    cache_path = cache_dir / f"{model_name}_{param_hash}.pkl"

    verbose_log(f"Checking cache for {model_name} with hash {param_hash}...")

    if model_name in cache_metadata and param_hash in cache_metadata[model_name]:
        cached_data = cache_metadata[model_name][param_hash]
        if cached_data["dataset"] == csv_name:
            verbose_log(f"Attempting to load cached model for {model_name}...")
            try:
                with cache_path.open("rb") as f:
                    cached_model = pickle.load(f)
                verbose_log(f"Cached model for {model_name} loaded successfully. Evaluating...")
                evaluate_model(cached_model, X_train, X_test, y_test, model_name)
                if display_data: 
                    visualize_model_classification(model=cached_model, X=X_test, y=y_test, feature_names=X_train.columns.tolist(), use_pca=True, title=f"{model_name} Classification")
                return
            except FileNotFoundError:
                verbose_log(f"Cache file for {model_name} not found. Retraining...", level="WARNING")

    if use_grid_search:
        verbose_log(f"Starting Grid Search for {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = normalize_parameters(grid_search.best_params_)
        verbose_log(f"Grid Search complete for {model_name}. Best parameters: {best_params}")
    else:
        verbose_log(f"Training {model_name} with predefined parameters...")
        best_model = model.set_params(**{k: v[0] for k, v in param_grid.items()})
        best_model.fit(X_train, y_train)
        best_params = normalize_parameters({k: v[0] for k, v in param_grid.items()})

    verbose_log(f"Caching model for {model_name} with parameters {best_params}...")
    with cache_path.open("wb") as f:
        pickle.dump(best_model, f)
    cache_metadata.setdefault(model_name, {})[param_hash] = {"dataset": csv_name, "parameters": best_params}
    save_cache_metadata()

    verbose_log(f"Evaluating newly trained model for {model_name}...")
    evaluate_model(best_model, X_train, X_test, y_test, model_name)

    if display_data: 
        visualize_model_classification(model=cached_model, X=X_test, y=y_test, feature_names=X_train.columns.tolist(), use_pca=True, title=f"{model_name} Classification")

# Main Loop
for model_name, config in model_configs.items():
    X_train, X_test, y_train, y_test = config["data_split"]
    train_and_get_predictions(model_name, config["model"], config["param_grid"], X_train, y_train, X_test, y_test)

meta_X_train = np.column_stack(list(sub_model_predictions_train.values()))
meta_X_test = np.column_stack(list(sub_model_predictions_test.values()))
meta_y_train = splits["General"][2]
meta_y_test = splits["General"][3]

meta_model = LogisticRegression(random_state=42)
meta_model.fit(meta_X_train, meta_y_train)

final_test_probs = meta_model.predict_proba(meta_X_test)[:, 1]
final_predictions = (final_test_probs >= 0.5).astype(int)

accuracy = accuracy_score(meta_y_test, final_predictions)
precision = precision_score(meta_y_test, final_predictions, pos_label=1)
recall = recall_score(meta_y_test, final_predictions, pos_label=1)
f1 = f1_score(meta_y_test, final_predictions, pos_label=1)
conf_matrix = confusion_matrix(meta_y_test, final_predictions)

final_metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1, "Confusion Matrix": conf_matrix.tolist()}
verbose_log("Final Metrics for Hierarchical Model with Learning-Based Weights:")
for key, value in final_metrics.items():
    verbose_log(f"{key}: {value}", level="RESULT")

# Calculate TPR and TNR
meta_rates = calculate_rates(conf_matrix)
verbose_log(f"{model_name} True Positive Rate (TPR): {meta_rates['TPR']:.4f}")
verbose_log(f"{model_name} False Positive Rate (FPR): {meta_rates['FPR']:.4f}")

if display_data:
    # Visualize meta-model predictions
    visualize_model_classification(model=meta_model, X=meta_X_test, y=meta_y_test, feature_names=["Sub-Model 1", "Sub-Model 2", "Sub-Model 3"], use_pca=True, title="Meta-Model Classification")

    plt.ioff()  # Disable interactive mode
    plt.show()  # Show all plots