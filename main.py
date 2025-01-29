import pickle
import json
import hashlib
import time
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
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
data.reset_index(inplace=True)  # Ensure index is usable
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

def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Use seaborn to create a heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", linewidths=0.5, linecolor="black", cbar=True)

    # Labels and title
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name} - Confusion Matrix")

    # Save the plot
    plt.savefig(f"Results/{model_name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_results(model_name, X_test, y_test, predictions, is_meta):
    plt.ioff()  # Disable interactive mode for speed

    feature_x, feature_y = "Tot Fwd Pkts", "Tot Bwd Pkts"
    use_log_scale=True

    # Ensure required columns exist in the dataset
    required_columns = [feature_x, feature_y]
    if not all(col in data.columns for col in required_columns):
        print(f"Missing required columns! Found: {data.columns.tolist()}")
        return

    # Extract selected features from the dataset
    plot_data = data.loc[:, required_columns]

    # Get packet indices
    packet_indices = y_test.index if is_meta else X_test.index
    merged_data = plot_data.loc[packet_indices]

    # Identify correct and incorrect classifications
    correct_mask = (predictions == y_test)
    incorrect_mask = ~correct_mask

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot with log scaling (if enabled)
    ax.scatter(merged_data.loc[correct_mask, feature_x], 
               merged_data.loc[correct_mask, feature_y], 
               color='blue', s=3, alpha=0.5, label="Correctly Classified")

    ax.scatter(merged_data.loc[incorrect_mask, feature_x], 
               merged_data.loc[incorrect_mask, feature_y], 
               color='red', s=3, alpha=0.5, label="Incorrectly Classified")

    # Apply log scale to axes to handle outliers
    if use_log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Labels and title
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f"{model_name} - Packet Classification ({feature_x} vs. {feature_y})")

    # Add legend
    ax.legend()

    # Hide unnecessary spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save and close plot to avoid memory issues
    plt.savefig(f"Results/{model_name}_classification.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

def evaluate_model(model, X_train, X_test, y_test, model_name, is_meta=False):
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
        "Confusion Matrix": cm.tolist(),
    }
    sub_model_metrics[model_name] = metrics

    verbose_log(f"\n{model_name} Metrics:")
    for key, value in metrics.items():
        verbose_log(f"{key}: {value}")

    # Calculate TPR and TNR
    rates = calculate_rates(cm)
    verbose_log(f"{model_name} True Positive Rate (TPR): {rates['TPR']:.4f}")
    verbose_log(f"{model_name} False Positive Rate (FPR): {rates['FPR']:.4f}")

    # Generate and save confusion matrix heatmap
    plot_confusion_matrix(cm, model_name)
    verbose_log(f"Confusion matrix saved for {model_name}.")

    # Generate spatial/temporal classification graph
    plot_results(model_name, X_test, y_test, preds, is_meta)
    verbose_log(f"Graph saved for {model_name}.")

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

evaluate_model(meta_model, meta_X_train, meta_X_test, meta_y_test, "Meta_Model", is_meta=True)