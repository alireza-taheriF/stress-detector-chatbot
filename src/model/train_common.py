"""Shared data, metrics, and tracking pieces for DistilBERT stress training.

The full fine-tune (`train_finetune.py`) and the LoRA run (`train_lora.py`) both
import this module so there is a single preprocessing load path and a single
metric definition.
"""

import os
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
TRAIN_DIR = "processed_train"
VAL_DIR = "processed_val"
STRESS_MODEL_DIR = "stress_model"

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
FULL_NUM_TRAIN_EPOCHS = 3
FULL_LEARNING_RATE = 2e-5

# Reported validation metrics. The full fine-tune stays the baseline;
# the LoRA comparison records these numbers instead of retraining it.
FULL_BASELINE_METRICS = {
    "accuracy": 0.8245,
    "f1": 0.8446,
    "auc": 0.9052,
}

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_lin", "k_lin", "v_lin", "out_lin"]
LORA_EPOCHS = 1
LORA_LEARNING_RATE = 2e-4
LORA_ADAPTER_DIR = "artifacts/lora-distilbert"
LORA_OUTPUT_DIR = "artifacts/lora-checkpoints"
COMPARE_PATH = "artifacts/compare_full_vs_lora.json"
MLFLOW_EXPERIMENT = "stress-detector"


def load_splits():
    """Load the tokenized train and validation splits used by the full fine-tune."""
    missing = [path for path in (TRAIN_DIR, VAL_DIR) if not os.path.isdir(path)]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing processed split(s): {joined}. "
            "Build them with the same preparation as the full fine-tune "
            "(python src/model/train_model.py) before training."
        )
    return load_from_disk(TRAIN_DIR), load_from_disk(VAL_DIR)


def compute_metrics(pred):
    """Accuracy, F1, precision, recall, and AUC on class-1 probabilities."""
    labels = np.asarray(pred.label_ids)
    logits = pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    preds = np.argmax(logits, axis=1)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
    }
    try:
        metrics["auc"] = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def count_trainable_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def directory_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def full_checkpoint_size_bytes(reference_model) -> int:
    """On-disk size of the latest full checkpoint, or the fp32 parameter bytes."""
    if os.path.isdir(STRESS_MODEL_DIR):
        checkpoints = [
            entry.path
            for entry in os.scandir(STRESS_MODEL_DIR)
            if entry.is_dir() and "checkpoint" in entry.name
        ]
        if checkpoints:
            latest = max(checkpoints, key=os.path.getctime)
            return directory_size_bytes(latest)
        size = directory_size_bytes(STRESS_MODEL_DIR)
        if size > 0:
            return size
    return sum(
        param.numel() * param.element_size() for param in reference_model.parameters()
    )


def latest_eval_metrics(log_history):
    for entry in reversed(log_history):
        if "eval_accuracy" not in entry or "eval_f1" not in entry:
            continue
        metrics = {
            "accuracy": float(entry["eval_accuracy"]),
            "f1": float(entry["eval_f1"]),
        }
        auc = entry.get("eval_auc")
        if isinstance(auc, (int, float)) and auc == auc:
            metrics["auc"] = float(auc)
        return metrics
    return {}


def log_mlflow_run(run_name, params, metrics, artifact_dir=None):
    """Log one run to the local file store at ./mlruns. No remote server."""
    # Current MLflow refuses the file store unless this opt-in is set.
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow

    tracking_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        clean_params = {
            key: value for key, value in params.items() if value is not None
        }
        mlflow.log_params(clean_params)
        clean_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and value == value
        }
        if clean_metrics:
            mlflow.log_metrics(clean_metrics)
        if artifact_dir and os.path.isdir(artifact_dir):
            mlflow.log_artifacts(artifact_dir, artifact_path="adapter")
