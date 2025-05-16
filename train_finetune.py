import torch
import os
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 1. Load data
train_ds = load_from_disk("processed_train")
val_ds = load_from_disk("processed_val")

# 2. Model setting for M2
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)

# 3. Training parameters
training_args = TrainingArguments(
    output_dir="stress_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    # Rename parameters to the new version
    eval_strategy="epoch",          # جایگزین evaluation_strategy → eval_strategy
    save_strategy="epoch",         
    eval_steps=1000,               
    save_steps=1000,
    save_total_limit=3,
    logging_dir="logs",
    logging_steps=10,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    # Using mixed precision suitable for M2
    bf16=torch.backends.mps.is_available()  # Activation for Apple Silicon
)

# 4. Metric function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='binary'),
        "precision": precision_score(labels, preds, average='binary'),
        "recall": recall_score(labels, preds, average='binary')
    }

# 5. Training implementation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# 6. Continued from Checkpoint
checkpoints = [f.path for f in os.scandir("stress_model") if f.is_dir() and "checkpoint" in f.name]
last_checkpoint = max(checkpoints, key=os.path.getctime) if checkpoints else None

try:
    trainer.train(resume_from_checkpoint=last_checkpoint)
except KeyboardInterrupt:
    print("\n🛑 Training stopped! Save the last state...")
    trainer.save_model("stress_model/interrupted")
    print("✅ Model saved in 'stress_model/interrupted'!")