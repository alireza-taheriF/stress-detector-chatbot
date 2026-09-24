import os

import torch
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from train_common import (
    EVAL_BATCH_SIZE,
    FULL_LEARNING_RATE,
    FULL_NUM_TRAIN_EPOCHS,
    MODEL_NAME,
    NUM_LABELS,
    STRESS_MODEL_DIR,
    TRAIN_BATCH_SIZE,
    compute_metrics,
    count_trainable_parameters,
    latest_eval_metrics,
    load_splits,
    log_mlflow_run,
)


def main():
    # 1. Load data (same processed splits the LoRA run imports)
    train_ds, val_ds = load_splits()

    # 2. Model setting for M2
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    ).to(device)

    # 3. Training parameters
    training_args = TrainingArguments(
        output_dir=STRESS_MODEL_DIR,
        num_train_epochs=FULL_NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        # Rename parameters to the new version
        eval_strategy="epoch",  # Replace evaluation_strategy → eval_strategy
        save_strategy="epoch",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=3,
        logging_dir="logs",
        logging_steps=10,
        learning_rate=FULL_LEARNING_RATE,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        # Using mixed precision suitable for M2
        bf16=torch.backends.mps.is_available(),  # Activation for Apple Silicon
    )

    # 5. Training implementation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # 6. Continued from Checkpoint
    checkpoints = []
    if os.path.isdir(STRESS_MODEL_DIR):
        checkpoints = [
            entry.path
            for entry in os.scandir(STRESS_MODEL_DIR)
            if entry.is_dir() and "checkpoint" in entry.name
        ]
    last_checkpoint = max(checkpoints, key=os.path.getctime) if checkpoints else None

    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except KeyboardInterrupt:
        print("\n🛑 Training stopped! Save the last state...")
        trainer.save_model("stress_model/interrupted")
        print("✅ Model saved in 'stress_model/interrupted'!")
        return

    metrics = latest_eval_metrics(trainer.state.log_history)
    log_mlflow_run(
        run_name="full-finetune",
        params={
            "model_name": MODEL_NAME,
            "num_labels": NUM_LABELS,
            "method": "full_finetune",
            "num_train_epochs": FULL_NUM_TRAIN_EPOCHS,
            "learning_rate": FULL_LEARNING_RATE,
            "per_device_train_batch_size": TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": EVAL_BATCH_SIZE,
            "trainable_params": count_trainable_parameters(trainer.model),
        },
        metrics=metrics,
    )
    print("Logged full fine-tune run to ./mlruns")


if __name__ == "__main__":
    main()
