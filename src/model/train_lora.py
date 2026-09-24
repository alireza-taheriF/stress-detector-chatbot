"""LoRA fine-tune of DistilBERT on the same splits as the full model.

The full fine-tune in train_finetune.py stays the baseline. This script trains
only a PEFT adapter on the attention projections and writes a comparison file
plus two local MLflow runs (baseline metrics and this adapter).
"""

import json
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from train_common import (
    COMPARE_PATH,
    EVAL_BATCH_SIZE,
    FULL_BASELINE_METRICS,
    FULL_LEARNING_RATE,
    FULL_NUM_TRAIN_EPOCHS,
    LORA_ADAPTER_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_EPOCHS,
    LORA_LEARNING_RATE,
    LORA_OUTPUT_DIR,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_NAME,
    NUM_LABELS,
    TRAIN_BATCH_SIZE,
    compute_metrics,
    count_trainable_parameters,
    directory_size_bytes,
    full_checkpoint_size_bytes,
    latest_eval_metrics,
    load_splits,
    log_mlflow_run,
)


def build_lora_config():
    """LoRA on DistilBERT attention projections, plus the classification head."""
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=list(LORA_TARGET_MODULES),
        modules_to_save=["pre_classifier", "classifier"],
    )


def build_lora_model(base_model=None):
    if base_model is None:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
        )
    # PEFT copies model.name_or_path into the adapter config and treats "" as missing.
    base_name = (
        getattr(base_model, "name_or_path", None)
        or getattr(base_model.config, "name_or_path", None)
        or MODEL_NAME
    )
    base_model.name_or_path = base_name
    base_model.config.name_or_path = base_name
    config = build_lora_config()
    config.base_model_name_or_path = base_name
    return get_peft_model(base_model, config)


def comparison_payload(
    full_metrics,
    full_trainable,
    full_checkpoint_size,
    lora_metrics,
    lora_trainable,
    adapter_size,
):
    return {
        "full_finetune": {
            "accuracy": float(full_metrics["accuracy"]),
            "f1": float(full_metrics["f1"]),
            "auc": float(full_metrics["auc"]),
            "trainable_params": int(full_trainable),
            "checkpoint_size_bytes": int(full_checkpoint_size),
        },
        "lora": {
            "accuracy": float(lora_metrics["accuracy"]),
            "f1": float(lora_metrics["f1"]),
            "auc": float(lora_metrics["auc"]),
            "trainable_params": int(lora_trainable),
            "adapter_size_bytes": int(adapter_size),
        },
    }


def write_comparison(payload, path=COMPARE_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def save_lora_adapter(model, adapter_dir, tokenizer=None):
    """Save the PEFT adapter and, when given, the tokenizer needed to reload it."""
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    readme = os.path.join(adapter_dir, "README.md")
    if os.path.isfile(readme):
        os.remove(readme)
    if tokenizer is not None:
        tokenizer.save_pretrained(adapter_dir)


def _training_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _eval_metrics(trainer):
    metrics = latest_eval_metrics(trainer.state.log_history)
    if "accuracy" in metrics and "f1" in metrics and "auc" in metrics:
        return metrics
    evaluated = trainer.evaluate()
    return {
        "accuracy": float(evaluated["eval_accuracy"]),
        "f1": float(evaluated["eval_f1"]),
        "auc": float(evaluated["eval_auc"]),
    }


def main():
    train_ds, val_ds = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    full_trainable = count_trainable_parameters(base_model)
    full_checkpoint_size = full_checkpoint_size_bytes(base_model)

    model = build_lora_model(base_model)
    lora_trainable = count_trainable_parameters(model)
    model.print_trainable_parameters()
    device = _training_device()
    model.to(device)

    use_cuda = device.type == "cuda"
    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT_DIR,
        num_train_epochs=LORA_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(LORA_OUTPUT_DIR, "logs"),
        logging_steps=10,
        learning_rate=LORA_LEARNING_RATE,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        fp16=use_cuda,
        bf16=device.type == "mps",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    checkpoints = []
    if os.path.isdir(LORA_OUTPUT_DIR):
        checkpoints = [
            entry.path
            for entry in os.scandir(LORA_OUTPUT_DIR)
            if entry.is_dir() and entry.name.startswith("checkpoint")
        ]
    last_checkpoint = max(checkpoints, key=os.path.getctime) if checkpoints else None

    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except KeyboardInterrupt:
        print("\nTraining stopped. Saving the current adapter...")
        save_lora_adapter(trainer.model, LORA_ADAPTER_DIR, tokenizer)
        print(f"Adapter saved in '{LORA_ADAPTER_DIR}'.")
        return

    lora_metrics = _eval_metrics(trainer)
    save_lora_adapter(trainer.model, LORA_ADAPTER_DIR, tokenizer)
    adapter_size = directory_size_bytes(LORA_ADAPTER_DIR)

    payload = comparison_payload(
        full_metrics=FULL_BASELINE_METRICS,
        full_trainable=full_trainable,
        full_checkpoint_size=full_checkpoint_size,
        lora_metrics=lora_metrics,
        lora_trainable=lora_trainable,
        adapter_size=adapter_size,
    )
    write_comparison(payload)
    print(f"Wrote {COMPARE_PATH}")
    print(
        "Full fine-tune baseline "
        f"accuracy={payload['full_finetune']['accuracy']:.4f} "
        f"f1={payload['full_finetune']['f1']:.4f} "
        f"auc={payload['full_finetune']['auc']:.4f}"
    )
    print(
        "LoRA "
        f"accuracy={payload['lora']['accuracy']:.4f} "
        f"f1={payload['lora']['f1']:.4f} "
        f"auc={payload['lora']['auc']:.4f}"
    )

    log_mlflow_run(
        run_name="full-finetune-baseline",
        params={
            "model_name": MODEL_NAME,
            "num_labels": NUM_LABELS,
            "method": "full_finetune",
            "metrics_source": "reported_validation_baseline",
            "num_train_epochs": FULL_NUM_TRAIN_EPOCHS,
            "learning_rate": FULL_LEARNING_RATE,
            "per_device_train_batch_size": TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": EVAL_BATCH_SIZE,
            "trainable_params": full_trainable,
            "checkpoint_size_bytes": full_checkpoint_size,
        },
        metrics=FULL_BASELINE_METRICS,
    )
    log_mlflow_run(
        run_name="lora-distilbert",
        params={
            "model_name": MODEL_NAME,
            "num_labels": NUM_LABELS,
            "method": "lora",
            "r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": ",".join(LORA_TARGET_MODULES),
            "num_train_epochs": LORA_EPOCHS,
            "learning_rate": LORA_LEARNING_RATE,
            "per_device_train_batch_size": TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": EVAL_BATCH_SIZE,
            "trainable_params": lora_trainable,
            "adapter_size_bytes": adapter_size,
        },
        metrics=lora_metrics,
        artifact_dir=LORA_ADAPTER_DIR,
    )
    print("Logged full-finetune-baseline and lora-distilbert to ./mlruns")


if __name__ == "__main__":
    main()
