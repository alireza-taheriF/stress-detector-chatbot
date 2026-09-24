import json
import os

import pytest
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from train_common import log_mlflow_run
from train_lora import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    build_lora_config,
    build_lora_model,
    comparison_payload,
    save_lora_adapter,
)


def _bare_distilbert():
    return DistilBertForSequenceClassification(DistilBertConfig(num_labels=2))


def test_lora_trainable_params_far_below_full_distilbert(tmp_path):
    config = build_lora_config()
    assert config.r == LORA_R == 8
    assert config.lora_alpha == LORA_ALPHA == 16
    assert config.lora_dropout == LORA_DROPOUT == 0.1
    assert set(config.target_modules) == set(LORA_TARGET_MODULES)
    assert set(LORA_TARGET_MODULES) == {"q_lin", "k_lin", "v_lin", "out_lin"}

    base = _bare_distilbert()
    full_count = sum(param.numel() for param in base.parameters())
    model = build_lora_model(base)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    ratio = trainable / full_count

    assert trainable > 0
    assert ratio < 0.05

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    for projection in ("q_lin", "k_lin", "v_lin", "out_lin"):
        assert any(projection in name and "lora_" in name for name in trainable_names)

    frozen_names = [name for name, param in model.named_parameters() if not param.requires_grad]
    assert any("q_lin" in name and "base_layer" in name for name in frozen_names)

    save_lora_adapter(model, tmp_path)
    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["base_model_name_or_path"] == "distilbert-base-uncased"
    names = {path.name for path in tmp_path.iterdir()}
    assert "adapter_config.json" in names
    assert "adapter_model.safetensors" in names or "adapter_model.bin" in names
    assert "model.safetensors" not in names
    assert "pytorch_model.bin" not in names
    adapter_bytes = sum(path.stat().st_size for path in tmp_path.iterdir() if path.is_file())
    assert adapter_bytes < full_count * 4 * 0.2


def test_comparison_payload_fields():
    payload = comparison_payload(
        full_metrics={"accuracy": 0.8245, "f1": 0.8446, "auc": 0.9052},
        full_trainable=1000,
        full_checkpoint_size=2000,
        lora_metrics={"accuracy": 0.1, "f1": 0.2, "auc": 0.3},
        lora_trainable=10,
        adapter_size=20,
    )
    assert set(payload) == {"full_finetune", "lora"}
    for block in payload.values():
        assert set(block) >= {"accuracy", "f1", "auc", "trainable_params"}
    assert payload["full_finetune"]["checkpoint_size_bytes"] == 2000
    assert payload["lora"]["adapter_size_bytes"] == 20
    assert payload["full_finetune"]["accuracy"] == 0.8245
    assert payload["lora"]["trainable_params"] < payload["full_finetune"]["trainable_params"]


def test_mlflow_logs_to_local_file_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "adapter"
    artifact.mkdir()
    (artifact / "adapter_config.json").write_text("{}", encoding="utf-8")
    log_mlflow_run(
        run_name="lora-distilbert",
        params={"method": "lora", "r": 8},
        metrics={"accuracy": 0.5, "f1": 0.4, "auc": 0.6},
        artifact_dir=str(artifact),
    )
    store = tmp_path / "mlruns"
    assert store.is_dir()
    metric_files = list(store.rglob("metrics/*"))
    assert {path.name for path in metric_files} >= {"accuracy", "f1", "auc"}
    param_method = next(store.rglob("params/method")).read_text(encoding="utf-8")
    assert "lora" in param_method
    assert any(path.name == "adapter_config.json" for path in store.rglob("adapter_config.json"))


@pytest.mark.skipif(
    os.environ.get("RUN_LORA_TRAIN") != "1",
    reason="set RUN_LORA_TRAIN=1 to run LoRA training",
)
def test_lora_training():
    from train_lora import main

    main()
