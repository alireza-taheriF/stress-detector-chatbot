import json
import os
import random

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load from HuggingFace Hub. This full fine-tune stays the default.
MODEL_ID = "avangard90/stress-detector-chatbot-model"
BASE_MODEL_NAME = "distilbert-base-uncased"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LORA_ADAPTER_DIR = os.path.join(REPO_ROOT, "artifacts", "lora-distilbert")

_full_bundle = None
_lora_bundle = None


def get_full_bundle():
    """Load the current full fine-tune from the Hugging Face model id."""
    global _full_bundle
    if _full_bundle is None:
        from huggingface_hub import file_exists

        if not file_exists(MODEL_ID, "config.json"):
            raise FileNotFoundError(
                f"{MODEL_ID} does not contain model files, so the full fine-tune "
                "cannot be loaded."
            )
        full_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        full_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        full_model.eval()
        _full_bundle = (full_tokenizer, full_model)
    return _full_bundle


def lora_adapter_available(adapter_dir=None) -> bool:
    directory = adapter_dir or LORA_ADAPTER_DIR
    if not os.path.isdir(directory):
        return False
    names = set(os.listdir(directory))
    has_weights = "adapter_model.safetensors" in names or "adapter_model.bin" in names
    return "adapter_config.json" in names and has_weights


def load_lora_model(adapter_dir=None):
    """Load base DistilBERT and attach the saved LoRA adapter."""
    from peft import PeftModel

    directory = adapter_dir or LORA_ADAPTER_DIR
    with open(os.path.join(directory, "adapter_config.json"), encoding="utf-8") as handle:
        adapter_cfg = json.load(handle)
    base_name = adapter_cfg.get("base_model_name_or_path") or BASE_MODEL_NAME
    if os.path.isfile(os.path.join(directory, "tokenizer_config.json")):
        lora_tokenizer = AutoTokenizer.from_pretrained(directory)
    else:
        lora_tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=2,
    )
    lora_model = PeftModel.from_pretrained(base_model, directory)
    lora_model.eval()
    return lora_tokenizer, lora_model


def get_lora_bundle():
    global _lora_bundle
    if _lora_bundle is None:
        _lora_bundle = load_lora_model()
    return _lora_bundle


def get_supportive_response(text: str, model_key: str = "full") -> tuple[str, float]:
    if model_key == "lora":
        active_tokenizer, active_model = get_lora_bundle()
    else:
        active_tokenizer, active_model = get_full_bundle()

    inputs = active_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = active_model(**inputs).logits
        score = torch.softmax(logits, dim=1)[0, 1].item()

    # Determine stress level
    if score < 0.5:
        level = "no_stress"
    elif score < 0.7:
        level = "low_stress"
    else:
        level = "high_stress"

    RESPONSES = {
        "no_stress": [
            "Great to hear you're feeling calm! Keep it up.",
            "You seem at ease today. Stay positive!",
            "Awesome, it looks like you're in a good spot.",
        ],
        "low_stress": [
            "I sense a bit of tension. Taking a deep breath might help.",
            "It seems you're slightly stressed. A short break can be refreshing.",
            "Would you like to try a quick relaxation exercise?",
        ],
        "high_stress": [
            "I understand things feel overwhelming. I'm here to listen.",
            "You're under a lot of pressure right now. Remember it’s okay to seek help.",
            "It sounds intense. How about pausing for a moment and breathing deeply?",
        ],
    }

    return random.choice(RESPONSES[level]), score
