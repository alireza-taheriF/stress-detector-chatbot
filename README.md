# Stress Detector Chatbot

A conversational chatbot that detects stress in user messages using DistilBERT and provides supportive responses.  
Built with Python, HuggingFace Transformers, and Streamlit.

## Features
- 🧠 **Stress detection** (binary classification) with a fine-tuned DistilBERT model  
- 🧩 **LoRA adapter** trained on the same splits, with the full fine-tune kept as the baseline  
- 📊 **Baseline comparison** against TF‑IDF + Logistic Regression  
- 💬 **Supportive response module** with three stress levels (no/low/high)  
- 🖥️ **Interactive web UI** built with Streamlit  

## Project Structure

```bash
stress-detector-chatbot/
│
├── src/
│   ├── data_processing/
│   │   ├── clean_data.py
│   │   └── combined.py
│   ├── model/
│   │   ├── train_model.py
│   │   ├── train_common.py
│   │   ├── train_finetune.py
│   │   ├── train_lora.py
│   │   └── baseline_lr.py
│   ├── app/
│   │   ├── app.py
│   │   └── response_module.py
│   └── utils/
│       ├── analyze_results.py
│       └── log_responses.py
├── assets/
│   ├── Figure_1.png
│   └── Figure_1-1.png
├── requirements.txt
├── README.md
└── .gitignore
```
## Installation

```bash
git clone https://github.com/your-username/stress-detector-chatbot.git
cd stress-detector-chatbot
python3 -m venv venv
source venv/bin/activate   # Linux / Mac
# .\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Data cleaning & combination
python clean_data.py
python combined.py

# Tokenization & dataset prep
python train_model.py

# Fine-tuning DistilBERT (baseline; unchanged)
python src/model/train_finetune.py

# LoRA adapter on the same processed splits (short run)
python src/model/train_lora.py

# Confusion matrix & ROC
python analyze_results.py

# Baseline comparison
python baseline_lr.py

python log_responses.py
# Fill in logs/test_logs.csv with your inputs/outputs and CSV‑based feedback

streamlit run src/app/app.py
```

## Full fine-tune baseline

The full DistilBERT fine-tune remains the baseline. `src/model/train_finetune.py` is unchanged in what it trains and where it writes checkpoints (`stress_model/`). Reported validation metrics:

Accuracy ≈ 0.8245, F1 ≈ 0.8446, AUC ≈ 0.9052

Train it from the repository root (after `processed_train` and `processed_val` exist):

```bash
python src/model/train_finetune.py
```

## LoRA adapter

`src/model/train_lora.py` imports the shared splits, labels, and metrics from `src/model/train_common.py` (the same pieces `train_finetune.py` uses). It does not build a second preprocessing pipeline. LoRA targets the DistilBERT attention projections `q_lin`, `k_lin`, `v_lin`, and `out_lin` with defaults `r=8`, `lora_alpha=16`, `lora_dropout=0.1`, and 1 epoch so a laptop can finish.

Exact train command, from the repository root:

```bash
python src/model/train_lora.py
```

That command writes:

- `artifacts/lora-distilbert/` — adapter weights only, plus the tokenizer files needed to reload them on top of `distilbert-base-uncased`
- `artifacts/compare_full_vs_lora.json` — both metric sets, trainable parameter counts, and adapter size on disk versus the full checkpoint size
- `./mlruns` — local MLflow file store (no remote tracking server). The training scripts opt into that file store for current MLflow releases. A `full-finetune-baseline` run and a `lora-distilbert` run record params, metrics, and, for LoRA, the adapter artifact.

`full_finetune` metrics in the JSON are the reported validation baseline above. Trainable-parameter counts are measured. `checkpoint_size_bytes` is the on-disk size of the latest `stress_model/checkpoint-*` directory when that checkpoint exists; otherwise it is the fp32 parameter byte size of the full classification model. `adapter_size_bytes` is the on-disk size of `artifacts/lora-distilbert/`.

### `artifacts/compare_full_vs_lora.json` fields

- `full_finetune.accuracy`
- `full_finetune.f1`
- `full_finetune.auc`
- `full_finetune.trainable_params`
- `full_finetune.checkpoint_size_bytes`
- `lora.accuracy`
- `lora.f1`
- `lora.auc`
- `lora.trainable_params`
- `lora.adapter_size_bytes`

Running `train_finetune.py` also logs its own `full-finetune` MLflow run under `./mlruns`. Checkpoints, downloaded datasets, and `mlruns/` are gitignored.

```bash
python -m pytest
```

The training test is skipped unless `RUN_LORA_TRAIN=1`.

In the Streamlit app, the detection-model selector defaults to **Full fine-tune**. Choosing **LoRA adapter** loads base DistilBERT plus `artifacts/lora-distilbert/`. If that folder is absent, the default full model is unchanged.

Baseline (LR+TF‑IDF):
Accuracy ≈ 0.7950, F1 ≈ 0.8171, AUC ≈ 0.8765


License
This project is licensed under the MIT License. See LICENSE for details.

