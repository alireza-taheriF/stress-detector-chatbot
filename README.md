# Stress Detector Chatbot

A conversational chatbot that detects stress in user messages using DistilBERT and provides supportive responses.  
Built with Python, HuggingFace Transformers, and Streamlit.

## Features
- 🧠 **Stress detection** (binary classification) with a fine-tuned DistilBERT model  
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
│   │   ├── train_finetune.py
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
├── training.log
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

# Fine-tuning DistilBERT
python train_finetune.py

# Confusion matrix & ROC
python analyze_results.py

# Baseline comparison
python baseline_lr.py

python log_responses.py
# Fill in logs/test_logs.csv with your inputs/outputs and CSV‑based feedback

streamlit run app.py

Examples
Validation Metrics for DistilBERT:
Accuracy ≈ 0.8245, F1 ≈ 0.8446, AUC ≈ 0.9052

Baseline (LR+TF‑IDF):
Accuracy ≈ 0.7950, F1 ≈ 0.8171, AUC ≈ 0.8765


License
This project is licensed under the MIT License. See LICENSE for details.

