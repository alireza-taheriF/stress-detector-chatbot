import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load data
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 2. FinTuned data
CHECKPOINT = "stress_model/checkpoint-273225"  # Last checkpoint route
model     = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
model.eval()



# 2. validation (Or test) 
df_val = pd.read_csv("val.csv")

# 3. Preparing inputs
texts = df_val["clean_text"].astype(str).tolist()
labels = df_val["label"].values

# 4. Tokenization and inference in small batches (batches)
all_probs = []
all_preds = []
batch_size = 64

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i : i + batch_size]
    inputs = tokenizer(
        batch_texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=1)[:,1].cpu().numpy()  # probability of class 1
    preds = (probs >= 0.5).astype(int)
    all_probs.extend(probs)
    all_preds.extend(preds)

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
# Confusion Matrix
cm = confusion_matrix(labels, all_preds)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(labels, all_preds, digits=4))
# ROC Curve
fpr, tpr, thresholds = roc_curve(labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0,1], [0,1], 'k--')  # خط قطری
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

print(f"AUC: {roc_auc:.4f}")
