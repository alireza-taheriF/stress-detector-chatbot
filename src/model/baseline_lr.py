# baseline_lr.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt

def main():
    # Load Data
    df_train = pd.read_csv("train.csv")
    df_val   = pd.read_csv("val.csv")
    df_test  = pd.read_csv("test.csv")
    
    # Clean text
    for df in (df_train, df_val, df_test):
        df.dropna(subset=['clean_text'], inplace=True)
        # If the clean_text column also contains empty strings, you want to remove them:
        df = df[df['clean_text'].str.strip() != '']
    
    # Clean text, lables column
    X_train_texts, y_train = df_train["clean_text"], df_train["label"]
    X_val_texts,   y_val   = df_val["clean_text"],   df_val["label"]
    X_test_texts,  y_test  = df_test["clean_text"],  df_test["label"]
    
    # 2. TF‑IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        lowercase=True
    )
    X_train = vectorizer.fit_transform(X_train_texts)
    X_val   = vectorizer.transform(X_val_texts)
    X_test  = vectorizer.transform(X_test_texts)
    
    # 3. Train Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    
    # 4. Evaluation
    y_val_pred = lr.predict(X_val)
    y_val_prob = lr.predict_proba(X_val)[:,1]
    
    print("=== Validation Metrics (Baseline LR) ===")
    print(f"Accuracy : {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"Recall   : {recall_score(y_val, y_val_pred):.4f}")
    print(f"F1-score : {f1_score(y_val, y_val_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print("Confusion Matrix:\n", cm)
    
    # ROC & AUC
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    roc_auc = auc(fpr, tpr)
    print(f"AUC      : {roc_auc:.4f}")
    
    # ROC
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'LR AUC = {roc_auc:.4f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.title("ROC Curve (Logistic Regression)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    # 5. DistilBERT
    print("\n--- Comparison (Validation) ---")
    print("DistilBERT: accuracy≈0.8245, F1≈0.8446, AUC≈0.9052")
    print(f"LR Baseline: accuracy={accuracy_score(y_val, y_val_pred):.4f}, "
          f"F1={f1_score(y_val, y_val_pred):.4f}, AUC={roc_auc:.4f}")

if __name__ == "__main__":
    main()
