import pandas as pd
from response_module import get_supportive_response

def main():
    # 1. لیست تست‌ها
    test_sentences = [
        "I have an important interview tomorrow, my hands are freezing and my heart is pounding!",
        "I took a hot bath tonight and listened to some relaxing music. It made me feel better!",
        "Give me space; work, studies, relationships... it's like the world is falling on me!",
        # ... لیست کامل ۲۰ جمله ...
    ]
    
    # 2. اجرا و جمع‌آوری نتایج
    records = []
    for sent in test_sentences:
        resp, score = get_supportive_response(sent)
        records.append({
            "input": sent,
            "stress_score": score,
            "response": resp
        })
    
    # 3. ساخت DataFrame و ذخیره
    df_logs = pd.DataFrame(records)
    df_logs.to_csv("logs/test_logs.csv", index=False)
    print("✅ Logs saved to logs/test_logs.csv")

if __name__ == "__main__":
    main()
