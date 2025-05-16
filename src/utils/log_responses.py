import pandas as pd
from response_module import get_supportive_response

def main():
    # 1. List of tests
    test_sentences = [
        "I have an important interview tomorrow, my hands are freezing and my heart is pounding!",
        "I took a hot bath tonight and listened to some relaxing music. It made me feel better!",
        "Give me space; work, studies, relationships... it's like the world is falling on me!",
        "My dog is sick and I can't pay the veterinary costs.",  
        "I just got promoted at work today.",  
        "I keep thinking about yesterday's argument with my best friend.", 
        "I have to finish three reports by the end of the day.",  
        "The sunset today was incredibly beautiful.",  
        "Why does my chest feel tight when I'm stressed?",  
        "I completely forgot our wedding anniversary.",  
        "I learned how to cook a new dish today.",  
        "I keep making errors in my work reports.",  
        "I can't decide between two job offers.",  

    ]
    
    #2. Execution and collection of results
    records = []
    for sent in test_sentences:
        resp, score = get_supportive_response(sent)
        records.append({
            "input": sent,
            "stress_score": score,
            "response": resp
        })
    
    #3. Build and save DataFrame
    df_logs = pd.DataFrame(records)
    df_logs.to_csv("logs/test_logs.csv", index=False)
    print("✅ Logs saved to logs/test_logs.csv")

if __name__ == "__main__":
    main()
