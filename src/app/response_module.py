import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model once at import
CHECKPOINT = "stress_model/checkpoint-273225"  # Last checkpoint route
model     = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def get_supportive_response(text: str) -> tuple[str, float]:
    """
    Analyze input text for stress and return a supportive response.

    Args:
        text (str): Input user message.

    Returns:
        response (str): Supportive message based on stress level.
        score (float): Probability score for stress class (1).
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0, 1].item()

    # Define stress levels
    if probs < 0.5:
        level = "no_stress"
    elif probs < 0.7:
        level = "low_stress"
    else:
        level = "high_stress"

    # Predefined responses
    RESPONSES = {
        "no_stress": [
            "Great to hear you're feeling calm! Keep it up.",
            "You seem at ease today. Stay positive!",
            "Awesome, it looks like you're in a good spot."
        ],
        "low_stress": [
            "I sense a bit of tension. Taking a deep breath might help.",
            "It seems you're slightly stressed. A short break can be refreshing.",
            "Would you like to try a quick relaxation exercise?"
        ],
        "high_stress": [
            "I understand things feel overwhelming. I'm here to listen.",
            "You're under a lot of pressure right now. Remember it’s okay to seek help.",
            "It sounds intense. How about pausing for a moment and breathing deeply?"
        ]
    }

    response = random.choice(RESPONSES[level])
    return response, probs

# Simple tests
if __name__ == "__main__":
    samples = [
        "I am so frustrated and can't calm down.",
        "Today was smooth and enjoyable.",
        "Feeling a little anxious about tomorrow's meeting."
    ]
    for text in samples:
        resp, score = get_supportive_response(text)
        print(f"Input: {text}\nScore: {score:.2f}\nResponse: {resp}\n")
