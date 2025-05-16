import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load from HuggingFace Hub
MODEL_ID  = "avangard90/stress-detector-chatbot-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load from HuggingFace Hub
MODEL_ID  = "avangard90/stress-detector-chatbot-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

def get_supportive_response(text: str) -> tuple[str, float]:
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        score  = torch.softmax(logits, dim=1)[0,1].item()

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

    return random.choice(RESPONSES[level]), score

