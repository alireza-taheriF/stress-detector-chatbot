import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. Definition of the set of English stopwords
STOPWORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    """Text Clearing Function (Same as before)"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join([w for w in text.split() if w not in STOPWORDS])

# 2. Loading a dataset with an absolute path
file_path = "training.1600000.processed.noemoticon.csv"
df2 = pd.read_csv(
    file_path,
    encoding='latin1', # This dataset usually works with this encoding
    names=['polarity','id','date','query','user','text']
)

# 3. Delete rows without text
df2 = df2.dropna(subset=['text'])

# 4. Apply the cleanup function
df2['clean_text'] = df2['text'].apply(clean_text)

# 5. Save the result
output_path = "/Users/alireza/Documents/project/training_cleaned.csv"
df2.to_csv(output_path, index=False)

# 6. Show the first 5 rows
print(df2[['text','clean_text']].head())
