import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. تعریف مجموعه‌ی stopwords انگلیسی
STOPWORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    """تابع پاک‌سازی متن (مشابه قبل)"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join([w for w in text.split() if w not in STOPWORDS])

# 2. بارگذاری دیتاست با مسیر مطلق
file_path = "/Users/alireza/Documents/project/training.1600000.processed.noemoticon.csv"
df2 = pd.read_csv(
    file_path,
    encoding='latin1',  # این دیتاست معمولاً با این انکدینگ کار می‌کند
    names=['polarity','id','date','query','user','text']
)

# 3. حذف ردیف‌های بدون متن
df2 = df2.dropna(subset=['text'])

# 4. اعمال تابع پاک‌سازی
df2['clean_text'] = df2['text'].apply(clean_text)

# 5. ذخیره‌ی نتیجه
output_path = "/Users/alireza/Documents/project/training_cleaned.csv"
df2.to_csv(output_path, index=False)

# 6. نمایش ۵ ردیف اول
print(df2[['text','clean_text']].head())