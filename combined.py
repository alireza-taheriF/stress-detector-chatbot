import pandas as pd
from sklearn.utils import shuffle

# 1. بارگذاری دیتافریم‌های تمیزشده
df_suicide = pd.read_csv("suicide_detection_cleaned.csv")
df_tweets  = pd.read_csv("training_cleaned.csv")

# 2. برچسب‌گذاری دیتافریم Suicide
df_suicide['label'] = 1

# 3. فیلتر و برچسب‌گذاری دیتافریم توییت‌ها
df_t = df_tweets[df_tweets['polarity'].isin([0,4])].copy()
df_t['label'] = df_t['polarity'].apply(lambda p: 1 if p==0 else 0)

# 4–5. انتخاب فقط ستون‌های لازم
df_suicide = df_suicide[['clean_text','label']]
df_t      = df_t[['clean_text','label']]

# 6. ادغام
df_combined = pd.concat([df_suicide, df_t], ignore_index=True)

# 7. Shuffle
df_combined = shuffle(df_combined, random_state=42)

# 8. ذخیره
df_combined.to_csv("combined_dataset.csv", index=False)

# 9. نمایش نمونه
print(df_combined.head())
