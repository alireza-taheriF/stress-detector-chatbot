import pandas as pd
from sklearn.model_selection import train_test_split

# 1. بارگذاری دیتافریم ترکیبی
df = pd.read_csv("combined_dataset.csv")

# 2. جدا کردن مجموعه‌ی train (80%) و موقت (20%)
df_train, df_temp = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df['label']  # حفظ توزیع برچسب‌ها
)

# از df_temp، 50% رو برای validation (10% کل) و 50% برای test (10% کل) جدا کن
df_val, df_test = train_test_split(
    df_temp,
    test_size=0.50,
    random_state=42,
    stratify=df_temp['label']
)

# 3. ذخیره‌ی فایل‌ها
df_train.to_csv("train.csv", index=False)
df_val.to_csv("val.csv", index=False)
df_test.to_csv("test.csv", index=False)

# 4. نمایش سایز هر مجموعه برای اطمینان
print("Train size:", len(df_train))
print("Validation size:", len(df_val))
print("Test size:", len(df_test))
print(df_train['label'].value_counts(normalize=True))
print(df_val['label'].value_counts(normalize=True))
print(df_test['label'].value_counts(normalize=True))
