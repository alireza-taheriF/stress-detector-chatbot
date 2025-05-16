import pandas as pd
from sklearn.model_selection import train_test_split

#1. Loading a hybrid dataframe
df = pd.read_csv("combined_dataset.csv")

# 2. Separate the train (80%) and temporary (20%) sets
df_train, df_temp = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df['label']   # Maintain label distribution
)

# From df_temp, separate 50% for validation (10% of total) and 50% for test (10% of total)
df_val, df_test = train_test_split(
    df_temp,
    test_size=0.50,
    random_state=42,
    stratify=df_temp['label']
)

# 3. Saving files
df_train.to_csv("train.csv", index=False)
df_val.to_csv("val.csv", index=False)
df_test.to_csv("test.csv", index=False)

#4. Show the size of each set for certainty
print("Train size:", len(df_train))
print("Validation size:", len(df_val))
print("Test size:", len(df_test))
print(df_train['label'].value_counts(normalize=True))
print(df_val['label'].value_counts(normalize=True))
print(df_test['label'].value_counts(normalize=True))
