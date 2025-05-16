from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

# 1. تابع برای بارگذاری داده با اعتبارسنجی
def load_data(file_path):
    """بارگذاری داده از CSV و بررسی وجود ستون clean_text"""
    df = pd.read_csv(file_path)
    
    # بررسی وجود ستون clean_text
    if 'clean_text' not in df.columns:
        raise ValueError(f"ستون 'clean_text' در فایل {file_path} وجود ندارد")
    
    # حذف ردیف‌های خالی و تبدیل به رشته
    df = df.dropna(subset=['clean_text'])
    df['clean_text'] = df['clean_text'].astype(str)
    
    return df

# بارگذاری داده‌های آموزشی و اعتبارسنجی
df_train = load_data("train.csv")
df_val = load_data("val.csv")

# 2. تبدیل به Dataset با انتخاب ستون‌های ضروری
dataset_train = Dataset.from_pandas(df_train[['clean_text', 'label']])
dataset_val = Dataset.from_pandas(df_val[['clean_text', 'label']])

# 3. انتخاب مدل و توکنایزر
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 4. تابع توکنایز اصلاح شده
def tokenize_function(batch):
    """تابع پردازش متن برای مدل HuggingFace"""
    try:
        # تبدیل متن به رشته و پردازش دسته‌ای
        texts = [str(text) for text in batch['clean_text']]
        return tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"  # ضروری برای PyTorch
        )
    except Exception as e:
        print(f"خطا در پردازش دسته: {e}")
        raise

# 5. اعمال توکنایزر روی داده‌ها
dataset_train = dataset_train.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=['clean_text']  # حذف متن اصلی
)

dataset_val = dataset_val.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=['clean_text']
)

# 6. تغییر نام ستون برچسب برای سازگاری با مدل
dataset_train = dataset_train.rename_column("label", "labels")
dataset_val = dataset_val.rename_column("label", "labels")

# 7. تنظیم فرمت خروجی برای PyTorch
dataset_train.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels']
)

dataset_val.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels']
)

# 8. ذخیره داده‌های پردازش شده
dataset_train.save_to_disk("processed_train")
dataset_val.save_to_disk("processed_val")

print("پردازش داده‌ها با موفقیت انجام شد! ✅")