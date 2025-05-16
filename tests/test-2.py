import pandas as pd
import os

# مسیرهای احتمالی
possible_paths = [
'/Users/alireza/Documents/project/Suicide_Detection copy.csv'
]
found = False
for path in possible_paths:
    if os.path.exists(path):
        try:
            data = pd.read_csv(path, encoding='latin-1')
            print(f"فایل یافت شد: {path}")
            print(data.head())
            found = True
            break
        except Exception as e:
            print(f"خطا در خواندن فایل {path}: {str(e)}")
            
if not found:
    print("فایل در هیچ یک از مسیرهای زیر یافت نشد:")
    for path in possible_paths:
        print(f"- {path}")
    print("\nراهنمایی:")
    print("1. نام فایل را دقیقاً بررسی کنید")
    print("2. از تب 'Finder' در مک، مسیر دقیق را کپی کنید")
    print("3. فایل را به محل ساده‌تری مثل دسکتاپ منتقل کنید")