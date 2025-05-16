from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

#1. Function to load data with validation
def load_data(file_path):
    """Loading data from CSV and checking for the existence of the clean_text column"""
    df = pd.read_csv(file_path)
    
    # Check for the existence of the clean_text column
    if 'clean_text' not in df.columns:
        raise ValueError(f"Column 'clean_text' does not exist in file {file_path}")
    
    # Remove empty rows and convert to string
    df = df.dropna(subset=['clean_text'])
    df['clean_text'] = df['clean_text'].astype(str)
    
    return df

# Loading training and validation data
df_train = load_data("train.csv")
df_val = load_data("val.csv")

# 2. Convert to Dataset by selecting the necessary columns
dataset_train = Dataset.from_pandas(df_train[['clean_text', 'label']])
dataset_val = Dataset.from_pandas(df_val[['clean_text', 'label']])

#3. Choosing a model and tokenizer
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#4. Modified tokenization function
def tokenize_function(batch):
    """Text processing function for the HuggingFace model"""
    try:
       # Convert text to string and batch processing
        texts = [str(text) for text in batch['clean_text']]
        return tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"  # Essential for PyTorch
        )
    except Exception as e:
        print(f"Error in batch processing: {e}")
        raise

# 5. Applying tokenizer to data
dataset_train = dataset_train.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=['clean_text']  # Delete the original text
)

dataset_val = dataset_val.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=['clean_text']
)

#6. Rename the label column to match the model
dataset_train = dataset_train.rename_column("label", "labels")
dataset_val = dataset_val.rename_column("label", "labels")

#7. Setting the output format for PyTorch
dataset_train.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels']
)

dataset_val.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels']
)

# 8. Storing processed data
dataset_train.save_to_disk("processed_train")
dataset_val.save_to_disk("processed_val")

print("Data processing was successful! ✅")
