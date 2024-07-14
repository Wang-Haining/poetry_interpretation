import pandas as pd
from datasets import Dataset, DatasetDict

CSV_PATH = 'data/poem_content_h2.csv'
df = pd.read_csv(CSV_PATH)

# filter out rows where 'Summary' field contains 'No \'Summary\' h2 tag found.'
df_filtered = df[~df['Summary'].str.contains("No 'Summary' h2 tag found")]

# split the dataset into training (80%), validation (10%), and test (10%) sets
train_df = df_filtered.sample(frac=0.8, random_state=42).reset_index(drop=True)
temp_df = df_filtered.drop(train_df.index).reset_index(drop=True)
val_df = temp_df.sample(frac=0.5, random_state=42).reset_index(drop=True)
test_df = temp_df.drop(val_df.index).reset_index(drop=True)

# convert DataFrames to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# create a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# save the dataset to the specified path
DATASET_PATH = 'data/poem_content_h2'
dataset_dict.save_to_disk(DATASET_PATH)
