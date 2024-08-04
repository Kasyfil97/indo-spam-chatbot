import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

class DatasetLoader:
    def __init__(self, config):
        self.config = config

    def load_dataset(self, text_col: str, target_col: str, convert_target: bool = True):
        data = pd.read_excel(self.config['train_data_path'], sheet_name=self.config['sheet_name'])
        data = data[[text_col, target_col]]
        data[text_col] = data[text_col].astype(str)
        if convert_target:
            data[target_col] = data[target_col].map(self.config['class_dict'])
        print(f'Loaded {data.shape[0]} data points')
        return data
    
    def create_dataset(self, data):
        df_train, df_val = train_test_split(data, test_size=self.config['val_size'], random_state=42)
        print(f'train: {df_train.shape}')
        print(f'train: {df_val.shape}')
        dataset_train = Dataset.from_pandas(df_train).shuffle(seed=42)
        dataset_val = Dataset.from_pandas(df_val)
        return DatasetDict({'train': dataset_train, 'val': dataset_val})