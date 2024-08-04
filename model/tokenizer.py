from transformers import AutoTokenizer
from datasets import Dataset

class TokenizerLoader:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self):
        print('Loading tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

class DatasetTokenizer:
    def __init__(self, tokenizer: AutoTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def tokenizer_func(self, examples: str):
            return self.tokenizer(examples['Message'],
                                  padding='max_length',
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=self.max_len,
                                  add_special_tokens=False)
        
    def tokenized_dataset(self, dataset: Dataset):
        
        tokenized_datasets = dataset.map(self.tokenizer_func, batched=True, remove_columns='__index_level_0__')
        tokenized_datasets = tokenized_datasets.rename_column("Label", "label")
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return tokenized_datasets