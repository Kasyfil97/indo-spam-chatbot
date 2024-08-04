import os
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import json
import datetime
from transformers import AutoModelForSequenceClassification

from model.model import Model
from model.tokenizer import TokenizerLoader, DatasetTokenizer

from config import Config

from pprint import pprint

if __init__ == '__main__': 

    #load config
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    message = input("Enter your message: ")

    model_path = config['model']['new_model']

    tokenizer = TokenizerLoader(config['model']['base_model'])
    message_tokenizer = DatasetTokenizer(tokenizer, max_len=config['model']['max_len'])
    tokenized_message = message_tokenizer.tokenizer_func(message)
    inputs = {k: v.to(device) for k, v in tokenized_message.items()}

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device) 

    category_map = {logit: category for category, logit in config['dataset_config']['class_dict'].items()}

    with torch.no_grad():
        outputs = model(**inputs)

    print(outputs)