from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from config.config import Config


class Model:
    def __init__(self, config: Config, tokenizer: AutoTokenizer, device_map = "auto"):
        self.model = config['base_model']
        self.tokenizer = tokenizer
        self.num_labels = config['num_labels']
        self.device_map = device_map
    
    def load_model(self, num_last_block_to_unfreeze = None, score_layer_to_unfreeze = True):
        model = AutoModelForSequenceClassification.from_pretrained(self.model, num_labels=self.num_labels, device_map = self.device_map)

        for param in model.parameters():
            param.requires_grad = False
        
        if score_layer_to_unfreeze:
            for param in model.score.parameters():
                param.requires_grad = True
        if num_last_block_to_unfreeze is not None:
            for param in model.model.layers[-num_last_block_to_unfreeze:].parameters():
                param.requires_grad = True
        
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.use_cache = False
        model.config.pretraining_tp = 1
    
        total_params = 0
        trainable_count = 0

        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_count += param.numel()

        print(f"Total parameters: {total_params} | Trainable parameters: {trainable_count} | Percentage: {trainable_count/total_params *100}%")
        return model