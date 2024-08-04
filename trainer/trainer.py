from transformers import TrainingArguments
from custom_trainer import CustomTrainer
from sklearn.metrics import accuracy_score
import numpy as np

class ModelTrainer:
    def __init__(self, config):
        self.training_args = TrainingArguments(
            output_dir=config['output_dir'],
            learning_rate=config['learning_rate'],
            per_device_train_batch_size=config['per_device_train_batch_size'],
            per_device_eval_batch_size=config['per_device_eval_batch_size'],
            num_train_epochs=config['num_train_epochs'],
            weight_decay=config['weight_decay'],
            logging_strategy=config['logging_strategy'],
            evaluation_strategy=config['eval_strategy'],
            save_strategy=config['save_strategy'],
            load_best_model_at_end=config['load_best_model_at_end'],
            optim=config['optim'],
            report_to=config['report_to'],
            save_total_limit=config['save_limit']
        )
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'val_accuracy': accuracy_score(predictions, labels)}
            
    def train(self, tokenized_datasets, model, tokenizer, collate_fn):
        trainer = CustomTrainer(
            model=model,
            args=self.training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['val'],
            tokenizer=tokenizer,
            data_collator=collate_fn,
            compute_metrics=self.compute_metrics
        )
        trainer.train()
        return trainer