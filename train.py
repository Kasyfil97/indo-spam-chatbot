import os
import wandb
from dotenv import load_dotenv
from transformers import DataCollatorWithPadding
import torch


from config.config import Config
from device_manager import DeviceManager
from data.data import DatasetLoader
from model.tokenizer import TokenizerLoader, DatasetTokenizer
from model.model import Model
from trainer.trainer import ModelTrainer
from gcs.model_saver_uploader import ModelSaverUploader
from model import *

torch.cuda.set_device(0)

if __name__ == '__main__':
    config = Config()
    wandb_config = config['wandb_config']

    load_dotenv()
    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        project=wandb_config['project'],
        job_type=wandb_config['job_type'],
        anonymous=wandb_config['anounymous'],
        name=wandb_config['run_name'],
        notes=wandb_config['notes']
    )

    # Create device map
    device_map = DeviceManager.create_device_map(config['model']['gpu_map'])

    # Load and preprocess dataset
    data_handler = DatasetLoader(config['dataset_config'])
    data = data_handler.load_dataset(text_col = 'Message', target_col = 'Label')
    dataset = data_handler.create_dataset(data)

    # Load tokenizer
    tokenizer_loader = TokenizerLoader(config['model']['base_model'])
    tokenizer = tokenizer_loader()

    # Tokenize dataset
    dataset_tokenizer = DatasetTokenizer(tokenizer, config['model']['max_len'])
    tokenized_datasets = dataset_tokenizer.tokenized_dataset(dataset)

    # Load model
    model = Model(config['model'], tokenizer, device_map=device_map)
    model_llama = model.load_model(
        score_layer_to_unfreeze=config['model']['score_layer_to_unfreeze'], 
        num_last_block_to_unfreeze=config['model']['num_last_block_to_unfreeze']
    )
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    # Configure and create trainer
    model_trainer = ModelTrainer(config['training_config'])
    
    print('Begin training')
    try:
        print('train model: ', config['model']['base_model'])
        trainer = model_trainer.train(
            tokenized_datasets, 
            model_llama, 
            tokenizer, 
            collate_fn)
    except KeyboardInterrupt:
        print("\nProgram terminated by user")

    # Save and upload model
    saver_uploader = ModelSaverUploader(config)
    saver_uploader.save_model(trainer)
    saver_uploader.upload_to_gcs()

    wandb.finish()
