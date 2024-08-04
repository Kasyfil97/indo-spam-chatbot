import shutil
from upload_gcs import upload_directory_to_gcs

class ModelSaverUploader:
    def __init__(self, config):
        self.config = config

    def save_model(self, trainer):
        new_model_path = self.config['model']['new_model']
        trainer.save_model(new_model_path)
        print(f"Model has been saved in {new_model_path}")

    def upload_to_gcs(self):
        config_gcs = self.config['blob_name']
        model_blob_name = config_gcs['model_blob_name']
        ckpnt_blob_name = config_gcs['ckpnt_blob_name']

        dir_path = self.config['training_config']['output_dir']
        upload_directory_to_gcs(dir_path, ckpnt_blob_name)
        try:
            shutil.rmtree(dir_path)
            print(f"{dir_path} and all its contents have been removed successfully.")
        except Exception as e:
            print(f"Error: {e}")

        upload_directory_to_gcs(self.config['model']['new_model'], model_blob_name)