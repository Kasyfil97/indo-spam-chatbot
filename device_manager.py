import torch

class DeviceManager:
    @staticmethod
    def create_device_map():
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            device_map = {f"gpu{i}": i for i in range(n_gpus)}
            return device_map
        else:
            raise RuntimeError("No CUDA available")