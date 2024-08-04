# finetune llama-3-8b for multiclass prediction task

This repository contains code for fine-tuning the Llama 3 8B model for multiclass classification using a conventional method.

## Project Overview
In this project, we fine-tune the Llama 3 8B model to perform multiclass classification tasks. Fine-tuning a large language model (LLM) like Llama 3 8B involves adjusting the model's weights based on the specific task at hand. For classification tasks, this usually entails modifying the model's output layer and training it with labeled data to predict the correct class.

Fine-tuning an LLM for classification can be done by:

- Modifying the output layer to match the number of classes in the target dataset.
- Training the model on the target dataset, where only a few layers (typically the final layer and a few preceding layers) are updated to avoid overfitting and to leverage the pre-trained knowledge of the model.

This project follows the conventional approach to fine-tuning, ensuring that the model retains its pre-trained knowledge while adapting to the new classification task.

## Requirements

Before you begin, ensure you have met the following requirements:

- Python 3.x
- Required libraries listed in requirements.txt

You can install the required libraries using:

```shell
pip install -r requirements.txt
```

## Training

To train the model, you can use the train.py script. This script does not take any arguments. Below is an example of how to run the training script:

```shell
python train.py
```

The `train.py` script will:
- Load the training data from the specified directory.
- Configure the Llama 3 8B model for multiclass classification.
- Train the model using the training data.
- Save the trained model to the models directory or upload it to the gcs.

To train the model, it is important to note that it requires a GPU with a memory capacity of at least 35 GB. Therefore, it is recommended to use a GPU like the NVIDIA A100 for optimal performance.

### Example
```shell
python train.py
```

This will start the training process using the predefined configurations in the `train.py` script.

## Inference

To perform inference using the trained model, you can use the inference.py script. This script does not take any arguments. Below is an example of how to run the inference script:

```shell
python inference.py
```

The inference.py script will:
- Load the trained model from the models directory.
- Perform inference and output the predictions.

### Example
```shell
python inference.py
```

This will load the trained model and perform inference on the provided input data.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

