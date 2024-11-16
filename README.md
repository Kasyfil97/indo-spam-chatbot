# Indo Spam Chatbot

## Introduction

Indo Spam Chatbot is a fine-tuned spam detection model based on the **Gemma 2 2B** architecture. This model is specifically designed to identify spam messages in WhatsApp chatbot interactions. It has been fine-tuned using a dataset of 40,000 spam messages collected over a year. The dataset consists of two labels: **spam** and **non-spam**.

The model supports detecting spam across multiple categories, including:
- Offensive and abusive words
- Profane language
- Gibberish words and numbers
- Spam links
- And more

This repository provides tools for fine-tuning, inference, and evaluating the model, along with configuration options for customization.

---

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Ensure that you are using Python 3.10 or higher for compatibility with the packages.

## Training the Model

If you wish to retrain the model, you can use the train.py script. Configure the training parameters by editing the config.yaml file. The configurable options include:

- Training and testing data paths
- Model architecture and hyperparameters
- Training settings (e.g., learning rate, epochs)

To start training, run:
```
python train.py
```
## Running Inference

For spam detection on new messages, use the inference.py script. This script takes input messages, processes them through the model, and outputs whether each message is spam or non-spam.

To run inference, execute:
```
python inference.py
```
Ensure that the model weights are properly downloaded and configured before running inference.

## Model Weights

The trained model weights are available for download on Hugging Face

## Example Use Cases

This model can be used in various chatbot systems to:

    Filter out spam messages in real time
    Improve user experience by blocking inappropriate content
    Identify and flag suspicious links or messages

Feel free to explore the example scripts and adapt them to your needs.