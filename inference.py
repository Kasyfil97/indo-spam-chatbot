import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pprint import pprint

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map='auto')
    model = model.to(device)
    return model, tokenizer

if __name__ == '__main__': 
    model_path = ''
    model, tokenizer = load_model_tokenizer(model_path)
    
    message = input("Enter your message: ")
    tokenized_message = tokenizer.encode(message, return_tensors="pt")
    tokenized_message=tokenized_message.to(device)

    with torch.no_grad():
        output = model(tokenized_message)
        label = torch.argmax(output.logits, dim=-1).item()
    print(f"{'spam' if label==1 else 'non-spam'}")