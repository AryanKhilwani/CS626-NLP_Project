from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# Load GoEmotions dataset
dataset = load_dataset("go_emotions")

# Check dataset structure
print(dataset)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert labels to multi-label format
tokenized_datasets = tokenized_datasets.map(lambda x: {"labels": [1 if i in x["labels"] else 0 for i in range(28)]})
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

# Prepare train, validation, and test datasets
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=28,  # GoEmotions has 28 emotion labels
    problem_type="multi_label_classification"
)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

loss_function = nn.BCEWithLogitsLoss()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to GPU

# Training loop
def train_model(model, train_loader, optimizer, loss_function, scheduler, epochs=1):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(train_loader, leave=True)
        
        for batch in loop:
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float()
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())
        
        scheduler.step()
        print(f"Epoch {epoch} Loss: {total_loss / len(train_loader)}")


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Train the model
train_model(model, train_loader, optimizer, loss_function, scheduler, epochs=1)

# Save the model and tokenizer
model.save_pretrained("goemotions_bert_finetuned")
tokenizer.save_pretrained("goemotions_bert_finetuned")

# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    true_labels = []
    predictions = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float()
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.sigmoid(logits).cpu().numpy() > 0.5
            
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds)
    
    # Compute metrics
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    f1 = f1_score(true_labels, predictions, average="macro")
    precision = precision_score(true_labels, predictions, average="macro")
    recall = recall_score(true_labels, predictions, average="macro")
    
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

# Evaluate the model
evaluate_model(model, val_loader)

# Predict function
def predict_emotions(model, tokenizer, text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()  # Move back to CPU
    return probs > 0.5  # Return multi-label predictions

# Example prediction
text = "I am so happy and excited!"
predictions = predict_emotions(model, tokenizer, text)
print(predictions)
