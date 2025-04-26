from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import json
import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from intent_classifier import INTENTS, IntentClassifier
import pandas as pd

def main():
    # Load or create training data
    intent_examples = IntentClassifier()._load_examples()
    
    # Prepare data for training
    texts = []
    labels = []
    
    for intent_idx, intent in enumerate(INTENTS):
        for example in intent_examples[intent]:
            texts.append(example)
            labels.append(intent_idx)
    
    # Create dataset
    df = pd.DataFrame({"text": texts, "label": labels})
    
    # Split into train and validation
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=len(INTENTS)
    )
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
    
    # Apply tokenization
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Define compute metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted")
        }
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./intent_model_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./intent_model_logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    os.makedirs("data/intent_model", exist_ok=True)
    model.save_pretrained("data/intent_model")
    tokenizer.save_pretrained("data/intent_model")
    
    print("Intent classification model trained and saved!")

if __name__ == "__main__":
    main()
