import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW
from transformers import AutoTokenizer

from dataloader import FriendsDataset

# Config
batch_size = 1
num_labels = 7

# Load model
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# Load data
dataset = FriendsDataset(json_file='data/json/friends_season_01.json', tokenizer=tokenizer)
dataloader = DataLoader(dataset, shuffle=True, num_workers=0)  # Some weird stuff with the inclusion of batch sizes

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=batch_size,   # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Simple trainer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(1):
    for id, utterance, speaker in dataloader    :
        print(id)
        print(utterance)
        print(speaker)
        optim.zero_grad()
        input_ids = id.to(device)
        attention_mask = utterance.to(device)
        labels = speaker.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        quit()

model.eval()
