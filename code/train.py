import torch
import argparse

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW, BertForSequenceClassification
from transformers import AutoTokenizer

from dataloader import FriendsDataset


def main(training_args, tokenizer):
    # Load data
    dataset = FriendsDataset(json_file='data/json/friends_season_01.json', tokenizer=tokenizer)
    dataloader = DataLoader(dataset, shuffle=True, num_workers=0)

    # Load model
    # model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=dataset.num_labels())
    model = BertForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=dataset.num_labels())

    # Simple trainer
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(training_args.num_train_epochs):
        for id, utterance, speaker in dataloader:
            optim.zero_grad()
            input_ids = id.to(device)
            attention_mask = utterance.to(device)
            labels = speaker.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
        print(f"Epoch: {epoch}")
        model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add parameters for training.')
    parser.add_argument('--batch_size', type=int, default=4, help='the batch size')
    # parser.add_argument('--model_checkpoint', type=str, default='distilbert-base-uncased', help='specify the model checkpoint')
    parser.add_argument('--model_checkpoint', type=str, default='bert-base-uncased', help='specify the model checkpoint')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='the learning rate')
    args = parser.parse_args()

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=args.batch_size,   # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)

    main(training_args, tokenizer)
