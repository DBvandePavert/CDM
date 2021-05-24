import torch
import argparse

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW, BertForSequenceClassification
from transformers import AutoTokenizer

from dataloader import FriendsDataset, create_splits


def main(training_args, tokenizer):
    # Load data
    dataset = FriendsDataset([
        'data/json/friends_season_01.json',
        'data/json/friends_season_02.json',
        'data/json/friends_season_03.json',
        'data/json/friends_season_04.json',
        'data/json/friends_season_05.json',
        'data/json/friends_season_06.json',
        'data/json/friends_season_07.json',
        'data/json/friends_season_08.json',
        'data/json/friends_season_09.json',
        'data/json/friends_season_10.json'
    ], tokenizer=tokenizer)
    train_set, val_set, test_set = create_splits(dataset, [0.05, 0.9, 0.05])
    train_loader = DataLoader(train_set, shuffle=True, num_workers=0)
    test_loader =  DataLoader(test_set, shuffle=True, num_workers=0)

    # Load model
    # model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=dataset.num_labels())
    model = BertForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=dataset.num_labels())

    # Simple trainer
    torch.cuda.init()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(training_args.num_train_epochs):
        for id, utterance, speaker in train_loader:
            optim.zero_grad()
            input_ids = id.to(device)
            attention_mask = utterance.to(device)
            labels = speaker.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
        print(f"Epoch: {epoch}")

    torch.save(model.state_dict(), training_args.output_dir)

    print("evaluation start")
    evaluate(model, test_loader)


def evaluate(model, test_loader):

    if not model:
        model = BertForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=7)
        model.load_state_dict(torch.load('./results'))

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            input_ids, attention_mask, labels = data
            outputs = model(input_ids)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(str(correct) + " / " + str(total))

    print('Accuracy: %d %%' % (
            100 * correct / total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add parameters for training.')
    parser.add_argument('--batch_size', type=int, default=4, help='the batch size')
    parser.add_argument('--model_checkpoint', type=str, default='bert-base-uncased', help='specify the model checkpoint')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='the learning rate')
    args = parser.parse_args()

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=1,              # total number of training epochs
        per_device_train_batch_size=args.batch_size,   # batch size per device during training
        per_device_eval_batch_size=128,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)

    main(training_args, tokenizer)

    # model = BertForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=7)
    # model.load_state_dict(torch.load('./results'))
    # model.eval()
