import torch
import argparse

from torch.utils.data import DataLoader

try:
  from transformers import TrainingArguments, AdamW, BertForSequenceClassification
except:
  from transformers import TrainingArguments, AdamW, BertForSequenceClassification
  
from dataloader import FriendsDataset, create_splits


def evaluate(model, test_loader, device):

    if not model:
        model = BertForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=7)
        model.load_state_dict(torch.load('./results_2.pth'))
        model.to(device)

    correct = torch.zeros((6))
    total = torch.zeros((6))

    for id, utterance, speaker in test_loader:
        input_ids = id.to(device)
        attention_mask = utterance.to(device)
        labels = speaker.to(device)
        outputs = model(input_ids)
        _, predicted = torch.max(outputs.logits, 1)

        for pred, label in zip(predicted, labels):
            if pred == label:
                correct[label] += 1
            total[label] += 1

        # print("label:", labels, "predicted:", predicted, "correct?", (predicted == labels).sum().item())
    print(correct / total)
    print(correct.sum() / total.sum())


def main(training_args, args):
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
    ])

    # Load model
    model = BertForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=dataset.num_labels())

    # Load data
    train_set, val_set, test_set = create_splits(dataset, [0.8, 0.1, 0.1])
    train_loader = DataLoader(train_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=0)

    # Simple trainer
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)  
    model.train()

    optim = AdamW(model.parameters(), lr=training_args.learning_rate)
    running_loss = 0.0

    for epoch in range(training_args.num_train_epochs):
        for i, (id, utterance, speaker) in enumerate(train_loader):
            optim.zero_grad()
            input_ids = id.to(device)
            attention_mask = utterance.to(device)
            labels = speaker.to(device)
                    
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

                model.eval()
                evaluate(model, test_loader, device)
                model.train()
                
                
        print(f"Epoch: {epoch}")
        model.eval()
        evaluate(model, test_loader, device)
        torch.save(model.state_dict(), training_args.output_dir + "_" + str(epoch) + ".pth")
        model.train()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add parameters for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch size')
    parser.add_argument('--model_checkpoint', type=str, default='bert-base-uncased', help='specify the model checkpoint')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='the learning rate')
    args = parser.parse_args()

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=1000,              # total number of training epochs
        per_device_train_batch_size=args.batch_size,   # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        learning_rate=args.learning_rate
    )

    main(training_args, args)