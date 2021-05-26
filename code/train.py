import torch
import argparse

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from dataloader import FriendsDataset, create_splits

try:
  from transformers import AdamW, BertForSequenceClassification
except:
  from transformers import AdamW, BertForSequenceClassification


def evaluate(model, test_loader, device):

    if not model:
        model = BertForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=7)
        model.load_state_dict(torch.load('./results_lr_5e5_concatenated_7.pth'))
        model.to(device)

    correct = torch.zeros((6))
    total = torch.zeros((6))
    
    pred_list = []
    label_list = []

    for id, utterance, speaker in test_loader:
        input_ids = id.to(device)
        attention_mask = utterance.to(device)
        labels = speaker.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        _, predicted = torch.max(outputs[1], 1)

        for pred, label in zip(predicted, labels):
        
            pred_list.append(int(pred))
            label_list.append(int(label))        
        
            if pred == label:
                correct[label] += 1
            total[label] += 1

        # print("label:", labels, "predicted:", predicted, "correct?", (predicted == labels).sum().item())
    print(correct / total)
    print(correct.sum() / total.sum())
    print(confusion_matrix(pred_list, label_list))
    

def main(args):
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
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Simple trainer
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)  
    model.train()
    
    # Evaluation
    only_eval = True
    if only_eval:
        model.eval()
        evaluate(None, test_loader, device)
        quit()

    optim = AdamW(model.parameters(), lr=args.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    running_loss = 0.0

    for epoch in range(args.num_train_epochs):
        for i, (id, utterance, speaker) in enumerate(train_loader):

            optim.zero_grad()
            input_ids = id.to(device)
            attention_mask = utterance.to(device)
            labels = speaker.to(device)
                    
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs[0]
            loss.backward()
            optim.step()
            running_loss += loss.item()

            # Print loss
            if i % 25 == 24:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 25))
                running_loss = 0.0

            # Print intermediate evaluation
            if i % 500 == 499:
                model.eval()
                evaluate(model, test_loader, device)
                model.train()
                
        print(f"Epoch: {epoch}")
        model.eval()
        evaluate(model, test_loader, device)
        torch.save(model.state_dict(), args.output_dir + "_lr_5e5_concatenated_" + str(epoch) + ".pth")
        model.train()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add parameters for training.')
    parser.add_argument('--batch_size', type=int, default=6, help='the batch size')
    parser.add_argument('--model_checkpoint', type=str, default='bert-base-uncased', help='specify the model checkpoint')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='the learning rate')
    parser.add_argument('--output_dir', type=str, default="./results", help='the learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='the learning rate')
    args = parser.parse_args()

    main(args)