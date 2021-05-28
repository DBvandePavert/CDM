import torch
import argparse
import numpy as np
from collections import Counter

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from dataloader import FriendsDataset, create_splits
from helpers import get_number_of_speakers

try:
  from transformers import AdamW, BertForSequenceClassification
except:
  from transformers import AdamW, BertForSequenceClassification


def evaluate(model, test_loader, device):
    if not model:
        model = BertForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=7)
        model.load_state_dict(torch.load('./results_lr_2e5_other_multi_paper_3.pth'))
        model.to(device)

    correct = torch.zeros((7))
    total = torch.zeros((7))
    
    pred_list = []
    label_list = []
    score_list = []
    long_list = []

    with torch.no_grad():
        for id, utterance, speaker in test_loader:
            input_ids = id.to(device)
            attention_mask = utterance.to(device)
            labels = speaker.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            _, predicted = torch.max(outputs[1], 1)

            for pred, label, ids, conf in zip(predicted, labels, input_ids, torch.max(outputs[1], 1).values):
                score_list.append([conf, ids[ids.nonzero()].T[0], label])
            
                pred_list.append(int(pred))
                label_list.append(int(label))        
            
                if pred == label:
                    correct[label] += 1
                else:
                    long_list.append([ids[ids.nonzero()].T[0].shape[0], ids[ids.nonzero()].T[0]])
                total[label] += 1

        # Confidence
        score = np.array(score_list)
        print("Most conf")
        print(score[np.argsort(score[:, 0])][:10])
        print("Least Conf")
        print(score[np.argsort(score[:, 0])][-10:])
        
        # Longest sentence
        print("longest")
        longest = np.array(long_list)
        print(longest[np.argsort(longest[:, 0])][:10])

        print(correct / total)
        print(correct.sum() / total.sum())
        print("conf", confusion_matrix(pred_list, label_list))

        # Detect number of speakers per scene
        speakers_correct = []
        speakers_total = []

        for uids, ids, utterance, speaker in test_loader:

            input_ids = ids.to(device)
            attention_mask = utterance.to(device)
            labels = speaker.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            _, predicted = torch.max(outputs[1], 1)

            for uid, pred, label in zip(uids, predicted, labels):
                scene_id = uid[:11]
                number_of_speakers = get_number_of_speakers(scene_id)

                if pred == label:
                    correct[label] += 1
                    speakers_correct.append(number_of_speakers)
                speakers_total.append(number_of_speakers)
                total[label] += 1

        speakers_correct_count = Counter(speakers_correct)
        speakers_total_count = Counter(speakers_total)
        speaker_distribution = []
        speaker_acc = []

        for i in range(1, 8):
            total = speakers_total_count[i]
            speaker_distribution.append(total)
            if total:
                speaker_acc.append(speakers_correct_count[i] / total)
            else:
                speaker_acc.append(0)
        
        print(speaker_distribution)
        print(speaker_acc)

def main(args):
    torch.manual_seed(args.seed)

    eval = True

    # Load data
    dataset = FriendsDataset(return_uids=eval)

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
    only_eval = eval
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
        torch.save(model.state_dict(), args.output_dir + "_lr_2e5_other_multi_" + str(epoch) + ".pth")
        model.train()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add parameters for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch size')
    parser.add_argument('--model_checkpoint', type=str, default='bert-base-uncased', help='specify the model checkpoint')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='the learning rate')
    parser.add_argument('--output_dir', type=str, default="./results", help='the learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='the learning rate')
    parser.add_argument('--seed', type=int, default=321, help='the seed')
    args = parser.parse_args()

    main(args)
