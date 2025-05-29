import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for data in tqdm(data_loader, desc="TRAINING -->", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        ## Calculate predictions and accuracy 
        pred = output.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")



    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy, f1