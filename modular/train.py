import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(train_acc_cater,data_loader, model,model_sp, optimizer, device,optimizer_overparametrization,train_loss, train_dataset, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    model_sp.train()

    total_loss = 0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(tqdm(data_loader, desc="Iterating training graphs", unit="batch")):
        inputs, labels = data.to(device), data.y.to(device)
        target = torch.zeros(len(labels), 6).to(device).scatter_(1, labels.view(-1, 1).long(), 1)

        # Estrai gli indici dei grafi nel batch
        indices_batch = data.batch.unique().tolist()

        # Risali al dataset originale (in caso data_loader.dataset sia un Subset o simile)
        dataset_obj = data_loader.dataset
        while hasattr(dataset_obj, 'dataset'):
            dataset_obj = dataset_obj.dataset

        # Usa dict_index per tradurre gli indici batch in indici originali
        index_run = [dataset_obj.dict_index[idx] for idx in indices_batch]


        outs_sp, _, _ = model_sp(inputs)
        prediction = F.softmax(outs_sp, dim=1)
        prediction = torch.sum((prediction * target), dim=1)
        train_loss.weight[index_run] = (prediction.detach()).view(-1, 1)

        optimizer.zero_grad()
        optimizer_overparametrization.zero_grad()
        outputs, emb, _ = model(inputs)
        loss = train_loss(index_run, outputs, target, emb, i, current_epoch,train_acc_cater)
        loss.backward()
        optimizer.step()
        optimizer_overparametrization.step()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels.squeeze()).sum().item()
        total_loss += loss.item()

    train_acc_cater = correct_train / total_train

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    loss_return = (correct_train / total_train) * 100, (total_loss / len(data_loader))
    return loss_return, train_acc_cater