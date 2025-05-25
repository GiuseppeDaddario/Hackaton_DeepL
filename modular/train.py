import torch
from tqdm import tqdm

def train(train_acc_cater, train_loader, model, optimizer, device, optimizer_overparametrization, train_loss, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(tqdm(train_loader, desc="Iterating training graphs", unit="batch")):
        inputs, labels = data.to(device), data.y.to(device)

        # Per CrossEntropyLoss i label devono essere class indices, non one-hot
        target = torch.zeros(len(labels), 6).to(device).scatter_(1, labels.view(-1, 1).long(), 1)

        # Estrai gli indici dei grafi nel batch
        indices_batch = data.batch.unique().tolist()

        # Risali al dataset originale (in caso data_loader.dataset sia un Subset o simile)
        dataset_obj = train_loader.dataset
        while hasattr(dataset_obj, 'dataset'):
            dataset_obj = dataset_obj.dataset

        # Usa dict_index per tradurre gli indici batch in indici originali
        index_run = [dataset_obj.dict_index[idx] for idx in indices_batch]

        optimizer.zero_grad()
        if optimizer_overparametrization is not None:
            optimizer_overparametrization.zero_grad()

        outputs, emb, _ = model(inputs)

        # Scegli il tipo di loss in base al tipo di oggetto
        if isinstance(train_loss, torch.nn.CrossEntropyLoss):
            loss = train_loss(outputs, labels)  # labels deve essere shape (batch_size,)
        else:
            # ncodLoss o altra loss custom
            loss = train_loss(index_run, outputs, target, emb, i, current_epoch, train_acc_cater)

        loss.backward()
        optimizer.step()
        if optimizer_overparametrization is not None:
            optimizer_overparametrization.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels.squeeze()).sum().item()
        running_loss += loss.item()

    train_acc_cater = correct_train / total_train

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return (correct_train / total_train) * 100, (running_loss / len(train_loader)), train_acc_cater