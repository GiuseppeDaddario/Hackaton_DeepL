import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from src.loss import gcodLoss

def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    accuracy = 0.0
    f1 = 0.0
    
    if criterion == "ce":
        
        criterion = torch.nn.CrossEntropyLoss()
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


    elif criterion == "gcod":
        for batch_iter_num, data in enumerate(tqdm(data_loader, desc="TRAINING -->", unit="batch")):
            data = data.to(device)
            optimizer.zero_grad()
            # output: logits, embeddings, _
            output_logits, gnn_embeddings = model(data, return_embedding=True)
            # Prepara le label one-hot
            true_labels_batch_one_hot = torch.nn.functional.one_hot(data.y, num_classes=output_logits.size(1)).float()
            # Indici originali (assicurati che data abbia .idx, altrimenti usa range)
            batch_original_indices = data.idx if hasattr(data, "idx") else torch.arange(data.y.size(0), device=data.y.device)
            # Calcola accuracy globale sul train (puoi aggiornarla ogni epoca, qui metti pure 1.0 come placeholder)
            atrain_overall_accuracy = 1.0  # Sostituisci con il valore reale se lo calcoli
            current_epoch = current_epoch  # già passato come argomento

            
            
            loss_function = gcodLoss( sample_labels_numpy, device, num_examp, num_classes, gnn_embedding_dim, total_epochs)
            l1_loss, l2_loss, l3_loss = loss_function.calculate_loss_components(
                batch_original_indices,
                output_logits,
                true_labels_batch_one_hot,
                gnn_embeddings,
                batch_iter_num,
                current_epoch,
                atrain_overall_accuracy
            )
            batch_loss = l1_loss + l2_loss + l3_loss
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

            # Calcola predizioni e accuracy
            pred = output_logits.argmax(dim=1)
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