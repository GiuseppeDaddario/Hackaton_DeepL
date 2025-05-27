import torch
from tqdm import tqdm
import torch.nn.functional as F

def evaluate_model(
        model,
        loader,
        device,
        criterion_obj,
        criterion_type="gcod",
        num_classes_dataset=None,
        lambda_l3_weight=0.0,
        current_epoch_for_gcod=-1,
        atrain_for_gcod=0.0,
        is_validation=False
):

    model.eval()  # Imposta il modello in modalità valutazione
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions_list = [] # Per il test set finale

    with torch.no_grad(): # Disabilita il calcolo dei gradienti
        for batch_idx, data_batch in enumerate(tqdm(loader, desc="Evaluating", unit="batch", leave=False, disable=not is_validation)):
            data_batch = data_batch.to(device)
            true_labels_int = data_batch.y.to(device) # (N,)

            output_logits, graph_embeddings, _ = model(data_batch) # (N, C), (N, emb_dim)

            # Calcolo della Loss (solo per validazione)
            if is_validation:
                if criterion_type == "ce":
                    # criterion_obj qui è LabelSmoothingCrossEntropy o nn.CrossEntropyLoss
                    loss = criterion_obj(output_logits, true_labels_int)
                elif criterion_type == "gcod":
                    # gcodLoss richiede più input
                    if not hasattr(data_batch, 'original_idx'):
                        raise ValueError("Per GCOD in valutazione, 'original_idx' deve essere presente nel batch.")
                    batch_original_indices = data_batch.original_idx.view(-1).tolist()
                    true_labels_one_hot = F.one_hot(true_labels_int, num_classes=num_classes_dataset).float()


                    l1, l2, l3 = criterion_obj.calculate_loss_components(
                        batch_original_indices=batch_original_indices,
                        gnn_logits_batch=output_logits,
                        true_labels_batch_one_hot=true_labels_one_hot,
                        gnn_embeddings_batch=graph_embeddings,
                        batch_iter_num=batch_idx,
                        current_epoch=current_epoch_for_gcod,
                        atrain_overall_accuracy=atrain_for_gcod
                    )

                    loss = l1 + lambda_l3_weight * l3
                else:
                    raise ValueError(f"Unsupported criterion_type: {criterion_type} in evaluation.")
                total_loss += loss.item() * data_batch.num_graphs # Moltiplica per la dimensione del batch

            _, predicted_labels = torch.max(output_logits, 1) # (N,)

            if is_validation:
                correct_predictions += (predicted_labels == true_labels_int).sum().item()
            else:
                all_predictions_list.extend(predicted_labels.cpu().numpy())

            total_samples += data_batch.num_graphs

    if is_validation:
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        return avg_loss, accuracy
    else:
        return all_predictions_list