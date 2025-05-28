import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F




## INFERENCE FUNCTION
def evaluate_model(
        model,
        loader,
        device,
        criterion_obj=None, # Può essere None se is_validation è False
        criterion_type="gcod",
        num_classes_dataset=None, # Necessario solo se is_validation e criterion_type="gcod"
        lambda_l3_weight=0.0,   # Necessario solo se is_validation e criterion_type="gcod"
        current_epoch_for_gcod=-1, # Necessario solo se is_validation e criterion_type="gcod"
        atrain_for_gcod=0.0,       # Necessario solo se is_validation e criterion_type="gcod"
        is_validation=False
):

    model.eval()  # Imposta il modello in modalità valutazione
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions_list = []
    all_labels = []
    desc_str = "Validating" if is_validation else "Testing (Predicting)"

    with torch.no_grad(): # Disable gradient computation for evaluation
        for batch_idx, data_batch in enumerate(tqdm(loader, desc=desc_str, unit="batch", leave=False, disable=not is_validation)):
            data_batch = data_batch.to(device)

            # Inference on the dataset
            output_logits, graph_embeddings, _ = model(data_batch) # (N, C), (N, emb_dim)
            _, predicted_labels = torch.max(output_logits, 1) # (N,) # takes the max probable label.
            



            ## VALIDATION
            if is_validation:
                if not hasattr(data_batch, 'y'):
                    raise ValueError("Per la validazione (is_validation=True), 'data_batch.y' (etichette vere) deve essere presente.")

                true_labels_int = data_batch.y.to(device)
                all_labels.extend(true_labels_int.cpu().numpy())
                correct_predictions += (predicted_labels == true_labels_int).sum().item()
                
                if criterion_obj is None:
                    raise ValueError("Per la validazione (is_validation=True), 'criterion_obj' deve essere fornito.")

                # LOSS RECALL
                if criterion_type == "ce":
                    # criterion_obj qui è LabelSmoothingCrossEntropy o nn.CrossEntropyLoss
                    loss = criterion_obj(output_logits, true_labels_int)
                
                ## GCOD RECALL
                elif criterion_type == "gcod":
                    if not hasattr(data_batch, 'original_idx'):
                        raise ValueError("Per GCOD in valutazione, 'original_idx' deve essere presente nel batch.")
                    if num_classes_dataset is None:
                        raise ValueError("Per GCOD in valutazione, 'num_classes_dataset' deve essere fornito.")

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
                    loss = l1 + lambda_l3_weight * l3 # l2 non è usato qui, come nell'originale
                else:
                    raise ValueError(f"Unsupported criterion_type: {criterion_type} in evaluation.")

                total_loss += loss.item() * data_batch.num_graphs # Moltiplica per la dimensione del batch
                all_predictions_list.extend(predicted_labels.cpu().numpy())
            # END OF VALIDATION
            
            
            ## JUST PREDICTIONS (NO metrics)
            else:
                # --- Logica specifica per il TEST (solo predizioni) ---
                all_predictions_list.extend(predicted_labels.cpu().numpy()) #Append predictions of the batch

            total_samples += data_batch.num_graphs # Sum up the number of graphs labelled in the batch


    ## RETURN METRICS
    if is_validation:
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        correct_predictions += (predicted_labels == true_labels_int).sum().item()
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        f1 = f1_score(all_labels, all_predictions_list, average='macro')
        return avg_loss, accuracy, f1
    
    

    ## RETURN PREDICTIONS
    else:
        return all_predictions_list