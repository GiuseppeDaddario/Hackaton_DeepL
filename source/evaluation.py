import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

eps = 1e-7

def evaluate_model(
        model,
        loader,
        device,
        criterion_obj=None,
        criterion_type="gcod",
        num_classes_dataset=None,
        lambda_l3_weight=0.0,
        current_epoch_for_gcod=-1,
        atrain_for_gcod=0.0,
        is_validation=False
):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions_list = []
    all_labels = []
    desc_str = "Validating" if is_validation else "Testing (Predicting)"

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(tqdm(loader, desc=desc_str, unit="batch", leave=False, disable=not is_validation)):
            data_batch = data_batch.to(device)

            output_logits, graph_embeddings, _ = model(data_batch)

            # Controlla NaN/Inf nei logits PRIMA di calcolare predicted_labels
            if torch.isnan(output_logits).any() or torch.isinf(output_logits).any():
                print(f"WARNING: NaN/Inf in output_logits during {desc_str} at batch {batch_idx}. Skipping loss calculation & metrics for this batch if validation.")
                total_samples += data_batch.num_graphs # Contiamo comunque i grafi totali
                if is_validation:
                    if hasattr(data_batch, 'y') and data_batch.y is not None:
                        all_labels.extend(data_batch.y.cpu().numpy())
                        # Aggiungiamo predizioni fittizie per non sfalsare la lunghezza delle liste per F1
                        all_predictions_list.extend(np.zeros(data_batch.num_graphs, dtype=int))
                continue

            _, predicted_labels = torch.max(output_logits, 1)

            if is_validation:
                if not hasattr(data_batch, 'y') or data_batch.y is None:
                    print(f"WARNING: Missing true labels (data_batch.y) in validation for batch {batch_idx}. Skipping.")
                    total_samples += data_batch.num_graphs
                    all_predictions_list.extend(predicted_labels.cpu().numpy())
                    continue

                true_labels_int = data_batch.y.to(device)
                all_labels.extend(true_labels_int.cpu().numpy())
                all_predictions_list.extend(predicted_labels.cpu().numpy())
                correct_predictions += (predicted_labels == true_labels_int).sum().item()

                if criterion_obj is not None:
                    loss_value = 0.0
                    if criterion_type == "ce":
                        loss = criterion_obj(output_logits, true_labels_int)
                        loss_value = loss.item()
                    elif criterion_type == "gcod":
                        if graph_embeddings is None:
                            print(f"ERROR: graph_embeddings is None for GCOD validation at batch {batch_idx}. This should not happen.")
                            # Gestisci l'errore, es. saltando la loss o sollevando eccezione
                            # Per ora saltiamo l'aggiunta alla loss totale per questo batch.
                            # Oppure: raise ValueError("graph_embeddings cannot be None for GCOD")
                            loss_value = float('nan') # per indicare un problema
                        elif not hasattr(data_batch, 'original_idx'):
                            raise ValueError("Per GCOD in valutazione, 'original_idx' deve essere presente nel batch.")
                        elif num_classes_dataset is None:
                            raise ValueError("Per GCOD in valutazione, 'num_classes_dataset' deve essere fornito.")
                        else:
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
                            loss_final = l1 + lambda_l3_weight * l3
                            loss_value = loss_final.item()
                    else:
                        raise ValueError(f"Unsupported criterion_type: {criterion_type} in evaluation.")

                    if not (np.isnan(loss_value) or np.isinf(loss_value)):
                        total_loss += loss_value * data_batch.num_graphs
                    else:
                        print(f"WARNING: NaN/Inf loss calculated or graph_embeddings missing for GCOD for batch {batch_idx}. Skipping addition to total_loss.")
            else:
                all_predictions_list.extend(predicted_labels.cpu().numpy())

            total_samples += data_batch.num_graphs

    if is_validation:
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        f1 = 0.0
        if all_labels and all_predictions_list and len(all_labels) == len(all_predictions_list):
            f1 = f1_score(all_labels, all_predictions_list, average='macro', zero_division=0)
        elif not all_labels and not all_predictions_list : # se il dataset di val fosse vuoto
            pass # f1 rimane 0.0, acc 0.0, loss 0.0 - caso limite
        else: # lunghezze non corrispondenti o uno vuoto e l'altro no, indica un problema
            print(f"WARNING: F1 score calculation issue. len(all_labels)={len(all_labels)}, len(all_predictions_list)={len(all_predictions_list)}.")
            # Potresti decidere di non calcolare F1 o gestirlo diversamente

        return avg_loss, accuracy, f1
    else:
        return all_predictions_list