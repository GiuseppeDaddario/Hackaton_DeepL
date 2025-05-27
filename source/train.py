import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

eps_train = 1e-7


def train_epoch(
        model,
        loader,
        optimizer_model,
        device,
        criterion_obj, # Oggetto loss (gcodLoss, LabelSmoothingCrossEntropy, o CE)
        criterion_type, # "ce" o "gcod"
        optimizer_loss_params, # Ottimizzatore per i parametri di gcodLoss (u), None per "ce"
        num_classes_dataset, # Necessario per gcodLoss e one-hot
        lambda_l3_weight, # Necessario per gcodLoss
        current_epoch,
        atrain_global_value, # Necessario per gcodLoss
        epoch_boost=0, # Per la fase di boosting con CE se usi GCOD
        gradient_clipping_norm=1.0
):

    model.train() # Imposta il modello in modalità training
    if criterion_type == "gcod" and hasattr(criterion_obj, 'train'):
        criterion_obj.train() # Se gcodLoss ha una modalità train (es. per aggiornamento centroidi)

    running_epoch_loss_display = 0.0
    correct_samples_in_epoch = 0
    total_samples_in_epoch = 0

    effective_criterion_type = criterion_type
    if criterion_type == "gcod" and current_epoch < epoch_boost:
        effective_criterion_type = "ce_boost" # Usa CE standard per il boosting


    for batch_idx, data_batch in enumerate(tqdm(loader, desc=f"Epoch {current_epoch+1} Training", unit="batch", leave=False)):
        data_batch = data_batch.to(device)
        true_labels_int = data_batch.y.to(device) # (N,)

        # --- Forward pass del modello ---
        output_logits, graph_embeddings, _ = model(data_batch) # (N, C), (N, emb_dim)

        # --- Calcolo Loss e Backward pass ---
        current_batch_loss_for_display = 0.0

        if effective_criterion_type == "ce" or effective_criterion_type == "ce_boost":
            optimizer_model.zero_grad()
            if effective_criterion_type == "ce_boost":
                loss_ce_boost = F.cross_entropy(output_logits, true_labels_int) # Semplice CE per boost
                loss_val = loss_ce_boost
            else: # "ce" normale
                loss_val = criterion_obj(output_logits, true_labels_int)

            loss_val.backward()
            if gradient_clipping_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_norm)
            optimizer_model.step()
            current_batch_loss_for_display = loss_val.item()

        elif criterion_type == "gcod":
            if not hasattr(data_batch, 'original_idx'):
                raise ValueError("Per GCOD, 'original_idx' deve essere presente nel batch.")
            batch_original_indices = data_batch.original_idx.view(-1).tolist()
            true_labels_one_hot = F.one_hot(true_labels_int, num_classes=num_classes_dataset).float()

            l1_val, l2_val, l3_val = criterion_obj.calculate_loss_components(
                batch_original_indices=batch_original_indices,
                gnn_logits_batch=output_logits,
                true_labels_batch_one_hot=true_labels_one_hot,
                gnn_embeddings_batch=graph_embeddings,
                batch_iter_num=batch_idx,
                current_epoch=current_epoch,
                atrain_overall_accuracy=atrain_global_value
            )

            # --- Aggiornamento per i parametri del modello θ (Eq. 10 nel paper GCOD: ∇θ(L1 + L3)) ---
            optimizer_model.zero_grad()
            loss_for_model_theta = l1_val + lambda_l3_weight * l3_val

            if torch.isnan(loss_for_model_theta).any() or torch.isinf(loss_for_model_theta).any():
                print(f"WARN - Epoch {current_epoch+1}, Batch {batch_idx}: NaN/Inf in loss_for_model_theta ({loss_for_model_theta.item()}). Skipping model update.")
            else:
                loss_for_model_theta.backward(retain_graph=True if optimizer_loss_params is not None else False)
                if gradient_clipping_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_norm)
                optimizer_model.step()

            # --- Aggiornamento per i parametri u ---
            if optimizer_loss_params is not None:
                optimizer_loss_params.zero_grad()
                if torch.isnan(l2_val).any() or torch.isinf(l2_val).any():
                    print(f"WARN - Epoch {current_epoch+1}, Batch {batch_idx}: NaN/Inf in l2_val ({l2_val.item()}). Skipping u update.")
                else:
                    l2_val.backward()
                    if gradient_clipping_norm > 0:
                        torch.nn.utils.clip_grad_norm_(criterion_obj.parameters(), max_norm=gradient_clipping_norm)
                    optimizer_loss_params.step()

                    with torch.no_grad():
                        criterion_obj.u.data.clamp_(min=eps_train, max=1.0 - eps_train)

            current_batch_loss_for_display = (l1_val + l2_val + lambda_l3_weight * l3_val).item()
            if np.isnan(current_batch_loss_for_display) or np.isinf(current_batch_loss_for_display):
                print(f"WARN - Epoch {current_epoch+1}, Batch {batch_idx}: Combined GCOD loss is NaN/Inf. Logging as 0.")
                current_batch_loss_for_display = 0.0
        else:
            raise ValueError(f"Tipo di criterion '{criterion_type}' non supportato durante il training.")

        running_epoch_loss_display += current_batch_loss_for_display * data_batch.num_graphs

        # Calcolo accuracy del batch (opzionale, ma utile per monitoring)
        with torch.no_grad():
            _, predicted_labels = torch.max(output_logits, 1)
            total_samples_in_epoch += data_batch.num_graphs
            correct_samples_in_epoch += (predicted_labels == true_labels_int).sum().item()

    avg_epoch_loss = running_epoch_loss_display / total_samples_in_epoch if total_samples_in_epoch > 0 else 0.0
    avg_epoch_accuracy = (correct_samples_in_epoch / total_samples_in_epoch) * 100 if total_samples_in_epoch > 0 else 0.0

    return avg_epoch_loss, avg_epoch_accuracy