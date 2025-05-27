# source/train.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def train(atrain_global_value, train_loader, model, optimizer_model, device,
          optimizer_u_or_loss_params, # Per GCOD è optimizer_u, per NCOD optimizer per i parametri della loss
          loss_function_obj, save_checkpoints, checkpoint_path, current_epoch,
          criterion_type, num_classes_dataset,
          lambda_l3_weight, # Usato per GCOD per pesare L3
          epoch_boost,
          gcod_eps=1e-7): # Epsilon per il clamping di u in GCOD

    model.train()
    # Se la loss custom ha una modalità train (es. per dropout interno), attivala
    if hasattr(loss_function_obj, 'train') and callable(getattr(loss_function_obj, 'train')):
        loss_function_obj.train()

    total_epoch_loss_sum = 0  # Somma delle loss dei batch (per la media finale)
    total_samples_epoch = 0
    batch_accuracies = []

    # Determina se siamo nella fase di boosting con CrossEntropy
    use_ce_for_boosting = current_epoch < epoch_boost

    if use_ce_for_boosting:
        print(f"Epoch {current_epoch + 1}: Using CrossEntropy loss (boosting phase).")

    progress_bar = tqdm(train_loader, desc=f"Epoch {current_epoch+1} Training", unit="batch", leave=False)

    for batch_idx, data_batch in enumerate(progress_bar):
        graphs = data_batch.to(device)
        true_labels_int = graphs.y.to(device).squeeze() # Assicurati che sia 1D (batch_size,)
        if true_labels_int.ndim == 0: # Se è uno scalare (batch_size=1)
            true_labels_int = true_labels_int.unsqueeze(0)

        # È FONDAMENTALE avere gli indici originali per GCOD/NCOD
        # Assumo che il DataLoader fornisca 'original_index' nel batch.
        # Se il nome del campo è diverso, adattalo qui.
        if not hasattr(data_batch, 'original_index'):
            if criterion_type in ["gcod", "ncod"] and not use_ce_for_boosting:
                raise AttributeError("Batch data must have 'original_index' attribute for GCOD/NCOD. "
                                     "Modifica GraphDataset per includerlo in ogni Data object.")
            batch_original_indices = None # Non necessario per CE
        else:
            batch_original_indices = data_batch.original_index.to(device)


        # One-hot encode true_labels (necessario per gcodLoss e forse ncodLoss)
        if criterion_type in ["gcod", "ncod"] or use_ce_for_boosting == False: # Solo se non usiamo CE semplice
            true_labels_one_hot = F.one_hot(true_labels_int, num_classes=num_classes_dataset).float().to(device)


        # Forward pass del modello
        # Assumo che il modello restituisca (logits, embeddings, [altri_output_opzionali])
        gnn_logits_batch, gnn_embeddings_batch, *_ = model(graphs)

        current_batch_loss = None

        # --- Calcolo della Loss e aggiornamento dei pesi ---
        if use_ce_for_boosting or criterion_type == "ce":
            loss = F.cross_entropy(gnn_logits_batch, true_labels_int)

            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()
            current_batch_loss = loss.item()

        elif criterion_type == "gcod":
            if batch_original_indices is None: # Doppio check per sicurezza
                raise ValueError("batch_original_indices is None, but required for GCOD.")

            l1, l2, l3 = loss_function_obj(
                batch_original_indices=batch_original_indices,
                gnn_logits_batch=gnn_logits_batch,
                true_labels_batch_one_hot=true_labels_one_hot,
                gnn_embeddings_batch=gnn_embeddings_batch,
                batch_iter_num=batch_idx,
                current_epoch=current_epoch,
                atrain_overall_accuracy=atrain_global_value
            )

            # 1. Aggiornamento di theta (parametri del modello GNN)
            optimizer_model.zero_grad()
            loss_for_theta = l1 + lambda_l3_weight * l3 # Usa il peso per L3
            loss_for_theta.backward() # Calcola ∇_θ (L1 + lambda*L3)
            optimizer_model.step()

            # 2. Aggiornamento di u (parametro della gcodLoss)
            if optimizer_u_or_loss_params is not None:
                optimizer_u_or_loss_params.zero_grad()
                loss_for_u = l2
                loss_for_u.backward() # Calcola ∇_u (L2)
                optimizer_u_or_loss_params.step()

                # 3. Clamping di u dopo l'aggiornamento
                with torch.no_grad():
                    loss_function_obj.u.data.clamp_(min=gcod_eps, max=1.0 - gcod_eps)

            current_batch_loss = loss_for_theta.item() # Loss usata per aggiornare il modello

        elif criterion_type == "ncod":
            # --------------- !!! IMPLEMENTARE LOGICA NCOD QUI !!! -----------------
            # Questa è una sezione placeholder. Dovrai adattarla alla tua ncodLoss.
            # Potrebbe richiedere:
            # 1. Chiamare loss_function_obj(...) con gli argomenti corretti.
            # 2. Calcolare una o più loss (es. loss_theta, loss_params_loss).
            # 3. Fare backward e step per optimizer_model.
            # 4. Fare backward e step per optimizer_u_or_loss_params (se ncodLoss ha parametri apprendibili).
            # 5. Eventuale clamping dei parametri della loss.
            print(f"WARN: NCOD training logic not fully implemented in train function for epoch {current_epoch+1}, batch {batch_idx+1}.")
            print("Falling back to CrossEntropy loss for this batch.")
            # Esempio di fallback a CE se NCOD non è implementato:
            loss = F.cross_entropy(gnn_logits_batch, true_labels_int)
            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()
            current_batch_loss = loss.item()
            # --------------- !!! FINE PLACEHOLDER NCOD !!! -----------------
        else:
            raise ValueError(f"Unsupported criterion_type: {criterion_type} in train function")

        # Statistiche del batch
        num_samples_in_batch = true_labels_int.size(0)
        total_epoch_loss_sum += current_batch_loss * num_samples_in_batch # Pondera per la dim del batch
        total_samples_epoch += num_samples_in_batch

        _, predicted_labels = torch.max(gnn_logits_batch, 1)
        correct_batch = (predicted_labels == true_labels_int).sum().item()
        batch_accuracy = (correct_batch / num_samples_in_batch) * 100 if num_samples_in_batch > 0 else 0
        batch_accuracies.append(batch_accuracy)

        progress_bar.set_postfix({
            'Loss': f'{current_batch_loss:.4f}', # Loss del batch corrente
            'Batch Acc': f'{batch_accuracy:.2f}%',
            'Avg Epoch Acc': f'{np.mean(batch_accuracies):.2f}%' # Media delle acc dei batch finora nell'epoca
        })

    avg_epoch_loss = total_epoch_loss_sum / total_samples_epoch if total_samples_epoch > 0 else 0
    avg_epoch_batch_accuracy = np.mean(batch_accuracies) if batch_accuracies else 0 # Media delle acc dei batch nell'intera epoca

    if save_checkpoints:
        checkpoint_file_name = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file_name)
        print(f"Checkpoint saved to {checkpoint_file_name}")

    return avg_epoch_batch_accuracy, avg_epoch_loss