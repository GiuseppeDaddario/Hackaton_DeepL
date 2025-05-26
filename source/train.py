# train.py (solo la parte rilevante)
import numpy as np
import torch
from tqdm import tqdm

# Assumiamo che eps sia definito globalmente o importato se necessario per il clamp di u
eps = 1e-7 # Stesso eps usato nella classe gcodLoss

def train(
        atrain_global_value,
        train_loader,
        model,
        optimizer_model,
        device,
        optimizer_loss_params, # Ottimizzatore per i parametri di gcodLoss (u)
        loss_function_obj,    # Istanza di gcodLoss
        save_checkpoints,
        checkpoint_path,
        current_epoch,
        criterion_type,
        num_classes_dataset,
        lambda_l3_weight,
        epoch_boost
):
    model.train()
    if hasattr(loss_function_obj, 'train'):
        loss_function_obj.train() # Metti gcodLoss in modalità train se necessario (non sembra esserlo)

    running_epoch_loss_display = 0.0 # Per il logging della loss combinata
    correct_samples_in_epoch = 0
    total_samples_in_epoch = 0

    for batch_idx, data_batch in enumerate(tqdm(train_loader, desc=f"Epoch {current_epoch+1} Training", unit="batch", leave=False)):
        graphs_in_batch = data_batch.to(device)
        true_labels_int = graphs_in_batch.y.to(device)
        batch_original_indices = graphs_in_batch.original_idx.view(-1).tolist()
        target_one_hot = torch.zeros(
            len(true_labels_int),
            num_classes_dataset,
            device=device
        ).scatter_(1, true_labels_int.view(-1, 1).long(), 1)

        # --- Forward pass del modello ---
        # È importante che questo avvenga prima di azzerare i gradienti se i gradienti di u dipendono dal forward del modello
        # (ma in questo caso, u è un parametro separato, e i suoi gradienti verranno da L2).
        output_logits, graph_level_embeddings, _ = model(graphs_in_batch) # Output del modello θ

        # --- Calcolo delle componenti della Loss ---
        if criterion_type == "ce":
            # Azzera gradienti per CE
            optimizer_model.zero_grad()
            loss_ce = loss_function_obj(output_logits, true_labels_int)
            loss_ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_model.step()
            current_batch_loss_for_display = loss_ce.item()

        elif criterion_type == "gcod":
            if current_epoch <= epoch_boost:
                optimizer_model.zero_grad()
                loss_fn_ce = torch.nn.CrossEntropyLoss()
                loss_ce = loss_fn_ce(output_logits, true_labels_int)
                loss_ce.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_model.step()
                current_batch_loss_for_display = loss_ce.item()
            else:
                # Chiamata al metodo che restituisce L1, L2, L3
                l1_val, l2_val, l3_val = loss_function_obj.calculate_loss_components(
                    batch_original_indices,
                    output_logits,             # Logits da θ
                    target_one_hot,
                    graph_level_embeddings,    # Embedding da θ
                    batch_idx,
                    current_epoch,
                    atrain_global_value
                )

                # --- Aggiornamento per i parametri del modello θ (Eq. 10: ∇θ(L1 + L3)) ---
                optimizer_model.zero_grad()
                loss_for_model_theta = l1_val + lambda_l3_weight * l3_val # Applica il peso a L3 qui
                if torch.isnan(loss_for_model_theta).any() or torch.isinf(loss_for_model_theta).any():
                    print(f"Epoch {current_epoch+1}, Batch {batch_idx}: NaN/Inf in loss_for_model_theta ({loss_for_model_theta.item()}). Skipping model update.")
                    # Potresti voler saltare l'aggiornamento o gestire l'errore
                else:
                    loss_for_model_theta.backward(retain_graph=True if optimizer_loss_params is not None else False) # retain_graph se L2 usa parti del grafo
                    # che sono state usate anche da L1 o L3 e
                    # che NON sono state detached.
                    # In questo caso, output_logits è usato da L1 e L3.
                    # L2 usa output_logits.detach().
                    # u_batch è usato da L1 e L2.
                    # Se L1 ha .backward(retain_graph=True) allora i buffer intermedi
                    # per calcolare i gradienti di u rispetto a L1 sono mantenuti.
                    # Ma vogliamo solo ∇u L2, non ∇u L1.
                    # Quindi è meglio fare backward separati senza retain_graph se possibile.
                    # Per essere sicuri, e data la struttura, è meglio
                    # azzerare i gradienti di u prima di L2.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer_model.step()

                # --- Aggiornamento per i parametri u (Eq. 11: ∇u L2) ---
                if optimizer_loss_params is not None:
                    optimizer_loss_params.zero_grad() # Azzera i gradienti di u PRIMA di L2.backward()
                    # per assicurarsi che solo L2 contribuisca ai gradienti di u.
                    if torch.isnan(l2_val).any() or torch.isinf(l2_val).any():
                        print(f"Epoch {current_epoch+1}, Batch {batch_idx}: NaN/Inf in l2_val ({l2_val.item()}). Skipping u update.")
                    else:
                        l2_val.backward() # Calcola i gradienti di L2 rispetto a u (e altri parametri se non detached)
                    # Clip dei gradienti specifici per i parametri della loss 'u'
                    torch.nn.utils.clip_grad_norm_(loss_function_obj.parameters(), max_norm=1.0) # Adatta max_norm se necessario
                    optimizer_loss_params.step()

                    # Clipping esplicito di u dopo l'aggiornamento per mantenerlo in [eps, 1-eps]
                    with torch.no_grad():
                        loss_function_obj.u.data.clamp_(min=eps, max=1.0 - eps)

                # Per il logging, puoi sommare le loss o usare una metrica specifica
                current_batch_loss_for_display = (l1_val + l2_val + lambda_l3_weight * l3_val).item()
                if np.isnan(current_batch_loss_for_display) or np.isinf(current_batch_loss_for_display):
                    print(f"Epoch {current_epoch+1}, Batch {batch_idx}: Combined loss is NaN/Inf. Logging as 0 for safety.")
                    current_batch_loss_for_display = 0 # Evita errori nel logging
        else:
            raise ValueError(f"Tipo di criterion '{criterion_type}' non supportato.")

        running_epoch_loss_display += current_batch_loss_for_display

        # Statistiche (invariate)
        with torch.no_grad():
            _, predicted_labels = torch.max(output_logits.data, 1)
            total_samples_in_epoch += true_labels_int.size(0)
            correct_samples_in_epoch += (predicted_labels == true_labels_int.squeeze()).sum().item()

    avg_epoch_loss = running_epoch_loss_display / len(train_loader) if len(train_loader) > 0 else 0.0
    avg_batch_accuracy_epoch = (correct_samples_in_epoch / total_samples_in_epoch) * 100 if total_samples_in_epoch > 0 else 0.0

    if save_checkpoints:
        final_checkpoint_path = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), final_checkpoint_path)
        print(f"Checkpoint saved at {final_checkpoint_path}")

    return avg_batch_accuracy_epoch, avg_epoch_loss