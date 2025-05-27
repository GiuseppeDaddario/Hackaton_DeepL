import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm
import torch_xla.core.xla_model as xm

def train(
        atrain_global_value,
        train_loader,
        model,
        optimizer_model,
        device,
        optimizer_loss_params,
        loss_function_obj,
        save_checkpoints, # Booleano, già considera is_master
        checkpoint_path, # Path base per checkpoint
        current_epoch,
        criterion_type,
        num_classes_dataset,
        lambda_l3_weight,
        epoch_boost,
        loss_fn_ce, # loss_fn_ce per il boosting, già su device
        is_tpu=True
):
    model.train()
    if hasattr(loss_function_obj, 'train') and callable(getattr(loss_function_obj, 'train')):
        loss_function_obj.train()

    running_epoch_loss_display = 0.0
    correct_samples_in_epoch = 0
    total_samples_in_epoch = 0

    eps_clamp_u = 1e-7

    data_iterator = tqdm(train_loader, desc=f"Epoch {current_epoch+1} Training (Rank {xm.get_ordinal()})", unit="batch", leave=False, disable=not xm.is_master_ordinal())

    for batch_idx, data_dict in enumerate(data_iterator):
        if not data_dict: continue

        num_graphs_val = data_dict.pop('_num_graphs', None)
        graphs_in_batch = Batch(**data_dict)
        if num_graphs_val is not None:
            graphs_in_batch.num_graphs = num_graphs_val

        true_labels_int = graphs_in_batch.y

        # original_idx dovrebbe essere un tensore [N] o [N,1]
        batch_original_indices_tensor = graphs_in_batch.original_idx
        if batch_original_indices_tensor.dim() > 1:
            batch_original_indices_tensor = batch_original_indices_tensor.view(-1)
        batch_original_indices_list = batch_original_indices_tensor.tolist()


        target_one_hot = torch.zeros(
            graphs_in_batch.num_graphs, # Usa num_graphs dal batch
            num_classes_dataset,
            device=device
        ).scatter_(1, true_labels_int.view(-1, 1).long(), 1)

        output_logits, graph_level_embeddings, _ = model(graphs_in_batch, train=True)

        current_batch_loss_for_display = 0.0

        if criterion_type == "ce":
            optimizer_model.zero_grad()
            loss_val = loss_function_obj(output_logits, true_labels_int) # loss_function_obj è già CE
            loss_val.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Opzionale
            if is_tpu:
                xm.optimizer_step(optimizer_model)
            else:
                optimizer_model.step()
            current_batch_loss_for_display = loss_val.item()

        elif criterion_type in ["gcod", "ncod"]: # Gestione unificata se la logica è simile
            if current_epoch < epoch_boost:
                optimizer_model.zero_grad()
                loss_boost = loss_fn_ce(output_logits, true_labels_int) # Usa loss_fn_ce per il boost
                loss_boost.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Opzionale
                if is_tpu:
                    xm.optimizer_step(optimizer_model)
                else:
                    optimizer_model.step()
                current_batch_loss_for_display = loss_boost.item()
            else: # Logica per NCOD/GCOD
                if criterion_type == "gcod":
                    l1_val, l2_val, l3_val = loss_function_obj.calculate_loss_components(
                        batch_original_indices_list, output_logits, target_one_hot,
                        graph_level_embeddings, batch_idx, current_epoch, atrain_global_value
                    )
                    loss_for_model_theta = l1_val + lambda_l3_weight * l3_val
                    loss_for_u_params = l2_val
                elif criterion_type == "ncod":
                    # Assumendo che ncodLoss restituisca (loss_theta, loss_u) o simili
                    # Oppure che calcoli e applichi grad internamente.
                    # Questo è un ESEMPIO, adatta alla tua ncodLoss API
                    loss_for_model_theta, loss_for_u_params = loss_function_obj(
                        output_logits, target_one_hot, batch_original_indices_list,
                        graph_level_embeddings, current_epoch # , atrain_global_value se serve
                    )


                # Aggiornamento parametri modello (theta)
                optimizer_model.zero_grad()
                if torch.isnan(loss_for_model_theta).any() or torch.isinf(loss_for_model_theta).any():
                    if xm.is_master_ordinal(): print(f"Epoch {current_epoch+1}, Batch {batch_idx}: NaN/Inf in loss_for_model_theta. Skipping model update.")
                else:
                    loss_for_model_theta.backward(retain_graph=True if optimizer_loss_params is not None else False)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Opzionale
                    if is_tpu:
                        xm.optimizer_step(optimizer_model)
                    else:
                        optimizer_model.step()

                # Aggiornamento parametri della loss (u)
                if optimizer_loss_params is not None: # Solo se ci sono parametri 'u' da ottimizzare
                    optimizer_loss_params.zero_grad()
                    if torch.isnan(loss_for_u_params).any() or torch.isinf(loss_for_u_params).any():
                        if xm.is_master_ordinal(): print(f"Epoch {current_epoch+1}, Batch {batch_idx}: NaN/Inf in loss_for_u_params. Skipping u update.")
                    else:
                        # Se loss_for_model_theta.backward() ha già calcolato i grad per u (se u è parte dello stesso grafo)
                        # allora la retain_graph=True era importante. Altrimenti, se sono grafi separati:
                        loss_for_u_params.backward()
                        # torch.nn.utils.clip_grad_norm_(loss_function_obj.parameters(), max_norm=1.0) # Opzionale
                        if is_tpu:
                            xm.optimizer_step(optimizer_loss_params)
                        else:
                            optimizer_loss_params.step()

                        with torch.no_grad(): # Clamp di u
                            if hasattr(loss_function_obj, 'u'): # Assicurati che 'u' esista
                                loss_function_obj.u.data.clamp_(min=eps_clamp_u, max=1.0 - eps_clamp_u)

                if criterion_type == "gcod":
                    current_batch_loss_for_display = (l1_val + l2_val + lambda_l3_weight * l3_val).item()
                elif criterion_type == "ncod":
                    current_batch_loss_for_display = (loss_for_model_theta + loss_for_u_params).item() # Esempio

                if np.isnan(current_batch_loss_for_display) or np.isinf(current_batch_loss_for_display):
                    if xm.is_master_ordinal(): print(f"Epoch {current_epoch+1}, Batch {batch_idx}: Combined loss is NaN/Inf. Logging as 0.")
                    current_batch_loss_for_display = 0.0
        else:
            raise ValueError(f"Tipo di criterion '{criterion_type}' non supportato.")

        running_epoch_loss_display += current_batch_loss_for_display

        with torch.no_grad():
            _, predicted_labels = torch.max(output_logits.data, 1)
            total_samples_in_epoch += true_labels_int.size(0)
            correct_samples_in_epoch += (predicted_labels == true_labels_int.squeeze()).sum().item()

        if is_tpu:
            xm.mark_step()

    # Calcolo medie (solo il master dovrebbe loggarle, ma i valori devono essere aggregati se nprocs > 1)
    # Con nprocs=1, questo è corretto. Per nprocs > 1, serve `xm.all_reduce`.
    # Per ora, ogni processo calcola le sue medie. Il log del master sarà quello di riferimento.
    num_batches = len(train_loader) if len(train_loader) > 0 else 1 # Evita divisione per zero se vuoto

    # Per distributed training, queste somme dovrebbero essere aggregate
    total_loss_sum = xm.all_reduce(xm.REDUCE_SUM, running_epoch_loss_display)
    total_correct_sum = xm.all_reduce(xm.REDUCE_SUM, correct_samples_in_epoch)
    total_samples_sum = xm.all_reduce(xm.REDUCE_SUM, total_samples_in_epoch)
    avg_epoch_loss = total_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_batch_accuracy_epoch = (total_correct_sum / total_samples_sum) * 100 if total_samples_sum > 0 else 0.0

    if save_checkpoints: # Questo è True solo per il master
        final_checkpoint_path = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        xm.save(model.state_dict(), final_checkpoint_path)
        xm.master_print(f"Checkpoint saved by master: {final_checkpoint_path}")

    return avg_batch_accuracy_epoch, avg_epoch_loss