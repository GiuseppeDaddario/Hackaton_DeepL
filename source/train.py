# train.py (solo la parte rilevante)
import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm
import torch_xla.core.xla_model as xm

# Assumiamo che eps sia definito globalmente o importato se necessario per il clamp di u
eps = 1e-7 # Stesso eps usato nella classe gcodLoss

def train(
        atrain_global_value,
        train_loader,
        model,
        optimizer_model,
        device,
        optimizer_loss_params,
        loss_function_obj,
        save_checkpoints,
        checkpoint_path,
        current_epoch,
        criterion_type,
        num_classes_dataset,
        lambda_l3_weight,
        epoch_boost,
        loss_fn_ce,
        is_tpu=True  # Parametro aggiunto per TPU
):
    model.train()
    if hasattr(loss_function_obj, 'train'):
        loss_function_obj.train()

    running_epoch_loss_display = 0.0
    correct_samples_in_epoch = 0
    total_samples_in_epoch = 0

    for batch_idx, data_batch in enumerate(tqdm(train_loader, desc=f"Epoch {current_epoch+1} Training", unit="batch", leave=False)):
        num_graphs = data_batch.pop('_num_graphs', None)
        graphs_in_batch = Batch(**data_batch)
        if num_graphs is not None: # Restore num_graphs if needed, though Batch might recalc it
            graphs_in_batch.num_graphs = num_graphs
        true_labels_int = graphs_in_batch.y
        batch_original_indices = graphs_in_batch.original_idx.view(-1).tolist()
        target_one_hot = torch.zeros(
            len(true_labels_int),
            num_classes_dataset,
            device=device
        ).scatter_(1, true_labels_int.view(-1, 1).long(), 1)

        output_logits, graph_level_embeddings, _ = model(graphs_in_batch)

        if criterion_type == "ce":
            optimizer_model.zero_grad()
            loss_ce = loss_function_obj(output_logits, true_labels_int)
            loss_ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Usa xm.optimizer_step per TPU
            if is_tpu:

                xm.optimizer_step(optimizer_model)
                xm.mark_step()
            else:
                optimizer_model.step()

            current_batch_loss_for_display = loss_ce.item()

        elif criterion_type == "gcod":
            if current_epoch < epoch_boost:
                optimizer_model.zero_grad()
                loss_ce = loss_fn_ce(output_logits, true_labels_int)
                loss_ce.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Usa xm.optimizer_step per TPU
                if is_tpu:
                    xm.optimizer_step(optimizer_model)
                    xm.mark_step()
                else:
                    optimizer_model.step()

                current_batch_loss_for_display = loss_ce.item()
            else:
                l1_val, l2_val, l3_val = loss_function_obj.calculate_loss_components(
                    batch_original_indices,
                    output_logits,
                    target_one_hot,
                    graph_level_embeddings,
                    batch_idx,
                    current_epoch,
                    atrain_global_value
                )

                optimizer_model.zero_grad()
                loss_for_model_theta = l1_val + lambda_l3_weight * l3_val
                if torch.isnan(loss_for_model_theta).any() or torch.isinf(loss_for_model_theta).any():
                    print(f"Epoch {current_epoch+1}, Batch {batch_idx}: NaN/Inf in loss_for_model_theta ({loss_for_model_theta.item()}). Skipping model update.")
                else:
                    loss_for_model_theta.backward(retain_graph=True if optimizer_loss_params is not None else False)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Usa xm.optimizer_step per TPU
                    if is_tpu:
                        xm.optimizer_step(optimizer_model)
                    else:
                        optimizer_model.step()

                if optimizer_loss_params is not None:
                    optimizer_loss_params.zero_grad()
                    if torch.isnan(l2_val).any() or torch.isinf(l2_val).any():
                        print(f"Epoch {current_epoch+1}, Batch {batch_idx}: NaN/Inf in l2_val ({l2_val.item()}). Skipping u update.")
                    else:
                        l2_val.backward()

                    torch.nn.utils.clip_grad_norm_(loss_function_obj.parameters(), max_norm=1.0)

                    # Usa xm.optimizer_step per TPU
                    if is_tpu:
                        xm.optimizer_step(optimizer_loss_params)
                    else:
                        optimizer_loss_params.step()

                    # Clipping esplicito di u dopo l'aggiornamento
                    with torch.no_grad():
                        eps = 1e-7  # Definizione di eps
                        loss_function_obj.u.data.clamp_(min=eps, max=1.0 - eps)

                # Marca il passaggio per TPU
                if is_tpu:
                    xm.mark_step()

                current_batch_loss_for_display = (l1_val + l2_val + lambda_l3_weight * l3_val).item()
                if np.isnan(current_batch_loss_for_display) or np.isinf(current_batch_loss_for_display):
                    print(f"Epoch {current_epoch+1}, Batch {batch_idx}: Combined loss is NaN/Inf. Logging as 0 for safety.")
                    current_batch_loss_for_display = 0
        else:
            raise ValueError(f"Tipo di criterion '{criterion_type}' non supportato.")

        running_epoch_loss_display += current_batch_loss_for_display

        # Statistiche
        with torch.no_grad():
            _, predicted_labels = torch.max(output_logits.data, 1)
            total_samples_in_epoch += true_labels_int.size(0)
            correct_samples_in_epoch += (predicted_labels == true_labels_int.squeeze()).sum().item()
            # Aggiungi mark_step per TPU dopo operazioni statistiche
            if is_tpu:
                xm.mark_step()

    avg_epoch_loss = running_epoch_loss_display / len(train_loader) if len(train_loader) > 0 else 0.0
    avg_batch_accuracy_epoch = (correct_samples_in_epoch / total_samples_in_epoch) * 100 if total_samples_in_epoch > 0 else 0.0

    if save_checkpoints:
        final_checkpoint_path = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        # Usa xm.save per TPU
        if is_tpu:
            xm.save(model.state_dict(), final_checkpoint_path)
        else:
            torch.save(model.state_dict(), final_checkpoint_path)
        print(f"Checkpoint saved at {final_checkpoint_path}")

    return avg_batch_accuracy_epoch, avg_epoch_loss