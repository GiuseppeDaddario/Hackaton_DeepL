import torch
from tqdm import tqdm

import torch
from tqdm import tqdm

def train(
        atrain_global_value,
        train_loader,
        model,
        optimizer_model, # Rinominato per chiarezza
        device,
        optimizer_loss_params, # Rinominato per chiarezza (per u di NCOD/GCOD)
        loss_function_obj,    # Rinominato per chiarezza
        save_checkpoints,
        checkpoint_path,
        current_epoch,
        criterion_type,
        num_classes_dataset # Aggiunto per target_one_hot
):
    model.train()
    if hasattr(loss_function_obj, 'train'): # Alcune loss custom potrebbero avere modalità train/eval
        loss_function_obj.train()

    running_epoch_loss = 0.0
    correct_samples_in_epoch = 0
    total_samples_in_epoch = 0

    for batch_idx, data_batch in enumerate(tqdm(train_loader, desc=f"Epoch {current_epoch+1} Training", unit="batch", leave=False)):
        # 'data_batch' è l'oggetto Batch di PyG che contiene uno o più grafi
        # e dovrebbe avere l'attributo 'original_idx' aggiunto nel tuo GraphDataset.get()

        graphs_in_batch = data_batch.to(device)
        true_labels_int = graphs_in_batch.y.to(device) # Etichette intere (indici di classe)

        if not hasattr(graphs_in_batch, 'original_idx'):
            raise ValueError(
                "L'oggetto 'data_batch' (Batch di PyG) deve avere un attributo 'original_idx'. "
                "Assicurati di aggiungerlo nel metodo get() del tuo GraphDataset."
            )
        # Indici originali dei campioni nel batch, usati per NCOD/GCOD
        batch_original_indices = graphs_in_batch.original_idx.view(-1).tolist()

        # Preparazione target one-hot per loss NCOD/GCOD
        target_one_hot = torch.zeros(
            len(true_labels_int),
            num_classes_dataset, # Usa il numero di classi passato
            device=device
        ).scatter_(1, true_labels_int.view(-1, 1).long(), 1)

        # Azzera gradienti
        optimizer_model.zero_grad()
        if optimizer_loss_params is not None:
            optimizer_loss_params.zero_grad()

        # Forward pass del modello
        # Assumiamo che model(graphs_in_batch) restituisca (logits, graph_embeddings, node_embeddings)
        output_logits, graph_level_embeddings, _ = model(graphs_in_batch)

        # Calcolo della loss
        current_batch_loss = 0
        if criterion_type == "ce":
            current_batch_loss = loss_function_obj(output_logits, true_labels_int)
        elif criterion_type == "ncod" or criterion_type == "gcod":
            current_batch_loss = loss_function_obj(
                batch_original_indices,    # Indici originali
                output_logits,             # Logits dal modello
                target_one_hot,            # Etichette one-hot
                graph_level_embeddings,    # Embedding a livello di grafo
                batch_idx,                 # Numero del batch corrente nell'epoca
                current_epoch,             # Epoca corrente
                atrain_global_value        # Accuratezza globale di training (atrain)
            )
        else:
            raise ValueError(f"Tipo di criterion '{criterion_type}' non supportato.")

        # Backward pass e ottimizzazione
        current_batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prova valori come 1.0, 5.0, 10.0
        if optimizer_loss_params is not None:
            # È importante clippare anche i gradienti dei parametri della loss (come 'u')
            # Potresti aver bisogno di un max_norm diverso qui, a seconda della scala dei gradienti di 'u'
            torch.nn.utils.clip_grad_norm_(loss_function_obj.parameters(), max_norm=1.0) # Prova anche 0.1, 0.5, 5.0

        optimizer_model.step()
        if optimizer_loss_params is not None:
            optimizer_loss_params.step()
            if hasattr(loss_function_obj, 'u'): # Controlla se l'attributo 'u' esiste
                with torch.no_grad():
                    current_eps_for_u_clamping = 1e-7
                    loss_function_obj.u.data.clamp_(min=current_eps_for_u_clamping, max=1.0 - current_eps_for_u_clamping)

        # Statistiche per l'accuratezza media dei batch nell'epoca
        with torch.no_grad():
            _, predicted_labels = torch.max(output_logits.data, 1)
            total_samples_in_epoch += true_labels_int.size(0)
            correct_samples_in_epoch += (predicted_labels == true_labels_int.squeeze()).sum().item()

        running_epoch_loss += current_batch_loss.item()

    # Calcolo medie per l'epoca
    avg_epoch_loss = running_epoch_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    avg_batch_accuracy_epoch = (correct_samples_in_epoch / total_samples_in_epoch) * 100 if total_samples_in_epoch > 0 else 0.0

    # Salvataggio checkpoint (logica invariata)
    if save_checkpoints: # save_checkpoints qui è un booleano (es. (epoch + 1 in checkpoint_intervals))
        final_checkpoint_path = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth" # checkpoint_path è il prefisso
        torch.save(model.state_dict(), final_checkpoint_path)
        print(f"Checkpoint saved at {final_checkpoint_path}")

    return avg_batch_accuracy_epoch, avg_epoch_loss
