import torch
from tqdm import tqdm
import torch.nn.functional as F

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
    all_predictions_list = [] # Per il test set finale

    # Per la barra di progresso, mostrala sempre a meno che specificamente disabilitata
    # Se is_validation è False, probabilmente è test, e vuoi vedere il progresso.
    # Potresti voler passare un flag esplicito per `disable_tqdm` se necessario.
    # Per ora, usiamo la logica originale: disable=not is_validation
    desc_str = "Validating" if is_validation else "Testing (Predicting)"

    with torch.no_grad(): # Disabilita il calcolo dei gradienti
        for batch_idx, data_batch in enumerate(tqdm(loader, desc=desc_str, unit="batch", leave=False, disable=not is_validation)):
            data_batch = data_batch.to(device)

            # --- Calcolo comune a validazione e test ---
            output_logits, graph_embeddings, _ = model(data_batch) # (N, C), (N, emb_dim)
            _, predicted_labels = torch.max(output_logits, 1) # (N,)

            # --- Logica specifica per la VALIDAZIONE (quando hai le etichette vere) ---
            if is_validation:
                if not hasattr(data_batch, 'y'):
                    raise ValueError("Per la validazione (is_validation=True), 'data_batch.y' (etichette vere) deve essere presente.")
                true_labels_int = data_batch.y.to(device) # (N,)

                if criterion_obj is None:
                    raise ValueError("Per la validazione (is_validation=True), 'criterion_obj' deve essere fornito.")

                # Calcolo della Loss
                if criterion_type == "ce":
                    # criterion_obj qui è LabelSmoothingCrossEntropy o nn.CrossEntropyLoss
                    loss = criterion_obj(output_logits, true_labels_int)
                elif criterion_type == "gcod":
                    # gcodLoss richiede più input
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
                correct_predictions += (predicted_labels == true_labels_int).sum().item()
            # --- Fine logica di validazione ---
            else:
                # --- Logica specifica per il TEST (solo predizioni) ---
                all_predictions_list.extend(predicted_labels.cpu().numpy())

            total_samples += data_batch.num_graphs

    if is_validation:
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        return avg_loss, accuracy
    else:
        # Se non è validazione, si assume sia test e si restituiscono le predizioni
        return all_predictions_list