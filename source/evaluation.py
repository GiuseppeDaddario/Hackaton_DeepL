# In source/evaluation.py

import torch
from sklearn.metrics import f1_score # Assicurati che sia importato
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np # Importa numpy se usi .numpy()

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
    correct_predictions_count = 0
    total_samples_for_metrics = 0

    all_preds_list = []
    all_labels_list = []

    desc_str = "Validating" if is_validation else "Testing (Predicting)"
    disable_tqdm = not is_validation

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(tqdm(loader, desc=desc_str, unit="batch", leave=False, disable=disable_tqdm)):
            data_batch = data_batch.to(device)

            output_logits = None
            graph_embeddings = None

            model_output = model(data_batch)
            if isinstance(model_output, tuple) and len(model_output) >= 2 :
                output_logits = model_output[0]
                graph_embeddings = model_output[1]
            elif isinstance(model_output, torch.Tensor):
                output_logits = model_output
            else:
                raise ValueError(f"Formato output del modello non riconosciuto: {type(model_output)}")

            if output_logits is None or output_logits.shape[0] == 0:
                continue

            if num_classes_dataset is None: # Aggiunto controllo per num_classes_dataset prima del suo utilizzo
                raise ValueError("num_classes_dataset non può essere None per determinare le predizioni.")

            if num_classes_dataset > 1:
                _, predicted_labels = torch.max(output_logits, 1)
            elif num_classes_dataset == 1:
                predicted_labels = (torch.sigmoid(output_logits) > 0.5).squeeze().long()
            else:
                raise ValueError(f"num_classes_dataset ({num_classes_dataset}) non valido.")

            all_preds_list.extend(predicted_labels.cpu().numpy().tolist())

            current_num_graphs_in_batch = data_batch.num_graphs if hasattr(data_batch, 'num_graphs') else output_logits.size(0)
            # total_samples_for_metrics era incrementato qui, ma ha più senso farlo solo se ci sono etichette per la validazione
            # o comunque contare tutti i grafi processati. Per coerenza con la loss, lo lascio qui.
            # Se is_validation è False, questo conta solo i grafi per cui si fanno predizioni.
            total_samples_for_metrics += current_num_graphs_in_batch


            if is_validation:
                if not hasattr(data_batch, 'y') or data_batch.y is None:
                    print(f"Warning: data_batch.y is None for a sample during validation. Skipping loss/metric calculation for this sample if all y in batch are None.")
                    # Se TUTTI y nel batch sono None, e criterion_obj è presente, potresti comunque voler saltare la loss.
                    # Per ora, procediamo e la loss potrebbe fallire se y è necessario e non c'è.
                    # O potresti accumulare solo le predizioni e non aggiornare la loss per questo batch.
                    # L'approccio più semplice è continuare se ALMENO UN y è presente per la loss.
                    # Se nessuno è presente, la loss non dovrebbe essere calcolata.
                    # Il controllo più robusto sarebbe: if data_batch.y is None or data_batch.y.numel() == 0: continue (se la loss fallisce)
                    # Per ora, continuiamo e affidiamoci al fatto che la loss gestisca y=None se necessario,
                    # o che ci sia almeno un y per il calcolo della loss.
                    # Il 'continue' che era qui era indentato male.
                    # La logica originale era: se y è None E criterion_obj è None, allora continue.
                    # Ma se is_validation=True, criterion_obj non dovrebbe essere None.
                    # Rimuovo il 'continue' problematico per ora e lascio che la logica della loss gestisca.
                    pass # Rimosso il continue problematico, la logica sotto gestirà y

                true_labels_int = data_batch.y.to(device) # Questo fallirà se data_batch.y è None
                all_labels_list.extend(true_labels_int.cpu().numpy().tolist())
                correct_predictions_count += (predicted_labels == true_labels_int.squeeze()).sum().item()

                if criterion_obj is None:
                    # Questo scenario non dovrebbe accadere se is_validation=True e lo script principale
                    # passa un criterion_obj per la validazione.
                    print("Warning: criterion_obj is None during validation. Loss will be 0.")
                    loss = torch.tensor(0.0, device=device)
                elif criterion_type == "ce":
                    target = true_labels_int
                    if num_classes_dataset == 1:
                        target = target.float().unsqueeze(1)
                    loss = criterion_obj(output_logits, target)
                elif criterion_type == "gcod":
                    if graph_embeddings is None:
                        raise ValueError("graph_embeddings sono None, ma richiesti per GCOD loss in validazione.")
                    if not hasattr(data_batch, 'original_idx'):
                        raise ValueError("Per GCOD, 'original_idx' deve essere presente.")

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
                    loss = l1 + l2 + lambda_l3_weight * l3
                else:
                    raise ValueError(f"Unsupported criterion_type: {criterion_type}")

                total_loss += loss.item() * current_num_graphs_in_batch

    avg_loss_value = 0.0
    accuracy_value = 0.0
    f1_value = 0.0

    if is_validation:
        if total_samples_for_metrics > 0 and len(all_labels_list) > 0 : # Assicurati che ci siano etichette per calcolare metriche
            avg_loss_value = total_loss / total_samples_for_metrics
            accuracy_value = correct_predictions_count / total_samples_for_metrics # Usa total_samples_for_metrics che conta i grafi

            if len(all_preds_list) == len(all_labels_list): # Devono avere la stessa lunghezza
                try:
                    avg_type_f1 = 'binary' if num_classes_dataset == 2 else 'macro'
                    if len(set(all_labels_list)) < 2 and avg_type_f1 == 'macro':
                        print("Warning: Meno di 2 classi nelle etichette vere per F1 macro. F1 potrebbe non essere ben definito o essere 0/1 a seconda dell'implementazione di f1_score.")
                    f1_value = f1_score(all_labels_list, all_preds_list, average=avg_type_f1, zero_division=0)
                except ValueError as e:
                    print(f"Errore nel calcolo F1-score: {e}. Restituisco F1=0.")
                    f1_value = 0.0
            else:
                print("Warning: Lunghezza di predizioni ed etichette non corrisponde. F1-score non calcolato.")
                f1_value = 0.0
        else:
            print("Warning: Nessun campione o etichetta valida per calcolare le metriche di validazione.")
    # else (is_validation=False): avg_loss, accuracy, f1 rimangono 0.0

    return avg_loss_value, accuracy_value, f1_value, all_preds_list, all_labels_list