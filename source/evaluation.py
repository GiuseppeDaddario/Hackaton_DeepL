# In source/evaluation.py

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

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
    total_loss_accumulator = 0.0
    correct_predictions_count = 0
    total_samples_processed = 0 # Conta i grafi/campioni effettivamente usati per le metriche

    all_preds_list = []
    all_labels_list = [] # Sarà popolata solo se is_validation=True e ci sono etichette

    desc_str = "Validating" if is_validation else "Testing (Predicting)"
    disable_tqdm = False # Lascia tqdm attivo per vedere il progresso

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(tqdm(loader, desc=desc_str, unit="batch", leave=False, disable=disable_tqdm)):
            data_batch = data_batch.to(device)

            output_logits = None
            graph_embeddings = None

            model_output = model(data_batch)
            if isinstance(model_output, tuple) and len(model_output) >= 1:
                output_logits = model_output[0]
                if len(model_output) >= 2:
                    graph_embeddings = model_output[1]
            elif isinstance(model_output, torch.Tensor):
                output_logits = model_output
            else:
                raise ValueError(f"Formato output del modello non riconosciuto: {type(model_output)}")

            if output_logits is None or output_logits.shape[0] == 0:
                continue

            if num_classes_dataset is None:
                raise ValueError("num_classes_dataset non può essere None.")

            if num_classes_dataset > 1:
                _, predicted_labels = torch.max(output_logits, 1)
            elif num_classes_dataset == 1:
                predicted_labels = (torch.sigmoid(output_logits) > 0.5).squeeze().long()
            else:
                raise ValueError(f"num_classes_dataset ({num_classes_dataset}) non valido.")

            all_preds_list.extend(predicted_labels.cpu().numpy().tolist())

            # Conta i grafi/campioni nel batch corrente
            current_num_graphs_in_batch = data_batch.num_graphs if hasattr(data_batch, 'num_graphs') else output_logits.size(0)

            if is_validation:
                if hasattr(data_batch, 'y') and data_batch.y is not None:
                    true_labels_int = data_batch.y.to(device)
                    all_labels_list.extend(true_labels_int.cpu().numpy().tolist())
                    correct_predictions_count += (predicted_labels == true_labels_int.squeeze()).sum().item()
                    total_samples_processed += current_num_graphs_in_batch # Incrementa solo se abbiamo etichette per validare

                    if criterion_obj is not None:
                        loss_value_current_batch = torch.tensor(0.0, device=device)
                        if criterion_type == "ce":
                            target = true_labels_int
                            if num_classes_dataset == 1:
                                target = target.float().unsqueeze(1)
                            loss_value_current_batch = criterion_obj(output_logits, target)
                        elif criterion_type == "gcod":
                            if graph_embeddings is None:
                                raise ValueError("graph_embeddings sono None, ma richiesti per GCOD.")
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
                            loss_value_current_batch = l1 + l2 + lambda_l3_weight * l3
                        else:
                            raise ValueError(f"Unsupported criterion_type: {criterion_type}")
                        total_loss_accumulator += loss_value_current_batch.item() * current_num_graphs_in_batch
                    else:
                        print("Warning: criterion_obj is None durante la validazione.")
                else:
                    print("Warning: data_batch.y is None durante la validazione per un batch.")
            # else (is_validation=False): si accumulano solo le predizioni. total_samples_processed non viene incrementato qui
            # perché non calcoliamo metriche basate su etichette.

    avg_loss_final = 0.0
    accuracy_final = 0.0
    f1_final = 0.0

    if is_validation:
        if total_samples_processed > 0 : # total_samples_processed ora conta solo campioni con etichette valide
            avg_loss_final = total_loss_accumulator / total_samples_processed
            accuracy_final = correct_predictions_count / total_samples_processed

            if len(all_labels_list) > 0 and len(all_preds_list) == len(all_labels_list):
                try:
                    avg_type_f1 = 'binary' if num_classes_dataset == 2 else 'macro'
                    if len(set(all_labels_list)) < 2 and avg_type_f1 == 'macro':
                        print("Warning: Meno di 2 classi uniche nelle etichette vere per F1 macro.")
                    f1_final = f1_score(all_labels_list, all_preds_list, average=avg_type_f1, zero_division=0)
                except ValueError as e:
                    print(f"Errore nel calcolo F1-score: {e}. F1 impostato a 0.")
            elif len(all_labels_list) > 0 : # Preds e labels non matchano in lunghezza, ma abbiamo labels
                print(f"Warning: Lunghezza di predizioni ({len(all_preds_list)}) ed etichette ({len(all_labels_list)}) non corrisponde. F1 non calcolato.")
        else:
            print("Warning: Nessun campione con etichette valide processato durante la validazione per calcolare metriche.")

    # RITORNA SEMPRE UN TUPLE DI 5 ELEMENTI
    return avg_loss_final, accuracy_final, f1_final, all_preds_list, all_labels_list