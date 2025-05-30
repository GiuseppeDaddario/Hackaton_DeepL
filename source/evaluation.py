import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F

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
    total_samples_processed = 0

    all_preds_list = []
    all_labels_list = []

    desc_str = "Validating" if is_validation else "Testing (Predicting)"
    disable_tqdm = False

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
                raise ValueError(f"Output model format not recognized: {type(model_output)}")

            if output_logits is None or output_logits.shape[0] == 0:
                continue

            if num_classes_dataset is None:
                raise ValueError("num_classes_dataset can't be None.")

            if num_classes_dataset > 1:
                _, predicted_labels = torch.max(output_logits, 1)
            elif num_classes_dataset == 1:
                predicted_labels = (torch.sigmoid(output_logits) > 0.5).squeeze().long()
            else:
                raise ValueError(f"num_classes_dataset ({num_classes_dataset}) not ok.")

            all_preds_list.extend(predicted_labels.cpu().numpy().tolist())
            current_num_graphs_in_batch = data_batch.num_graphs if hasattr(data_batch, 'num_graphs') else output_logits.size(0)

            if is_validation:
                if hasattr(data_batch, 'y') and data_batch.y is not None:
                    true_labels_int = data_batch.y.to(device)
                    all_labels_list.extend(true_labels_int.cpu().numpy().tolist())
                    correct_predictions_count += (predicted_labels == true_labels_int.squeeze()).sum().item()
                    total_samples_processed += current_num_graphs_in_batch

                    if criterion_obj is not None:
                        loss_value_current_batch = torch.tensor(0.0, device=device)
                        if criterion_type == "ce":
                            target = true_labels_int
                            if num_classes_dataset == 1:
                                target = target.float().unsqueeze(1)
                            loss_value_current_batch = criterion_obj(output_logits, target)
                        elif criterion_type == "gcod":
                            if graph_embeddings is None:
                                raise ValueError("graph_embeddings are None, but required for GCOD loss calculation")
                            if not hasattr(data_batch, 'original_idx'):
                                raise ValueError("For GCOD, 'original_idx' must be present in the batch")

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
                        print("Warning: criterion_obj is None during validation, loss will not be calculated")
                else:
                    print("Warning: data_batch.y is None during validation, no labels to compare against")

    avg_loss_final = 0.0
    accuracy_final = 0.0
    f1_final = 0.0

    if is_validation:
        if total_samples_processed > 0 :
            avg_loss_final = total_loss_accumulator / total_samples_processed
            accuracy_final = correct_predictions_count / total_samples_processed

            if len(all_labels_list) > 0 and len(all_preds_list) == len(all_labels_list):
                try:
                    avg_type_f1 = 'binary' if num_classes_dataset == 2 else 'macro'
                    if len(set(all_labels_list)) < 2 and avg_type_f1 == 'macro':
                        print("Warning: Less than 2 unique labels found, setting F1 to 0")
                    f1_final = f1_score(all_labels_list, all_preds_list, average=avg_type_f1, zero_division=0)
                except ValueError as e:
                    print(f"Error while computing F1 score: {e}. F1 set to 0")
            elif len(all_labels_list) > 0 :
                print(f"Warning: all_preds_list ({len(all_preds_list)}) and all_labels_list ({len(all_labels_list)}) have different lengths. F1 score cannot be computed")
        else:
            print("Warning: No samples processed during validation, setting all metrics to 0")

    return avg_loss_final, accuracy_final, f1_final, all_preds_list, all_labels_list