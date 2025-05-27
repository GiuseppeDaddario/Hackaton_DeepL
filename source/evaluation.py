from tqdm import tqdm
import torch
from torch_geometric.data import Batch
import torch_xla.core.xla_model as xm

def evaluate(data_loader, model, device, calculate_accuracy=False, is_tpu=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []

    # Disabilita la barra se non è il master
    data_iterator = tqdm(data_loader, desc=f"Evaluating (Rank {xm.get_ordinal()})", unit="batch", disable=not xm.is_master_ordinal())

    with torch.no_grad():
        for data_dict in data_iterator:
            if not data_dict: continue

            num_graphs_val = data_dict.pop('_num_graphs', None)
            data = Batch(**data_dict)
            if num_graphs_val is not None:
                data.num_graphs = num_graphs_val
            # data è già su device XLA grazie a MpDeviceLoader

            output_logits, _, _ = model(data, train=False)
            pred = output_logits.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())

            if calculate_accuracy:
                if data.y is not None:
                    correct += (pred == data.y.squeeze()).sum().item()
                    total += data.y.size(0)

            if is_tpu:
                xm.mark_step()

    # Per distributed evaluation, aggregare correct e total
    if is_tpu and xm.xrt_world_size() > 1:
       correct_sum = xm.all_reduce(xm.REDUCE_SUM, torch.tensor(correct, device=device))
       total_sum = xm.all_reduce(xm.REDUCE_SUM, torch.tensor(total, device=device))
       correct = correct_sum.item()
       total = total_sum.item()
       all_predictions = xm.all_gather(torch.tensor(predictions, device=device))
       predictions = all_predictions.cpu().numpy().flatten().tolist()


    if calculate_accuracy:
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        return accuracy, predictions

    return predictions