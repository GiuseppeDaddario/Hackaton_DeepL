from tqdm import tqdm
import torch

def evaluate(data_loader, model, device, calculate_accuracy=False, is_tpu=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            # Passaggio esplicito del parametro train=False
            output = model(data, train=False)
            pred = output[0].argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)

            # Sincronizzazione per TPU
            if is_tpu:
                import torch_xla.core.xla_model as xm
                xm.mark_step()

    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions