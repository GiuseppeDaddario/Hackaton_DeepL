

##############################
##       EVALUATION         ##
##############################
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

## Returns predictions and optionally accuracy
def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="EVALUATION -->", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions


## Returns accuracy and average loss


def evaluateTwo(data_loader, model, device, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluation -->", unit="batch"):
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item()

            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return accuracy, avg_loss, f1

##############################
##############################











#####################################
##     Saving Predictions to CSV   ##
#####################################

import pandas as pd
import matplotlib.pyplot as plt
import os


def save_predictions(predictions, test_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_folder = os.path.join(os.path.dirname(script_dir), "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path)) #Extracts: A/B/C/D
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

##################################
##################################








##################################
### Plotting Training Progress ###
##################################


def plot_training_progress(train_losses, train_accuracies, train_f1s,
                           valid_losses, valid_accuracies, valid_f1s,
                           output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 6))  # Più larga per i 3 grafici

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color='blue')
    plt.plot(epochs, valid_losses, label="Valid Loss", color='skyblue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy", color='green')
    plt.plot(epochs, valid_accuracies, label="Valid Accuracy", color='lightgreen')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True)

    # Plot F1-score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1s, label="Train F1", color='orange')
    plt.plot(epochs, valid_f1s, label="Valid F1", color='gold')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Epoch')
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()


##################################
##################################



