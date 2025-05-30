import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm


#########################################
##             SAVING CSV              ##
#########################################


def save_predictions(predictions, test_path):
    source_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(source_dir)
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))

    os.makedirs(submission_folder, exist_ok=True)

    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")

    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })

    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

#########################################
##          END OF SAVING CSV          ##
#########################################








#########################################
## GENERATE PLOTS OF TRAINING PROGRESS ##
#########################################
def plot_training_progress(losses_dict, accuracies_dict, output_dir):
    num_epochs = 0
    if "train_loss" in losses_dict and losses_dict["train_loss"]:
        num_epochs = len(losses_dict["train_loss"])
    elif "train_acc" in accuracies_dict and accuracies_dict["train_acc"]:
        num_epochs = len(accuracies_dict["train_acc"])
    elif "train_f1" in accuracies_dict and accuracies_dict["train_f1"]:
        num_epochs = len(accuracies_dict["train_f1"])

    if num_epochs == 0:
        print("Warning: No data found to plot training progress.")
        return

    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(18, 6))

    # --- Plot Loss ---
    plt.subplot(1, 3, 1)
    if "train_loss" in losses_dict:
        plt.plot(epochs_range, losses_dict["train_loss"], 'bo-', label="Training Loss", linewidth=2, markersize=5)
    if "val_loss" in losses_dict:
        plt.plot(epochs_range, losses_dict["val_loss"], 'ro-', label="Validation Loss", linewidth=2, markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # --- Plot Accuracy ---
    plt.subplot(1, 3, 2)
    if "train_acc" in accuracies_dict:
        plt.plot(epochs_range, accuracies_dict["train_acc"], 'go-', label="Training Accuracy", linewidth=2, markersize=5)
    if "val_acc" in accuracies_dict:
        plt.plot(epochs_range, accuracies_dict["val_acc"], 'mo-', label="Validation Accuracy", linewidth=2, markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True)

    # --- Plot F1-score ---
    plt.subplot(1, 3, 3)
    if "train_f1" in accuracies_dict:
        plt.plot(epochs_range, accuracies_dict["train_f1"], 'c^-', label="Training F1-score", linewidth=2, markersize=5)
    if "val_f1" in accuracies_dict:
        plt.plot(epochs_range, accuracies_dict["val_f1"], 'ys-', label="Validation F1-score", linewidth=2, markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('F1-score (%)')
    plt.title('F1-score per Epoch')
    plt.legend()
    plt.grid(True)

    # Saving
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "training_validation_progress.png")
    try:
        plt.tight_layout(pad=2.0)
        plt.savefig(plot_path)
        plt.close()
        print(f"Training progress plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving plot to {plot_path}: {e}")
        plt.close()
#########################################
##              END PLOT               ##
#########################################





#########################################
##         COMPUTING A_TRAIN           ##
#########################################
def calculate_global_train_accuracy(model, full_train_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch in tqdm(full_train_loader, desc="Calculating global train accuracy (atrain)", unit="batch", leave=False, disable=True):
            graphs = data_batch.to(device)
            if graphs.y is None: continue
            labels_int = graphs.y.to(device)

            model_output = model(graphs)
            actual_logits = model_output[0] if isinstance(model_output, tuple) else model_output

            if actual_logits.shape[0] == 0: continue
            _, predicted = torch.max(actual_logits, 1)
            total += labels_int.size(0)
            correct += (predicted == labels_int.squeeze()).sum().item()
    if total == 0: return 0.0
    return correct / total
#########################################
##             END A_TRAIN             ##
#########################################