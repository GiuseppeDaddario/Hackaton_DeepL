import os
import matplotlib.pyplot as plt
import pandas as pd



#########################################
##             SAVING CSV              ##
#########################################

# predictions: lista di predizioni da salvare
# test_path: percorso del file di test [VIENE SOVRASCRITTO!]

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
    """
    Genera e salva i plot dell'andamento di loss, accuracy e f1 durante il training.
    Args:
        losses_dict (dict): Es: {"train_loss": [...], "val_loss": [...]}
        accuracies_dict (dict): Es: {"train_acc": [...], "val_acc": [...], "train_f1": [...], "val_f1": [...]}
        output_dir (str): Cartella dove salvare i plot.
    """
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

    plt.figure(figsize=(18, 6))  # 3 subplot in 1 riga

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

    # Salvataggio
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