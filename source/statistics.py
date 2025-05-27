import os
import matplotlib.pyplot as plt
import pandas as pd


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


def plot_training_progress(losses_dict, accuracies_dict, output_dir):
    """
    Genera e salva i plot dell'andamento di loss e accuracy durante il training.
    Args:
        losses_dict (dict): Dizionario contenente le liste delle loss.
                            Es: {"train_loss": [0.5, 0.4, ...], "val_loss": [0.6, 0.5, ...]}
        accuracies_dict (dict): Dizionario contenente le liste delle accuracies.
                                Es: {"train_acc": [70.0, 75.0, ...], "val_acc": [65.0, 72.0, ...]}
        output_dir (str): Cartella dove salvare i plot.
    """
    # Determina il numero di epoche dalla lunghezza della prima history disponibile
    # (assumendo che tutte le history abbiano la stessa lunghezza)
    num_epochs = 0
    if "train_loss" in losses_dict and losses_dict["train_loss"]:
        num_epochs = len(losses_dict["train_loss"])
    elif "train_acc" in accuracies_dict and accuracies_dict["train_acc"]:
        num_epochs = len(accuracies_dict["train_acc"])

    if num_epochs == 0:
        print("Warning: No data found to plot training progress.")
        return

    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(15, 6)) # Aumenta la larghezza per due subplot

    # --- Plot Loss ---
    plt.subplot(1, 2, 1)
    if "train_loss" in losses_dict and losses_dict["train_loss"]:
        plt.plot(epochs_range, losses_dict["train_loss"], 'bo-', label="Training Loss", linewidth=2, markersize=5)
    if "val_loss" in losses_dict and losses_dict["val_loss"]:
        plt.plot(epochs_range, losses_dict["val_loss"], 'ro-', label="Validation Loss", linewidth=2, markersize=5)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss per Epoch', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # --- Plot Accuracy ---
    plt.subplot(1, 2, 2)
    if "train_acc" in accuracies_dict and accuracies_dict["train_acc"]:
        plt.plot(epochs_range, accuracies_dict["train_acc"], 'go-', label="Training Accuracy", linewidth=2, markersize=5)
    if "val_acc" in accuracies_dict and accuracies_dict["val_acc"]:
        plt.plot(epochs_range, accuracies_dict["val_acc"], 'mo-', label="Validation Accuracy", linewidth=2, markersize=5)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12) # Assumendo che le accuracy siano in %
    plt.title('Training and Validation Accuracy per Epoch', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if accuracies_dict.get("train_acc") or accuracies_dict.get("val_acc"): # Solo se ci sono dati di acc
        all_acc_values = []
        if "train_acc" in accuracies_dict: all_acc_values.extend(accuracies_dict["train_acc"])
        if "val_acc" in accuracies_dict: all_acc_values.extend(accuracies_dict["val_acc"])
        if all_acc_values: # Solo se ci sono valori di accuracy
            min_acc = min(all_acc_values) if all_acc_values else 0
            max_acc = max(all_acc_values) if all_acc_values else 100
            plt.ylim([max(0, min_acc - 5), min(100, max_acc + 5)]) # Adatta ylim per l'accuracy

    # Salvataggio del plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "training_validation_progress.png")
    try:
        plt.tight_layout(pad=2.0) # Aggiunge un po' di padding
        plt.savefig(plot_path)
        plt.close() # Chiudi la figura per liberare memoria
        # logging.info(f"Training progress plot saved to {plot_path}") # Se usi logging
        print(f"Training progress plot saved to {plot_path}")
    except Exception as e:
        # logging.error(f"Failed to save plot to {plot_path}: {e}") # Se usi logging
        print(f"Error saving plot to {plot_path}: {e}")
        plt.close()