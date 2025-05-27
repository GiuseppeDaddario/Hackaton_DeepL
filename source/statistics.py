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





def save_predictions_1(predictions, output_path):
    """
    Salva le predizioni in un file CSV.
    
    Args:
        predictions (List[int] or List[float]): Lista delle predizioni (etichette o probabilità).
        output_path (str): Percorso del file CSV di output.
    """
    # Creazione della directory se non esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Creazione del DataFrame
    df = pd.DataFrame({"id": list(range(len(predictions))), "pred": predictions})

    # Salvataggio CSV
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")






def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()