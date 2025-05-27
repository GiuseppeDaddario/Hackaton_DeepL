## Imports
import argparse
import gc
import logging
import os
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

## Modular imports (assicurati che questi path e file esistano e siano corretti)
from source.evaluation import evaluate
from source.statistics import save_predictions, plot_training_progress
from source.train import train
from source.dataLoader import add_zeros # Assumi che sia definita o rimuovi se non usata
from source.loadData import GraphDataset
from source.loss import ncodLoss, gcodLoss
from source.models import GNN
from source.utils import set_seed

# Set the random seed
set_seed()

# Funzione per calcolare l'accuratezza globale di training (atrain)
def calculate_global_train_accuracy(model, full_train_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch in tqdm(full_train_loader, desc="Calculating global train accuracy (atrain)", unit="batch", leave=False, disable=True):
            graphs = data_batch.to(device)
            labels_int = graphs.y.to(device)
            outputs_logits, _, _ = model(graphs)
            _, predicted = torch.max(outputs_logits.data, 1)
            total += labels_int.size(0)
            correct += (predicted == labels_int.squeeze()).sum().item()
    model.train()
    if total == 0: return 0.0
    return correct / total

######################################################
##                                                  ##
##                   MAIN FUNCTION                  ##
##                                                  ##
######################################################

def main(args, train_dataset_arg=None, train_loader_for_batches_arg=None, model_arg=None):
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3
    num_dataset_classes = 6

    print("Building the model...")
    model = model_arg
    if model is None:
        if args.gnn == 'gin':
            model = GNN(gnn_type = 'gin', num_class = num_dataset_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        elif args.gnn == 'gin-virtual':
            model = GNN(gnn_type = 'gin', num_class = num_dataset_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
        elif args.gnn == 'gcn':
            model = GNN(gnn_type = 'gcn', num_class = num_dataset_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        elif args.gnn == 'gcn-virtual':
            model = GNN(gnn_type = 'gcn', num_class = num_dataset_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
        else:
            raise ValueError('Invalid GNN type')

    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr_model, weight_decay=1e-4)

    # Usa un nome di riferimento per logs e checkpoints, basato su D per coerenza con il tuo codice originale
    # Potresti volerlo rendere più generico se i nomi cambiano molto
    base_name_for_logs_and_checkpoints = os.path.basename(os.path.dirname(args.train_path_D)) # Scegli una logica
    logs_folder = os.path.join(script_dir, "logs", base_name_for_logs_and_checkpoints)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(logs_folder, exist_ok=True) # Crea anche la cartella plots se plot_training_progress non lo fa
    os.makedirs(os.path.join(logs_folder, "plots"), exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    checkpoint_path_best = os.path.join(script_dir, "checkpoints", f"model_{base_name_for_logs_and_checkpoints}_best.pth")
    checkpoints_folder_epochs = os.path.join(script_dir, "checkpoints", base_name_for_logs_and_checkpoints)
    os.makedirs(checkpoints_folder_epochs, exist_ok=True)

    # Inizializzazione globale per best accuracy e history dei plot
    best_train_accuracy = 0.0
    cumulative_train_losses_history = []
    cumulative_train_accuracies_history = []

    if args.train_all == 1:
        train_paths = {
            "A": args.train_path_A,
            "B": args.train_path_B,
            "C": args.train_path_C,
            "D": args.train_path_D,
        }

        for dataset_id, current_train_path in train_paths.items():
            print(f"\n--- Training on Dataset: {dataset_id} ---")
            print(f"Path: {current_train_path}")

            # Definisci le variabili che verranno create in questo ciclo a None
            # per assicurare che del funzioni anche se qualcosa va storto prima della loro assegnazione
            train_dataset = None
            train_loader_for_batches = None
            full_train_loader_for_atrain = None
            loss_function_obj = None
            optimizer_loss_params = None
            y_values_numpy = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Loading train dataset {dataset_id} from {current_train_path}...")
            train_dataset = GraphDataset(current_train_path, transform=add_zeros if "add_zeros" in globals() else None)

            if not train_dataset or len(train_dataset) == 0:
                print(f"Dataset {dataset_id} is empty or failed to load. Skipping.")
                if train_dataset is not None: del train_dataset # Assicura pulizia
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue

            print(f"Loading train dataset {dataset_id} into DataLoader...")
            train_loader_for_batches = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            if args.criterion in ["ncod", "gcod"]:
                print("Preparing full train loader for atrain calculation...")
                full_train_loader_for_atrain = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

            print(f"Initializing loss function: {args.criterion} for dataset {dataset_id}")
            if not hasattr(train_dataset, 'graphs_dicts_list') or not train_dataset.graphs_dicts_list:
                print(f"Attribute 'graphs_dicts_list' missing or empty in train_dataset for {dataset_id}. Skipping this dataset.")
                del train_dataset, train_loader_for_batches
                if full_train_loader_for_atrain is not None: del full_train_loader_for_atrain
                gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue

            try:
                y_values_numpy = np.array([graph["y"][0] for graph in train_dataset.graphs_dicts_list if graph.get("y") and len(graph["y"]) > 0])
                if len(y_values_numpy) == 0 and args.criterion != "ce":
                    print(f"No valid y_values found for {args.criterion} in dataset {dataset_id}. Skipping training for this dataset.")
                    del train_dataset, train_loader_for_batches, y_values_numpy
                    if full_train_loader_for_atrain is not None: del full_train_loader_for_atrain
                    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue
            except Exception as e:
                print(f"Error processing y_values for dataset {dataset_id}: {e}. Skipping training for this dataset.")
                del train_dataset, train_loader_for_batches
                if full_train_loader_for_atrain is not None: del full_train_loader_for_atrain
                if y_values_numpy is not None: del y_values_numpy
                gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue

            if args.criterion == "ncod":
                loss_function_obj = ncodLoss(
                    sample_labels=y_values_numpy, device=device, num_examp=len(train_dataset),
                    num_classes=num_dataset_classes, ratio_consistency=args.ratio_consistency_ncod, # Aggiungi args se li hai
                    ratio_balance=args.ratio_balance_ncod,   # Aggiungi args se li hai
                    encoder_features=args.emb_dim, total_epochs=args.epochs
                )
                optimizer_loss_params = optim.SGD(loss_function_obj.parameters(), lr=args.lr_u)
            elif args.criterion == "gcod":
                loss_function_obj = gcodLoss(
                    sample_labels_numpy=y_values_numpy, device=device, num_examp=len(train_dataset),
                    num_classes=num_dataset_classes, gnn_embedding_dim=args.emb_dim,
                    total_epochs=args.epochs
                )
                optimizer_loss_params = optim.SGD(loss_function_obj.parameters(), lr=args.lr_u)
            elif args.criterion == "ce":
                loss_function_obj = torch.nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unsupported criterion: {args.criterion}")

            if hasattr(loss_function_obj, 'to'):
                loss_function_obj.to(device)

            num_epochs_for_this_dataset = args.epochs # Epoche per il dataset corrente

            # checkpoint_intervals basato su args.num_checkpoints e num_epochs_for_this_dataset
            checkpoint_intervals = []
            if args.num_checkpoints is not None and args.num_checkpoints > 0:
                if args.num_checkpoints == 1:
                    checkpoint_intervals = [num_epochs_for_this_dataset]
                else:
                    checkpoint_intervals = [int((i + 1) * num_epochs_for_this_dataset / args.num_checkpoints) for i in range(args.num_checkpoints)]

            atrain_value_for_epoch = 0.0 # Rinominato per chiarezza
            print(f"Starting training for dataset {dataset_id} ({num_epochs_for_this_dataset} epochs)...")
            for epoch in range(num_epochs_for_this_dataset):
                current_criterion_for_epoch = args.criterion
                current_loss_function_for_epoch = loss_function_obj
                current_optimizer_loss_params_for_epoch = optimizer_loss_params

                if epoch < args.epoch_boost and args.criterion in ["ncod", "gcod"]:
                    print(f"Dataset {dataset_id} - Epoch {epoch + 1} (Boosting Phase with CE)")
                    current_criterion_for_epoch = "ce"
                    # Devi creare un'istanza temporanea per la CE loss se ncod/gcod hanno parametri incompatibili
                    if not isinstance(loss_function_obj, torch.nn.CrossEntropyLoss): # Evita di ricreare se già CE
                        current_loss_function_for_epoch = torch.nn.CrossEntropyLoss().to(device)
                    current_optimizer_loss_params_for_epoch = None # CE non ha optimizer per i suoi parametri
                    atrain_value_for_epoch = 0.0 # Non calcolare atrain per CE boost
                elif args.criterion in ["ncod", "gcod"] and full_train_loader_for_atrain is not None:
                    atrain_value_for_epoch = calculate_global_train_accuracy(model, full_train_loader_for_atrain, device)

                avg_batch_acc_epoch, epoch_loss_avg = train(
                    atrain_global_value=atrain_value_for_epoch,
                    train_loader=train_loader_for_batches,
                    model=model,
                    optimizer_model=optimizer_model,
                    device=device,
                    optimizer_loss_params=current_optimizer_loss_params_for_epoch,
                    loss_function_obj=current_loss_function_for_epoch,
                    save_checkpoints=(epoch + 1 in checkpoint_intervals),
                    checkpoint_path=os.path.join(checkpoints_folder_epochs, f"model_{base_name_for_logs_and_checkpoints}_ds_{dataset_id}"),
                    current_epoch=epoch,
                    criterion_type=current_criterion_for_epoch,
                    num_classes_dataset=num_dataset_classes,
                    lambda_l3_weight=args.lambda_l3_weight if current_criterion_for_epoch == "gcod" else 0.0,
                    epoch_boost=args.epoch_boost # La funzione train potrebbe usarlo per logica interna
                )

                atrain_log_str = f"{atrain_value_for_epoch:.4f}" if current_criterion_for_epoch in ['ncod', 'gcod'] and epoch >= args.epoch_boost else 'N/A'
                print(f"Dataset {dataset_id} - Epoch {epoch + 1}/{num_epochs_for_this_dataset}, Avg Batch Train Acc: {avg_batch_acc_epoch:.2f}%, Epoch Train Loss: {epoch_loss_avg:.4f}, Atrain: {atrain_log_str}")
                logging.info(f"Dataset {dataset_id} - Epoch {epoch + 1}/{num_epochs_for_this_dataset}, Avg Batch Train Acc: {avg_batch_acc_epoch:.2f}%, Epoch Train Loss: {epoch_loss_avg:.4f}, Atrain: {atrain_log_str}")

                cumulative_train_losses_history.append(epoch_loss_avg)
                cumulative_train_accuracies_history.append(avg_batch_acc_epoch)

                if avg_batch_acc_epoch > best_train_accuracy:
                    best_train_accuracy = avg_batch_acc_epoch
                    torch.save(model.state_dict(), checkpoint_path_best)
                    print(f"Best GLOBAL model (based on avg batch train acc: {avg_batch_acc_epoch:.2f}% on dataset {dataset_id}, epoch {epoch+1}) updated and saved at {checkpoint_path_best}")

            # Plot cumulativo dopo ogni dataset
            cumulative_plot_filename = f"training_progress_cumulative_after_{dataset_id}.png"
            plot_training_progress(cumulative_train_losses_history, cumulative_train_accuracies_history, os.path.join(logs_folder, "plots", cumulative_plot_filename))
            print(f"Cumulative training progress plot saved to {os.path.join(logs_folder, 'plots', cumulative_plot_filename)}")

            # Pulizia esplicita delle risorse del dataset corrente
            print(f"Cleaning up memory after training on dataset {dataset_id}...")
            del train_dataset
            del train_loader_for_batches
            if full_train_loader_for_atrain is not None: del full_train_loader_for_atrain
            if loss_function_obj is not None: del loss_function_obj # Anche la CE loss se creata dinamicamente
            if optimizer_loss_params is not None: del optimizer_loss_params
            if y_values_numpy is not None: del y_values_numpy
            # Le variabili del loop epoch (avg_batch_acc_epoch, ecc.) sono sovrascritte o escono dallo scope

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Memory cleanup for dataset {dataset_id} complete.")

        # Fine del loop sui dataset di training
        print(f"\nFinished training on all datasets. Final best global training accuracy achieved: {best_train_accuracy:.2f}%")

        # Restituzione basata su args.ret
        # Nota: train_dataset e train_loader_for_batches qui non esisterebbero più a causa del del.
        # Se devi restituirli, devi gestire la logica di `del` diversamente per l'ultimo dataset.
        if args.ret == "all":
            print("Warning: 'ret=all' would require specific handling for last dataset's data. Returning model only.")
            return None, model, None
        elif args.ret == "model":
            return model
        # Se args.ret è None, la funzione termina implicitamente restituendo None

    # Sezione Predict
    if args.predict == 1:
        if not os.path.exists(checkpoint_path_best):
            print(f"Error: Best model checkpoint not found at {checkpoint_path_best}. Cannot perform prediction.")
            return

        model.load_state_dict(torch.load(checkpoint_path_best))
        model.eval()
        print(f"Loaded best model from {checkpoint_path_best} for prediction.")

        test_paths = {
            "A": args.test_path_A,
            "B": args.test_path_B,
            "C": args.test_path_C,
            "D": args.test_path_D,
        }

        for dataset_id, current_test_path in test_paths.items():
            print(f"\n--- Predicting on Dataset: {dataset_id} ---")
            print(f"Path: {current_test_path}")

            test_dataset = None
            test_loader = None
            predictions_data = None

            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            print(f"Preparing test dataset {dataset_id}...")
            test_dataset = GraphDataset(current_test_path, transform=add_zeros if "add_zeros" in globals() else None)

            if not test_dataset or len(test_dataset) == 0:
                print(f"Test dataset {dataset_id} is empty or failed to load. Skipping.")
                if test_dataset is not None: del test_dataset
                gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue

            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            print(f"Generating predictions for the test set {dataset_id}...")
            predictions_data = evaluate(test_loader, model, device, calculate_accuracy=False) # evaluate restituisce le predizioni

            prediction_output_folder = os.path.join(script_dir, "predictions", base_name_for_logs_and_checkpoints)
            os.makedirs(prediction_output_folder, exist_ok=True)
            # L'argomento `args.test_path` non esiste nel parser, crea un nome file specifico.
            prediction_output_file = os.path.join(prediction_output_folder, f"predictions_{dataset_id}.txt")
            save_predictions(predictions_data, prediction_output_file)
            print(f"Predictions for {dataset_id} saved successfully to {prediction_output_file}.")

            print(f"Cleaning up memory after predicting on dataset {dataset_id}...")
            del test_dataset, test_loader, predictions_data
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Se non c'è train_all né predict, la funzione potrebbe restituire il modello non addestrato se è stato creato
    if args.train_all == 0 and args.predict == 0 and model_arg is None: # Se il modello è stato creato ma non addestrato/usato
        return model
    elif args.train_all == 0 and args.predict == 0 and model_arg is not None: # Se un modello è stato passato
        return model_arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_all", type=int, default=1, choices=[0, 1], help="Enable training on all specified datasets.")
    parser.add_argument("--train_path_A", type=str, default="datasets/A/train.json.gz", help="Path to training dataset A.")
    parser.add_argument("--train_path_B", type=str, default="datasets/B/train.json.gz", help="Path to training dataset B.")
    parser.add_argument("--train_path_C", type=str, default="datasets/C/train.json.gz", help="Path to training dataset C.")
    parser.add_argument("--train_path_D", type=str, default="datasets/D/train.json.gz", help="Path to training dataset D.")

    parser.add_argument("--criterion", type=str, default="gcod", choices=["ce", "ncod", "gcod"], help="Loss function type.")
    parser.add_argument("--lr_model", type=float, default=0.001, help="Learning rate for the GNN model.")
    parser.add_argument("--lr_u", type=float, default=0.01, help="Learning rate for NCOD/GCOD parameters.")
    parser.add_argument("--lambda_l3_weight", type=float, default=0.7, help="Weight for L3 in GCOD loss.")
    # Aggiungi qui gli argomenti per ratio_consistency_ncod e ratio_balance_ncod se li usi
    parser.add_argument("--ratio_consistency_ncod", type=float, default=0.0, help="NCOD ratio_consistency.")
    parser.add_argument("--ratio_balance_ncod", type=float, default=0.0, help="NCOD ratio_balance.")


    parser.add_argument("--test_path_A", type=str, default="datasets/A/test.json.gz", help="Path to test dataset A.")
    parser.add_argument("--test_path_B", type=str, default="datasets/B/test.json.gz", help="Path to test dataset B.")
    parser.add_argument("--test_path_C", type=str, default="datasets/C/test.json.gz", help="Path to test dataset C.")
    parser.add_argument("--test_path_D", type=str, default="datasets/D/test.json.gz", help="Path to test dataset D.")

    parser.add_argument("--predict", type=int, default=1, choices=[0, 1], help="Enable prediction on test sets.")
    parser.add_argument("--num_checkpoints", type=int, default=3, help="Number of intermediate checkpoints per dataset (0 for none, 1 for end only).")
    parser.add_argument('--device', type=int, default=0, help='GPU device ID if available (e.g., 0).')
    parser.add_argument('--gnn', type=str, default='gin', choices=['gin', 'gin-virtual', 'gcn', 'gcn-virtual'], help='GNN architecture type.')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='Dropout ratio.')
    parser.add_argument('--num_layer', type=int, default=5, help='Number of GNN layers.')
    parser.add_argument('--emb_dim', type=int, default=300, help='Embedding dimensionality.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train per dataset.')
    parser.add_argument('--epoch_boost', type=int, default=0, help='Number of initial epochs with CE loss for NCOD/GCOD.')
    parser.add_argument('--ret', type=str, default=None, choices=[None, "model", "all"], help="Return value for Kaggle (model or all - 'all' has limitations).")

    args = parser.parse_args()
    main(args)