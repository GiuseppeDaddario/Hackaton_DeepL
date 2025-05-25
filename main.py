## Imports
import argparse
import logging
import os
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np # NECESSARIO per y_values_numpy
from tqdm import tqdm # NECESSARIO per calculate_global_train_accuracy

## Modular imports
from modular.evaluation import evaluate
from modular.statistics import save_predictions, plot_training_progress
from modular.train import train
from modular.dataLoader import add_zeros # Mantengo se ancora usato altrove

from src.loadData import GraphDataset # Assumiamo sia aggiornato
from src.loss import ncodLoss, gcodLoss # Importa gcodLoss
from src.models import GNN
from src.utils import set_seed

# Set the random seed
set_seed()

# Funzione per calcolare l'accuratezza globale di training (atrain)
# Questa funzione è NECESSARIA per GCOD e NCOD (se usi atrain_global)
def calculate_global_train_accuracy(model, full_train_loader, device, num_classes_dataset_calc):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch in tqdm(full_train_loader, desc="Calculating global train accuracy (atrain)", unit="batch", leave=False, disable=True): # disable tqdm se non vuoi output
            graphs = data_batch.to(device)
            labels_int = graphs.y.to(device)

            outputs_logits, _, _ = model(graphs) # Assumendo che il modello restituisca logits, emb, _

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

def main(args):
    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3
    num_dataset_classes = 6 # Definisci il numero di classi del tuo dataset

    print("Building the model...")
    # Costruzione del modello (invariata)
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

    optimizer_model = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Setup logging e checkpoint (invariati)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    logs_folder = os.path.join(script_dir, "logs", test_dir_name) # Potresti voler aggiungere args.gnn e args.criterion qui per organizzazione
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    checkpoint_path_best = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth") # Path per il modello migliore
    checkpoints_folder_epochs = os.path.join(script_dir, "checkpoints", test_dir_name) # Path per i checkpoint intermedi
    os.makedirs(checkpoints_folder_epochs, exist_ok=True)

    if os.path.exists(checkpoint_path_best) and not args.train_path:
        model.load_state_dict(torch.load(checkpoint_path_best))
        print(f"Loaded best model from {checkpoint_path_best}")

    if args.train_path:
        print("Preparing train dataset...")
        train_dataset = GraphDataset(args.train_path, transform=add_zeros if "add_zeros" in globals() else None) # Mantieni add_zeros se esiste e serve

        print("Loading train dataset into DataLoader...")
        train_loader_for_batches = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # DataLoader per calcolare atrain (sull'intero dataset, no shuffle)
        full_train_loader_for_atrain = None
        if args.criterion in ["ncod", "gcod"]: # NECESSARIO per NCOD/GCOD se usano atrain
            print("Preparing full train loader for atrain calculation...")
            # Usa lo stesso train_dataset ma con shuffle=False per atrain
            full_train_loader_for_atrain = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)


        print(f"Initializing loss function: {args.criterion}")
        # Assicurati che train_dataset.graphs_dicts_list esista come da modifica a GraphDataset
        if not hasattr(train_dataset, 'graphs_dicts_list'):
            raise AttributeError("L'oggetto train_dataset deve avere l'attributo 'graphs_dicts_list'. "
                                 "Assicurati che GraphDataset sia stato modificato per caricarlo in __init__.")

        # Converti y_values in numpy array per NCOD/GCOD
        y_values_numpy = np.array([graph["y"][0] for graph in train_dataset.graphs_dicts_list if graph.get("y") and len(graph["y"]) > 0])

        loss_function_obj = None
        optimizer_loss_params = None # Per i parametri 'u' di NCOD/GCOD

        if args.criterion == "ncod":
            loss_function_obj = ncodLoss(
                sample_labels=y_values_numpy, # Modificato per passare numpy array
                device=device,
                num_examp=len(train_dataset),
                num_classes=num_dataset_classes,
                # Mantieni i tuoi valori o rendili args
                ratio_consistency=0,
                ratio_balance=0,
                encoder_features=args.emb_dim, # Assumendo che encoder_features sia emb_dim
                total_epochs=args.epochs
            )
            # La riga train_loss.USE_CUDA non serve se la classe gestisce device internamente
            optimizer_loss_params = optim.SGD(loss_function_obj.parameters(), lr=args.lr_u) # Usa .parameters()
        elif args.criterion == "gcod": # NUOVA SEZIONE PER GCOD
            loss_function_obj = gcodLoss(
                sample_labels_numpy=y_values_numpy,
                device=device,
                num_examp=len(train_dataset),
                num_classes=num_dataset_classes,
                gnn_embedding_dim=args.emb_dim,
                total_epochs=args.epochs
            )
            optimizer_loss_params = optim.SGD(loss_function_obj.parameters(), lr=args.lr_u) # Usa .parameters()
        elif args.criterion == "ce":
            loss_function_obj = torch.nn.CrossEntropyLoss()
            optimizer_loss_params = None # Nessun parametro extra per CE
        else:
            raise ValueError(f"Unsupported criterion: {args.criterion}")

        # Sposta la loss su device se ha il metodo .to() (CrossEntropyLoss non lo ha)
        if hasattr(loss_function_obj, 'to'):
            loss_function_obj.to(device)


        # Training loop
        num_epochs = args.epochs
        best_train_accuracy = 0.0 # Stai usando train_acc per best model
        train_losses_history = []
        train_accuracies_history = []

        if args.num_checkpoints is not None and args.num_checkpoints > 0:
            if args.num_checkpoints == 1:
                checkpoint_intervals = [num_epochs]
            else:
                checkpoint_intervals = [int((i + 1) * num_epochs / args.num_checkpoints) for i in range(args.num_checkpoints)]
        else: # Se num_checkpoints è 0 o None, non salvare checkpoint intermedi
            checkpoint_intervals = []


        atrain_global = 0.0 # Inizializza atrain globale
        print("Starting training...")
        for epoch in range(num_epochs):
            # Calcola atrain_global all'inizio di ogni epoca se si usa NCOD/GCOD
            if args.criterion in ["ncod", "gcod"] and full_train_loader_for_atrain is not None:
                atrain_global = calculate_global_train_accuracy(model, full_train_loader_for_atrain, device, num_dataset_classes)
                # La print di atrain è già dentro calculate_global_train_accuracy se vuoi tenerla lì
                # o loggala qui:
                # logging.info(f"Epoch {epoch + 1} - Global Train Accuracy (atrain): {atrain_global:.4f}")

            # Chiamata aggiornata alla funzione train
            # Il terzo valore restituito da train (vecchio train_acc_cater) non è più usato qui
            # perché atrain_global viene calcolato separatamente.
            avg_batch_acc_epoch, epoch_loss_avg = train(
                atrain_global_value=atrain_global,
                train_loader=train_loader_for_batches,
                model=model,
                optimizer_model=optimizer_model, # Passa l'ottimizzatore del modello
                device=device,
                optimizer_loss_params=optimizer_loss_params, # Passa l'ottimizzatore per i parametri della loss
                loss_function_obj=loss_function_obj, # Passa l'oggetto loss
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder_epochs, f"model_{test_dir_name}"), # Prefisso per i checkpoint intermedi
                current_epoch=epoch,
                criterion_type=args.criterion, # Passa il tipo di criterion
                num_classes_dataset=num_dataset_classes # Passa il numero di classi
            )

            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Batch Train Acc: {avg_batch_acc_epoch:.2f}%, Epoch Train Loss: {epoch_loss_avg:.4f}")
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Avg Batch Train Acc: {avg_batch_acc_epoch:.2f}%, Epoch Train Loss: {epoch_loss_avg:.4f}, Atrain (if used): {atrain_global if args.criterion in ['ncod', 'gcod'] else 'N/A' :.4f}")


            train_losses_history.append(epoch_loss_avg)
            train_accuracies_history.append(avg_batch_acc_epoch) # Stai salvando l'accuratezza media dei batch

            # Save best model basato su avg_batch_acc_epoch
            if avg_batch_acc_epoch > best_train_accuracy:
                best_train_accuracy = avg_batch_acc_epoch
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f"Best model (based on avg batch train acc) updated and saved at {checkpoint_path_best}")

        plot_training_progress(train_losses_history, train_accuracies_history, os.path.join(logs_folder, "plots")) # Path per i plot

    # Sezione predict (invariata, ma assicurati che checkpoint_path_best sia corretto)
    if args.predict == 1:
        if not os.path.exists(checkpoint_path_best):
            print(f"Error: Best model checkpoint not found at {checkpoint_path_best}. Cannot perform prediction.")
            return

        print("Preparing test dataset...")
        test_dataset = GraphDataset(args.test_path, transform=add_zeros if "add_zeros" in globals() else None)
        print("Loading test dataset...")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print("Generating predictions for the test set...")
        model.load_state_dict(torch.load(checkpoint_path_best)) # Carica dal path del modello migliore
        predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
        save_predictions(predictions, args.test_path) # Assicurati che save_predictions salvi dove vuoi
        print("Predictions saved successfully.")


if __name__ == "__main__":
    # Definizione degli argomenti (invariata, ma aggiungi gcod a choices)
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--criterion", type=str, default="ce", choices=["ce", "ncod", "gcod"], help="Type of loss to use (ce, ncod, gcod)") # Aggiunto gcod
    parser.add_argument("--lr_u", type=float, default=0.01, help="lr for u parameters in NCOD/GCOD") # Modificato default
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--predict", type=int, default=1, choices=[0,1], help="Save or not the predictions")
    parser.add_argument("--num_checkpoints", type=int, default=0, help="Number of intermediate checkpoints to save (0 for none, 1 for end only).") # Modificato help
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)') # Modificato default

    args = parser.parse_args()
    main(args)