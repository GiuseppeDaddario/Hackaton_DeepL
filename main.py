## Imports
import argparse
import logging
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Importazioni per TPU
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

## Modular imports
from source.evaluation import evaluate
from source.statistics import save_predictions, plot_training_progress
from source.train import train
from source.dataLoader import add_zeros

from source.loadData import GraphDataset
from source.loss import ncodLoss, gcodLoss
from source.models import GNN
from source.utils import set_seed

# Set the random seed
set_seed()


def custom_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        values = [d[key] for d in batch if d[key] is not None]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values
    return collated


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
            # Necessario per TPU per sincronizzare operazioni
            xm.mark_step()
    model.train()
    if total == 0: return 0.0
    return correct / total


def _run_on_tpu(rank, args):
    # Otteniamo il dispositivo TPU
    device = xm.xla_device()
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3
    num_dataset_classes = 6
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Building the model...")
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

    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    checkpoint_path_best = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder_epochs = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder_epochs, exist_ok=True)

    if os.path.exists(checkpoint_path_best) and not args.train_path:
        # Caricamento adattato per TPU
        model.load_state_dict(torch.load(checkpoint_path_best, map_location="cpu"))
        model = model.to(device)
        print(f"Loaded best model from {checkpoint_path_best}")

    if args.train_path:
        print("Preparing train dataset...")
        train_dataset = GraphDataset(args.train_path, transform=add_zeros if "add_zeros" in globals() else None)
        print("Loading train dataset into DataLoader...")
        # Wrapper DataLoader per TPU
        train_loader_for_batches = pl.MpDeviceLoader(
            DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn),
            device
        )

        full_train_loader_for_atrain = None
        if args.criterion in ["ncod", "gcod"]:
            print("Preparing full train loader for atrain calculation...")
            full_train_loader_for_atrain = pl.MpDeviceLoader(
                DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False),
                device
            )

        print(f"Initializing loss function: {args.criterion}")
        if not hasattr(train_dataset, 'graphs_dicts_list'):
            raise AttributeError("L'oggetto train_dataset deve avere l'attributo 'graphs_dicts_list'.")

        y_values_numpy = np.array([graph["y"][0] for graph in train_dataset.graphs_dicts_list if graph.get("y") and len(graph["y"]) > 0])
        loss_function_obj = None
        optimizer_loss_params = None

        if args.criterion == "ncod":
            loss_function_obj = ncodLoss(
                sample_labels=y_values_numpy,
                device=device,
                num_examp=len(train_dataset),
                num_classes=num_dataset_classes,
                ratio_consistency=0,
                ratio_balance=0,
                encoder_features=args.emb_dim,
                total_epochs=args.epochs
            )
            optimizer_loss_params = optim.SGD(loss_function_obj.parameters(), lr=args.lr_u)
        elif args.criterion == "gcod":
            loss_function_obj = gcodLoss(
                sample_labels_numpy=y_values_numpy,
                device=device,
                num_examp=len(train_dataset),
                num_classes=num_dataset_classes,
                gnn_embedding_dim=args.emb_dim,
                total_epochs=args.epochs
            )
            optimizer_loss_params = optim.SGD(loss_function_obj.parameters(), lr=args.lr_u)
        elif args.criterion == "ce":
            loss_function_obj = torch.nn.CrossEntropyLoss()
            optimizer_loss_params = None
        else:
            raise ValueError(f"Unsupported criterion: {args.criterion}")

        if hasattr(loss_function_obj, 'to'):
            loss_function_obj.to(device)

        num_epochs = args.epochs
        best_train_accuracy = 0.0
        train_losses_history = []
        train_accuracies_history = []

        if args.num_checkpoints is not None and args.num_checkpoints > 0:
            if args.num_checkpoints == 1: checkpoint_intervals = [num_epochs]
            else: checkpoint_intervals = [int((i + 1) * num_epochs / args.num_checkpoints) for i in range(args.num_checkpoints)]
        else: checkpoint_intervals = []

        atrain_global = 0.0
        print("Starting training...")
        loss_fn_ce = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            if epoch < args.epoch_boost:
                print("Current in boosting: CE loss")
            if args.criterion in ["ncod", "gcod"] and full_train_loader_for_atrain is not None:
                atrain_global = calculate_global_train_accuracy(model, full_train_loader_for_atrain, device)

            avg_batch_acc_epoch, epoch_loss_avg = train(
                atrain_global_value=atrain_global,
                train_loader=train_loader_for_batches,
                model=model,
                optimizer_model=optimizer_model,
                device=device,
                optimizer_loss_params=optimizer_loss_params,
                loss_function_obj=loss_function_obj,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder_epochs, f"model_{test_dir_name}"),
                current_epoch=epoch,
                criterion_type=args.criterion,
                num_classes_dataset=num_dataset_classes,
                lambda_l3_weight=args.lambda_l3_weight if args.criterion == "gcod" else 0.0,
                epoch_boost=args.epoch_boost,
                loss_fn_ce=loss_fn_ce,
                is_tpu=True  # Aggiungi questo flag per identificare l'uso di TPU
            )

            # Formatta il logging di atrain per evitare errore se non usato
            atrain_log_str = f"{atrain_global:.4f}" if args.criterion in ['ncod', 'gcod'] else 'N/A'
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Batch Train Acc: {avg_batch_acc_epoch:.2f}%, Epoch Train Loss: {epoch_loss_avg:.4f}")
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Avg Batch Train Acc: {avg_batch_acc_epoch:.2f}%, Epoch Train Loss: {epoch_loss_avg:.4f}, Atrain: {atrain_log_str}")

            train_losses_history.append(epoch_loss_avg)
            train_accuracies_history.append(avg_batch_acc_epoch)

            if avg_batch_acc_epoch > best_train_accuracy:
                best_train_accuracy = avg_batch_acc_epoch
                # Usa xm.save per TPU
                xm.save(model.state_dict(), checkpoint_path_best)
                print(f"Best model (based on avg batch train acc) updated and saved at {checkpoint_path_best}")

        plot_training_progress(train_losses_history, train_accuracies_history, os.path.join(logs_folder, "plots"))

    # Sezione predict per TPU
    if args.predict == 1:
        if not os.path.exists(checkpoint_path_best):
            print(f"Error: Best model checkpoint not found at {checkpoint_path_best}. Cannot perform prediction.")
            return

        print("Preparing test dataset...")
        test_dataset = GraphDataset(args.test_path, transform=add_zeros if "add_zeros" in globals() else None)
        print("Loading test dataset...")
        # Wrapper DataLoader per TPU
        test_loader = pl.MpDeviceLoader(
            DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False),
            device
        )

        print("Generating predictions for the test set...")
        model.load_state_dict(torch.load(checkpoint_path_best, map_location="cpu"))
        model = model.to(device)
        predictions = evaluate(test_loader, model, device, calculate_accuracy=False, is_tpu=True)
        save_predictions(predictions, args.test_path)
        print("Predictions saved successfully.")


def main(args):
    # Avvia l'ambiente TPU
    xmp.spawn(_run_on_tpu, args=(args,), nprocs=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--criterion", type=str, default="gcod", choices=["ce", "ncod", "gcod"], help="Type of loss to use (ce, ncod, gcod)")
    parser.add_argument("--lr_model", type=float, default=0.001, help="learning rate for the main GNN model (default: 0.001)")
    parser.add_argument("--lr_u", type=float, default=0.01, help="lr for u parameters in NCOD/GCOD (default: 0.0001)")
    parser.add_argument("--lambda_l3_weight", type=float, default=0.7, help="Weight for L3 component in GCOD loss when updating model parameters (default: 0.3)")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--predict", type=int, default=1, choices=[0,1], help="Save or not the predictions")
    parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of intermediate checkpoints to save (0 for none, 1 for end only).")
    parser.add_argument('--device', type=int, default=0, help='unused for TPU')
    parser.add_argument('--gnn', type=str, default='gin', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--epoch_boost', type=int, default=0, help='number of epochs to do with CE loss before starting with GCOD')

    args = parser.parse_args()
    main(args)