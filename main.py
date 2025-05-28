import argparse
import logging
import os
from types import SimpleNamespace

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import time

# Modular imports
from source.evaluation import evaluate_model
from source.statistics import save_predictions, plot_training_progress
from source.train import train_epoch
from source.dataLoader import add_zeros # Utilizzato da GraphDataset
from source.loadData import GraphDataset # La tua classe GraphDataset
from source.loss import gcodLoss, LabelSmoothingCrossEntropy
from source.models import GNN, CustomGNN
from source.utils import set_seed

set_seed()


def calculate_global_train_accuracy(model, full_train_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch in tqdm(full_train_loader, desc="Calculating global train accuracy (atrain)", unit="batch", leave=False, disable=True):
            graphs = data_batch.to(device)
            print("graphs.x shape:", graphs.x.shape)
            labels_int = graphs.y.to(device)
            outputs_logits, _, _ = model(graphs)
            _, predicted = torch.max(outputs_logits.data, 1)
            total += labels_int.size(0)
            correct += (predicted == labels_int.squeeze()).sum().item()
    if total == 0: return 0.0
    return correct / total

def main(args, full_train_dataset=None, train_loader=None, val_loader=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")

    set_seed(args.seed)

    # --- Setup Cartelle e Logging ---
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
    logging.info("\n" + "="*60)
    logging.info(">>> Starting main script")
    logging.info("="*60 + "\n")
    logging.info(f"• Device     : {device}")
    logging.info(f"• Input args : {args}\n")

    num_dataset_classes = 6
    num_edge_features_resolved = args.num_edge_features
    sample_graph = GraphDataset(args.train_path, transform=add_zeros)[0]
    num_node_features_resolved = sample_graph.edge_attr.size(1)

    # --- Costruzione Modello ---
    logging.info(">>> Building the model...")
    if args.gnn_type == "transformer":
        model = GNN(
            gnn_type=args.gnn_type,
            num_class=num_dataset_classes,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            JK=args.jk_mode,
            graph_pooling=args.graph_pooling,
            num_edge_features=num_edge_features_resolved,
            transformer_heads=args.transformer_heads,
            residual=not args.no_residual
        ).to(device)
    elif args.gnn_type == "gated-gcn":
        cfg_custom = SimpleNamespace(
            dataset=SimpleNamespace(
                node_encoder_bn=False, # Valore di default se non specificato diversamente
                edge_encoder_bn=False  # Valore di default
            ),
            model=SimpleNamespace(
                type='custom_gnn',
                loss_fun='cross_entropy', # Questo influenzerà la scelta di args.criterion
                edge_decoding='dot',    # Non usato direttamente in CustomGNN forward
                graph_pooling='mean'    # Usato in GNNHeadPlaceholder
            ),
            gnn=SimpleNamespace(
                head='mlp_graph',             # Usato per GNNHeadPlaceholder
                layers_mp=4,
                layer_type='gatedgcn',      # Usato per GatedGCNLayer
                layers_pre_mp=0,
                layers_post_mp=2,           # Usato in GNNHeadPlaceholder
                dim_inner=512,
                batchnorm=True,
                act='relu', # Stringa, verrà convertita in funzione nn.ReLU
                dropout=0.15,
                agg='mean',                 # Non usato direttamente da GatedGCNLayer, l'aggregazione è implicita
                normalize_adj=False,        # Non usato da GatedGCNLayer
                ffn=False,                  # Passato a GatedGCNLayer
                residual=True
            )
        )
        model = CustomGNN(
            dim_node_feat_raw=num_node_features_resolved,
            dim_out=num_dataset_classes,
            cfg=cfg_custom
        ).to(device)
        logging.info(f"• Model architecture      : {args.gnn_type} (CustomGNN with GatedGCNLayer)")
        logging.info(f"• Layers Pre-MP           : {cfg_custom.gnn.layers_pre_mp}")
        logging.info(f"• Layers MP (GatedGCN)    : {cfg_custom.gnn.layers_mp}")
        logging.info(f"• Layers Post-MP (Head)   : {cfg_custom.gnn.layers_post_mp}")
        logging.info(f"• Inner dimension         : {cfg_custom.gnn.dim_inner}")
        logging.info(f"• Node features (raw)   : {num_node_features_resolved}")
        logging.info(f"• Edge features (raw)   : {num_edge_features_resolved}")
        logging.info(f"• Dropout                 : {cfg_custom.gnn.dropout}")
        logging.info(f"• Graph Pooling (Head)    : {cfg_custom.model.graph_pooling}")
        logging.info(f"• Batchnorm               : {cfg_custom.gnn.batchnorm}")
        logging.info(f"• Residual                : {cfg_custom.gnn.residual}")
        logging.info(f"• Activation              : {cfg_custom.gnn.act}")


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"• Model architecture      : {args.gnn_type}")
    logging.info(f"• Number of layers        : {args.num_layer}")
    logging.info(f"• Embedding dimension     : {args.emb_dim}")
    logging.info(f"• Dropout                 : {args.drop_ratio}")
    logging.info(f"• Pooling                 : {args.graph_pooling}")
    logging.info(f"• Residual                : {not args.no_residual}")
    logging.info(f"• Total parameters        : {total_params:,}\n")


    optimizer_model = torch.optim.AdamW(model.parameters(), lr=args.lr_model, weight_decay=args.weight_decay)

    checkpoint_path_best = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder_epochs = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder_epochs, exist_ok=True)

    if os.path.exists(checkpoint_path_best) and not args.train_path:
        model.load_state_dict(torch.load(checkpoint_path_best))
        print(f"Loaded best model from {checkpoint_path_best}")

    if args.num_checkpoints is not None and args.num_checkpoints > 0:
        if args.num_checkpoints == 1: checkpoint_intervals = [args.epochs]
        else: checkpoint_intervals = [int((i + 1) * args.epochs / args.num_checkpoints) for i in range(args.num_checkpoints)]
    else: checkpoint_intervals = []

    # --- Training ---
    if args.train_path:
        logging.info(">>> Preparing train and validation datasets...")
        if full_train_dataset is None:
            full_train_dataset = GraphDataset(args.train_path, transform=add_zeros) # Assumendo che add_zeros sia corretto

        try:
            labels_for_split = [data.y.item() for data in full_train_dataset if data.y is not None]
        except Exception as e:
            logging.error(f"Could not extract labels for stratified split: {e}. Ensure dataset elements have a .y attribute.")
            raise

        train_indices, val_indices = train_test_split(
            range(len(full_train_dataset)),
            test_size=args.val_split_ratio,
            shuffle=True,
            stratify=labels_for_split,
            random_state=args.seed
        )

        train_dataset_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
        val_dataset_subset = torch.utils.data.Subset(full_train_dataset, val_indices)

        logging.info("\n" + "-"*60)
        logging.info(">> Dataset: split in training/validation")
        logging.info("-"*60 + "\n")

        logging.info(f"• Full dataset size       : {len(full_train_dataset)}")
        logging.info(f"• Training set            : {len(train_dataset_subset)} campioni")
        logging.info(f"• Validation set          : {len(val_dataset_subset)} campioni\n")

        if train_loader is None:
            train_loader = DataLoader(train_dataset_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        if val_loader is None:
            val_loader = DataLoader(val_dataset_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')

        full_train_loader_for_atrain = None
        if args.criterion == "gcod":
            logging.info(">>> Preparing current training subset loader for atrain calculation (used by GCOD)...")
            full_train_loader_for_atrain = DataLoader(train_dataset_subset, batch_size=args.batch_size, shuffle=False)

        # --- Inizializzazione Loss Function ---
        criterion_obj = None
        optimizer_loss_params = None

        if args.criterion == "ce":
            criterion_obj = LabelSmoothingCrossEntropy(classes=num_dataset_classes, smoothing=args.label_smoothing).to(device)
        elif args.criterion == "gcod":

            if not hasattr(full_train_dataset, 'graphs_dicts_list'):
                logging.warning("full_train_dataset does not have 'graphs_dicts_list'. GCOD might not initialize correctly if 'original_idx' handling is not robust.")
                y_values_numpy_for_gcod = np.array([d.y.item() for d in full_train_dataset if d.y is not None])
            else:
                y_values_numpy_for_gcod = np.array([
                    graph_dict["y"][0] for graph_dict in full_train_dataset.graphs_dicts_list
                    if graph_dict.get("y") and len(graph_dict["y"]) > 0
                ])

            if len(y_values_numpy_for_gcod) != len(full_train_dataset):
                logging.warning(f"Mismatch in length for GCOD y_values ({len(y_values_numpy_for_gcod)}) and full_train_dataset ({len(full_train_dataset)}). This can be problematic.")

            logging.info("\n" + "-"*60)
            logging.info(">> Initializing Loss Function")
            logging.info("-"*60 + "\n")

            logging.info(f"• Loss function used    : GCODLoss")
            logging.info(f"• Number of classes     : {num_dataset_classes}")
            logging.info(f"• Embeddings size       : {args.emb_dim}")
            logging.info(f"• Total train samples   : {len(full_train_dataset)}")
            logging.info(f"• Total epochs          : {args.epochs}")
            logging.info(f"• Device                : {device}\n")

            criterion_obj = gcodLoss(
                sample_labels_numpy=y_values_numpy_for_gcod,
                device=device,
                num_examp=len(full_train_dataset), # Numero totale di esempi originali
                num_classes=num_dataset_classes,
                gnn_embedding_dim=args.emb_dim,
                total_epochs=args.epochs
            ).to(device)
            optimizer_loss_params = optim.SGD(criterion_obj.parameters(), lr=args.lr_u)
        else:
            raise ValueError(f"Unsupported criterion: {args.criterion}")

        logging.info(">>> Loss function initialized and moved to device\n")

        # --- Training Loop ---
        best_val_f1 = 0.0
        epochs_no_improve = 0
        train_losses_history, train_f1_history, train_accuracy_history = [], [], []
        val_losses_history, val_f1_history, val_accuracy_history = [], [], []
        atrain_global = 0.0
        logging.info(">>> Starting training loop...")
        start_time_train = time.time()

        for epoch in range(args.epochs):
            epoch_start_time = time.time()


            if args.criterion == "gcod" and full_train_loader_for_atrain is not None:
                atrain_global = calculate_global_train_accuracy(model, full_train_loader_for_atrain, device)

            avg_train_loss, avg_train_accuracy, train_f1 = train_epoch(
                model=model, loader=train_loader, optimizer_model=optimizer_model, device=device,
                criterion_obj=criterion_obj, criterion_type=args.criterion,
                optimizer_loss_params=optimizer_loss_params, num_classes_dataset=num_dataset_classes,
                lambda_l3_weight=args.lambda_l3_weight, current_epoch=epoch,
                atrain_global_value=atrain_global, save_checkpoints=(epoch + 1 in checkpoint_intervals), checkpoint_path=os.path.join(checkpoints_folder_epochs, f"model_{test_dir_name}"),
                epoch_boost=args.epoch_boost,
                gradient_clipping_norm=args.gradient_clipping
            )
            train_losses_history.append(avg_train_loss)
            train_f1_history.append(train_f1)
            train_accuracy_history.append(avg_train_accuracy)

            avg_val_loss, val_acc, val_f1 = evaluate_model(
                model=model, loader=val_loader, device=device, criterion_obj=criterion_obj,
                criterion_type=args.criterion if not (args.criterion == "gcod" and epoch < args.epoch_boost) else "ce",
                num_classes_dataset=num_dataset_classes, lambda_l3_weight=args.lambda_l3_weight,
                current_epoch_for_gcod=epoch, atrain_for_gcod=atrain_global, is_validation=True
            )
            val_losses_history.append(avg_val_loss)
            val_f1_history.append(val_f1 * 100)
            val_accuracy_history.append(val_acc)


            epoch_duration = time.time() - epoch_start_time
            atrain_log_str = f"{atrain_global:.4f}" if args.criterion == 'gcod' and full_train_loader_for_atrain is not None else 'N/A'

            logging.info(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train F1: {train_f1 * 100:.2f}% | "
                f"Train Acc: {avg_train_accuracy * 100:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val F1: {val_f1 * 100:.2f}% | "
                f"Val Acc: {val_acc * 100:.2f}% | "
                f"Time: {epoch_duration:.2f}s | "
                f"a_train: {atrain_log_str}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), checkpoint_path_best)
                logging.info(f"Best validation model updated: {checkpoints_folder_epochs} (Val F1: {best_val_f1*100:.2f}%)")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement for {epochs_no_improve} epoch(s). Best Val F1: {best_val_f1*100:.2f}%")


        total_training_time = time.time() - start_time_train
        logging.info("\n" + "="*60)
        logging.info(f">>> Training completed in {total_training_time:.2f} seconds")
        logging.info("="*60 + "\n")
        plot_training_progress({"train_loss": train_losses_history, "val_loss": val_losses_history},
                               {"train_acc": train_accuracy_history, "train_f1": train_f1_history, "val_acc": val_accuracy_history, "val_f1": val_f1_history},
                               os.path.join(logs_folder, "training_plots"))

    # --- Predizione sul Test Set ---
    if args.predict == 1:
        if not os.path.exists(args.test_path):
            logging.error(f"Test path {args.test_path} DNE.")
            return

        checkpoint_to_load_path = checkpoint_path_best
        if not os.path.exists(checkpoint_to_load_path):
            logging.error(f"No model checkpoint found at {checkpoint_to_load_path}.")
            return

        logging.info(f">>> Loading model from {checkpoint_to_load_path} for prediction.")
        loaded_data = torch.load(checkpoint_to_load_path, map_location=device)

        if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
            model.load_state_dict(loaded_data['model_state_dict'])
        else:
            model.load_state_dict(loaded_data)

        logging.info(">>> Preparing test dataset...")
        test_dataset = GraphDataset(args.test_path, transform=add_zeros) # Assumendo che add_zeros sia corretto
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        logging.info(">>> Generating predictions for the test set...")
        test_predictions = evaluate_model(
            model=model, loader=test_loader, device=device,
            criterion_obj=None, criterion_type=args.criterion, is_validation=False
        )

        output_prediction_path = os.path.join(logs_folder, f"predictions_{test_dir_name}.csv")
        save_predictions(test_predictions, output_prediction_path) # Assicurati che save_predictions accetti il path
        logging.info(f"Predictions saved to {output_prediction_path}")

    logging.info("Main script finished <<<")
    return full_train_dataset, train_loader, val_loader


TEST_PATH = "../datasets/B/test.json.gz"  # Replace with actual test dataset path
TRAIN_PATH = "../datasets/B/train.json.gz"   # Optional, replace with actual train dataset path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")

    # Dataset and Paths
    parser.add_argument("--train_path", type=str,default= TRAIN_PATH, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, default = TEST_PATH, help="Path to the test dataset.")

    parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints to save during training (default: 5). Set to 0 to disable.")
    # Model Architecture
    parser.add_argument('--gnn_type', type=str, default='transformer', choices=['transformer','gated-gcn'], help='GNN architecture type (default: transformer)')
    parser.add_argument('--num_layer', type=int, default=3, help='Number of GNN message passing layers (default: 3)')
    parser.add_argument('--emb_dim', type=int, default=204, help='Dimensionality of hidden units in GNNs (default: 128)')
    parser.add_argument('--drop_ratio', type=float, default=0.1, help='Dropout ratio (default: 0.1)')
    parser.add_argument('--transformer_heads', type=int, default=3, help='Number of attention heads for TransformerConv (default: 4)')
    parser.add_argument('--num_edge_features', type=int, default=7, help='Dimensionality of edge features (default: 7, VERIFY FROM DATASET)')
    parser.add_argument('--jk_mode', type=str, default="mean", choices=["last", "sum", "mean", "concat"], help="Jumping Knowledge mode (default: last)")
    parser.add_argument('--graph_pooling', type=str, default="attention", choices=["sum", "mean", "max", "attention", "set2set"], help="Graph pooling method (default: mean)")
    parser.add_argument('--no_residual', action='store_true', help='Disable residual connections in GNN layers.')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32)')
    parser.add_argument('--lr_model', type=float, default=1e-3, help='Learning rate for the GNN model (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty) for AdamW (default: 1e-5)')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Max norm for gradient clipping (0 to disable, default: 1.0)')
    parser.add_argument('--val_split_ratio', type=float, default=0.15, help='Fraction of training data to use for validation (default: 0.15)')

    # Loss Function and GCOD specific
    parser.add_argument("--criterion", type=str, default="gcod", choices=["ce", "gcod"], help="Type of loss to use (default: ce)")
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Amount of label smoothing for CE loss (default: 0.1, use 0 for no smoothing)')
    parser.add_argument("--lr_u", type=float, default=0.01, help="Learning rate for 'u' parameters in GCOD (default: 0.01)")
    parser.add_argument("--lambda_l3_weight", type=float, default=0.7, help="Weight for L3 component in GCOD loss (default: 0.7)")
    parser.add_argument('--epoch_boost', type=int, default=0, help='Number of initial epochs with CE loss before GCOD (default: 0)')

    # Runtime and Misc
    parser.add_argument('--device', type=int, default=0, help='GPU device ID to use if available (default: 0, -1 for CPU)')
    parser.add_argument('--seed', type=int, default=777, help='Random seed (default: 777)')
    parser.add_argument("--predict", type=int, default=1, choices=[0,1], help="Generate and save predictions on the test set (default: 1)")
    parser.add_argument('--num_workers', type=int, default=0, help='Number of Dataloader workers (default: 0 for main process, >0 for multiprocessing)')

    args = parser.parse_args()
    main(args, full_train_dataset=None, train_loader=None, val_loader=None)