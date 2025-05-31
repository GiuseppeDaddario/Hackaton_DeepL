import argparse
import logging
import os
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import time

from source.evaluation import evaluate_model
from source.statistics import save_predictions, plot_training_progress, calculate_global_train_accuracy
from source.train import train_epoch
from source.loadData import GraphDataset, AddNodeFeatures
from source.loss import gcodLoss, LabelSmoothingCrossEntropy
from source.models import GNN, GINtrans
from source.utils import set_seed

set_seed()


def main(args, full_train_dataset_outer=None, train_loader_outer=None, val_loader_outer=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")

    set_seed(args.seed)

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
    logging.info("\n" + "="*60 + f"\n>>> Starting main script for GNN type: {args.gnn_type}\n" + "="*60 + "\n")
    logging.info(f"• Device     : {device}")
    logging.info(f"• Input args : {args}\n")

    node_feat_dim_initial = 1
    node_feature_creator = AddNodeFeatures(node_feature_dim=node_feat_dim_initial)

    num_dataset_classes = 6
    num_edge_features_resolved = 7
    num_node_features_initial = node_feat_dim_initial

    full_train_dataset = full_train_dataset_outer
    if args.train_path:
        if full_train_dataset is None:
            logging.info(f">>> Loading training dataset from: {args.train_path}")
            full_train_dataset = GraphDataset(args.train_path, add_zeros_transform=node_feature_creator)

        if len(full_train_dataset) > 0:
            num_dataset_classes = full_train_dataset.num_classes
            num_edge_features_resolved = full_train_dataset.num_edge_features
            if hasattr(full_train_dataset[0], 'x') and full_train_dataset[0].x is not None:
                num_node_features_initial = full_train_dataset[0].x.size(1)
            else:
                num_node_features_initial = node_feat_dim_initial
            logging.info(">>> Parameters inferred from training dataset:")
            logging.info(f"  Num node features (initial): {num_node_features_initial}")
            logging.info(f"  Num edge features          : {num_edge_features_resolved}")
            logging.info(f"  Num classes                : {num_dataset_classes}")
        else:
            logging.error("Training dataset is empty. Cannot infer parameters.")
            return
    else:
        logging.warning(">>> No training path provided. Using parameters from args or defaults.")
        num_dataset_classes = args.num_classes_fallback
        num_edge_features_resolved = args.num_edge_features
        num_node_features_initial = node_feat_dim_initial
        logging.info(">>> Parameters from args/defaults:")
        logging.info(f"  Num node features (initial): {num_node_features_initial}")
        logging.info(f"  Num edge features          : {num_edge_features_resolved}")
        logging.info(f"  Num classes                : {num_dataset_classes}")

    if num_dataset_classes is None:
        logging.error("Number of classes could not be determined. Exiting.")
        return
    if num_edge_features_resolved is None:
        logging.error("Number of edge features could not be determined. Exiting.")
        return

    logging.info(">>> Building the model...")
    if args.gnn_type == 'transformer':
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
        logging.info(f"• Model architecture      : GNN+Transformer")
    elif args.gnn_type == 'GINEtrans':
        logging.info(f"Building GINEtrans...")
        model = GINtrans(
            emb_dim=args.emb_dim,
            edge_input_dim=num_edge_features_resolved,
            num_classes=num_dataset_classes,
            dropout=args.drop_ratio
        ).to(device)
        logging.info(f"• Model architecture      : GINEtrans")
        logging.info(f"• Using emb_dim           : {args.emb_dim}")
        logging.info(f"• Using drop_ratio        : {args.drop_ratio}")
    else:
        raise ValueError(f"Unsupported GNN type: {args.gnn_type}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"• Total parameters        : {total_params:,}\n")

    optimizer_model = torch.optim.AdamW(model.parameters(), lr=args.lr_model, weight_decay=args.weight_decay)

    checkpoint_path_best = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder_epochs = os.path.join(script_dir, "checkpoints", test_dir_name, args.gnn_type)
    os.makedirs(checkpoints_folder_epochs, exist_ok=True)

    if os.path.exists(checkpoint_path_best) and not args.train_path:
        logging.info(f"Loading pre-trained model from {checkpoint_path_best}")
        checkpoint = torch.load(checkpoint_path_best, map_location=device)
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        model.load_state_dict(state_dict)
        #model.load_state_dict(torch.load(checkpoint_path_best, map_location=device))
        logging.info("Model loaded successfully.")

    if args.num_checkpoints is not None and args.num_checkpoints > 0:
        if args.num_checkpoints == 1: checkpoint_intervals = [args.epochs]
        else: checkpoint_intervals = [int((i + 1) * args.epochs / args.num_checkpoints) for i in range(args.num_checkpoints)]
    else: checkpoint_intervals = []

    train_loader = train_loader_outer
    val_loader = val_loader_outer

    if args.train_path:
        if full_train_dataset is None:
            logging.error("full_train_dataset is None, though args.train_path was provided.")
            return

        logging.info(">>> Preparing train and validation datasets...")
        try:
            labels_for_split = [data.y.item() if isinstance(data.y, torch.Tensor) else data.y for data in full_train_dataset if data.y is not None]
            if not all(isinstance(label, (int, float, np.integer)) for label in labels_for_split):
                raise ValueError("Labels for split must be numeric.")
        except Exception as e:
            logging.error(f"Could not extract labels for stratified split: {e}.")
            logging.warning("Falling back to non-stratified split.")
            train_indices, val_indices = train_test_split(
                range(len(full_train_dataset)), test_size=args.val_split_ratio, shuffle=True, random_state=args.seed
            )
        else:
            can_stratify = len(set(labels_for_split)) >= 2 and \
                           (len(full_train_dataset) * args.val_split_ratio >= len(set(labels_for_split)) if args.val_split_ratio > 0 else False)

            if args.val_split_ratio > 0:
                if can_stratify:
                    try:
                        train_indices, val_indices = train_test_split(
                            range(len(full_train_dataset)), test_size=args.val_split_ratio, shuffle=True, stratify=labels_for_split, random_state=args.seed
                        )
                    except ValueError as e_stratify:
                        logging.warning(f"Stratified split failed: {e_stratify}. Falling back to non-stratified split.")
                        train_indices, val_indices = train_test_split(
                            range(len(full_train_dataset)), test_size=args.val_split_ratio, shuffle=True, random_state=args.seed
                        )
                else:
                    logging.warning("Not enough unique labels or validation set too small for stratified split. Using non-stratified split.")
                    train_indices, val_indices = train_test_split(
                        range(len(full_train_dataset)), test_size=args.val_split_ratio, shuffle=True, random_state=args.seed
                    )
            else:
                train_indices = list(range(len(full_train_dataset)))
                val_indices = []

        train_dataset_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
        if val_indices:
            val_dataset_subset = torch.utils.data.Subset(full_train_dataset, val_indices)
        else:
            val_dataset_subset = None

        logging.info("\n" + "-"*60 + "\n>> Dataset: split in training/validation\n" + "-"*60 + "\n")
        logging.info(f"• Full dataset size       : {len(full_train_dataset)}")
        logging.info(f"• Training set            : {len(train_dataset_subset)} samples")
        if val_dataset_subset:
            logging.info(f"• Validation set          : {len(val_dataset_subset)} samples\n")
        else:
            logging.info(f"• Validation set          : 0 samples\n")

        if train_loader is None:
            train_loader = DataLoader(train_dataset_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        if val_loader is None and val_dataset_subset:
            val_loader = DataLoader(val_dataset_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        elif val_loader is None and not val_dataset_subset:
            val_loader = None

        full_train_loader_for_atrain = None
        if args.criterion == "gcod":
            logging.info(">>> Preparing current training subset loader for atrain calculation (used by GCOD)...")
            full_train_loader_for_atrain = DataLoader(train_dataset_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')

        criterion_obj = None
        optimizer_loss_params = None
        criterion_obj_ce = None

        if args.criterion == "ce":
            criterion_obj = LabelSmoothingCrossEntropy(classes=num_dataset_classes, smoothing=args.label_smoothing).to(device)
        elif args.criterion == "gcod":
            y_values_numpy_for_gcod = np.array([
                data.y.item() if isinstance(data.y, torch.Tensor) else data.y
                for data in full_train_dataset if data.y is not None
            ])
            logging.info("\n" + "-"*60 + "\n>> Initializing Loss Function\n" + "-"*60 + "\n")
            logging.info(f"• Loss function used    : GCODLoss")
            logging.info(f"• Embeddings size       : {args.emb_dim}")
            criterion_obj = gcodLoss(
                sample_labels_numpy=y_values_numpy_for_gcod,
                device=device,
                num_examp=len(y_values_numpy_for_gcod),
                num_classes=num_dataset_classes,
                gnn_embedding_dim=args.emb_dim,
                total_epochs=args.epochs
            ).to(device)
            optimizer_loss_params = optim.SGD(criterion_obj.parameters(), lr=args.lr_u)
            if args.epoch_boost > 0:
                criterion_obj_ce = LabelSmoothingCrossEntropy(classes=num_dataset_classes, smoothing=args.label_smoothing).to(device)
        else:
            raise ValueError(f"Unsupported criterion: {args.criterion}")
        logging.info(">>> Loss function initialized and moved to device\n")

        best_val_metric = 0.0
        epochs_no_improve = 0
        train_losses_history, train_f1_history, train_accuracy_history = [], [], []
        val_losses_history, val_f1_history, val_accuracy_history = [], [], []
        atrain_global = 0.0
        logging.info(">>> Starting training loop...")
        start_time_train = time.time()

        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            atrain_global = 0.0
            if args.criterion == "gcod" and full_train_loader_for_atrain is not None:
                atrain_global = calculate_global_train_accuracy(model, full_train_loader_for_atrain, device)

            current_criterion_obj_for_epoch = criterion_obj
            current_criterion_type_for_epoch = args.criterion
            current_optimizer_loss_params_for_epoch = optimizer_loss_params

            if args.criterion == "gcod" and args.epoch_boost > 0 and epoch < args.epoch_boost:
                current_criterion_obj_for_epoch = criterion_obj_ce
                current_criterion_type_for_epoch = "ce"
                current_optimizer_loss_params_for_epoch = None

            avg_train_loss, avg_train_accuracy, train_f1 = train_epoch(
                model=model, loader=train_loader, optimizer_model=optimizer_model, device=device,
                criterion_obj=current_criterion_obj_for_epoch, criterion_type=current_criterion_type_for_epoch,
                optimizer_loss_params=current_optimizer_loss_params_for_epoch, num_classes_dataset=num_dataset_classes,
                lambda_l3_weight=args.lambda_l3_weight, current_epoch=epoch,
                atrain_global_value=atrain_global,
                save_checkpoints=(epoch + 1 in checkpoint_intervals), checkpoint_path=checkpoints_folder_epochs,
                epoch_boost=args.epoch_boost if args.criterion == "gcod" else 0,
                gradient_clipping_norm=args.gradient_clipping
            )
            train_losses_history.append(avg_train_loss)
            train_f1_history.append(train_f1)
            train_accuracy_history.append(avg_train_accuracy)

            avg_val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
            if val_loader:
                avg_val_loss, val_acc, val_f1, _, _ = evaluate_model(
                    model=model, loader=val_loader, device=device,
                    criterion_obj=current_criterion_obj_for_epoch, criterion_type=current_criterion_type_for_epoch,
                    num_classes_dataset=num_dataset_classes, lambda_l3_weight=args.lambda_l3_weight,
                    current_epoch_for_gcod=epoch, atrain_for_gcod=atrain_global, is_validation=True
                )
            val_losses_history.append(avg_val_loss)
            val_f1_history.append(val_f1)
            val_accuracy_history.append(val_acc)

            epoch_duration = time.time() - epoch_start_time
            atrain_log_str = f"{atrain_global:.4f}" if args.criterion == 'gcod' and full_train_loader_for_atrain is not None else 'N/A'
            logging.info(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_accuracy*100:.2f}%, F1: {train_f1*100:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1*100:.2f}% | "
                f"Time: {epoch_duration:.2f}s | a_train: {atrain_log_str}"
            )

            current_val_metric = val_f1 if val_loader else train_f1
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                torch.save(model.state_dict(), checkpoint_path_best)
                logging.info(f"Best model updated and saved to {checkpoint_path_best} (Val F1: {best_val_metric*100:.2f}%)")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement for {epochs_no_improve} epoch(s). Best Val F1: {best_val_metric*100:.2f}%")

            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                logging.info(f"Early stopping triggered after {args.early_stopping_patience} epochs with no improvement.")
                break

        total_training_time = time.time() - start_time_train
        logging.info("\n" + "="*60 + f"\n>>> Training completed in {total_training_time:.2f} seconds\n" + "="*60 + "\n")
        plot_training_progress({"train_loss": train_losses_history, "val_loss": val_losses_history},
                               {"train_acc": train_accuracy_history, "train_f1": train_f1_history, "val_acc": val_accuracy_history, "val_f1": val_f1_history},
                               os.path.join(logs_folder, f"training_plots"))

    if args.predict == 1:
        if not os.path.exists(args.test_path):
            logging.error(f"Test path {args.test_path} does not exist.")
            return

        checkpoint_to_load_path = checkpoint_path_best
        if not os.path.exists(checkpoint_to_load_path):
            logging.error(f"No model checkpoint found at {checkpoint_to_load_path} for GNN type {args.gnn_type}.")
            return

        logging.info(f">>> Loading model from {checkpoint_to_load_path} for prediction.")
        checkpoint = torch.load(checkpoint_to_load_path, map_location=device)
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        model.load_state_dict(state_dict)
        #model.load_state_dict(torch.load(checkpoint_to_load_path, map_location=device))
        logging.info("Model loaded successfully for prediction.")

        logging.info(">>> Preparing test dataset...")
        test_dataset = GraphDataset(args.test_path, add_zeros_transform=node_feature_creator)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        logging.info(">>> Generating predictions for the test set...")
        _ , _ , _ , test_predictions, _ = evaluate_model(
            model=model, loader=test_loader, device=device,
            criterion_obj=None, criterion_type="ce",
            is_validation=False, num_classes_dataset=num_dataset_classes
        )

        submission_folder = os.path.join(script_dir, "submission")
        os.makedirs(submission_folder, exist_ok=True)
        submission_filename = f"predictions_{test_dir_name}.csv"
        submission_path = os.path.join(submission_folder, submission_filename)

        save_predictions(test_predictions, args.test_path)

    logging.info("Main script finished <<<")
    return full_train_dataset, train_loader if 'train_loader' in locals() else train_loader_outer, val_loader if 'val_loader' in locals() else val_loader_outer



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")

    parser.add_argument("--train_path", type=str)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--num_checkpoints", type=int, default=5)

    parser.add_argument('--gnn_type', type=str, default='transformer', choices=['transformer', 'GINEtrans'])
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--drop_ratio', type=float, default=0.1)
    parser.add_argument('--transformer_heads', type=int, default=4)
    parser.add_argument('--transformer_layers', type=int, default=1)
    parser.add_argument('--num_edge_features', type=int, default=7)
    parser.add_argument('--jk_mode', type=str, default="last", choices=["last","concat","sum","max"])
    parser.add_argument('--graph_pooling', type=str, default="attention", choices=["attention", "mean", "max", "sum"])
    parser.add_argument('--no_residual', action='store_true')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_model', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    parser.add_argument('--val_split_ratio', type=float, default=0.15)
    parser.add_argument('--early_stopping_patience', type=int, default=0)

    parser.add_argument("--criterion", type=str, default="gcod", choices=["ce", "gcod"])
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument("--lr_u", type=float, default=0.01)
    parser.add_argument("--lambda_l3_weight", type=float, default=0.7)
    parser.add_argument('--epoch_boost', type=int, default=0)

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument("--predict", type=int, default=1, choices=[0,1])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_classes_fallback', type=int, default=6)

    args = parser.parse_args()

    # Update args based on the dataset name inferred from test_path
    if "/A/" in args.test_path:
        dataset_name = "A"
    elif "/B/" in args.test_path:
        dataset_name = "B"
    elif "/C/" in args.test_path:
        dataset_name = "C"
    elif "/D/" in args.test_path:
        dataset_name = "D"
    else:
        raise ValueError("Dataset name could not be determined from test_path. Please check the path format. Expected formats include '/A/', '/B/', '/C/', or '/D/'")

    if dataset_name == "C" or dataset_name== "D":
        args.gnn_type = 'transformer'
        args.no_residual = False
        args.jk_mode = "last"
        args.graph_pooling = "attention"

        if dataset_name== "C":
            args.num_layer = 3
            args.emb_dim = 204
            args.drop_ratio = 0.1
            args.transformer_heads = 4

        elif dataset_name== "D":
            args.num_layer = 3
            args.emb_dim = 153
            args.drop_ratio = 0.1
            args.transformer_heads = 3

    elif dataset_name == "A" or dataset_name == "B":
        args.gnn_type = 'GINEtrans'
        args.emb_dim = 128
        args.drop_ratio = 0.2

    main(args)