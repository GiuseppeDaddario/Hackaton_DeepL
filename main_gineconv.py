import argparse
import logging
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import time

from source.evaluation import evaluate_model
from source.statistics import save_predictions, plot_training_progress
from source.train import train_epoch
from source.loadData import GraphDataset
from source.loss import gcodLoss, LabelSmoothingCrossEntropy
from source.models import GNN, GINENet
from source.dataLoader import AddNodeFeatures
from source.utils import set_seed

def calculate_global_train_accuracy(model, full_train_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch in tqdm(full_train_loader, desc="Calculating global train accuracy (atrain)", unit="batch", leave=False, disable=True):
            graphs = data_batch.to(device)
            if graphs.y is None: continue
            labels_int = graphs.y.to(device)

            # Adatta in base a cosa restituisce il tuo modello
            if hasattr(model, 'forward_for_gcod_metrics') and callable(getattr(model, 'forward_for_gcod_metrics')):
                outputs_logits, _, _ = model.forward_for_gcod_metrics(graphs) # Se hai un metodo separato
            else: # Altrimenti, assumi che il forward standard restituisca i logits (o un tuple)
                model_output = model(graphs)
                if isinstance(model_output, tuple):
                    outputs_logits = model_output[0] # Se restituisce (logits, embeddings, ...)
                else:
                    outputs_logits = model_output # Se restituisce solo logits

            if outputs_logits.shape[0] == 0: continue
            _, predicted = torch.max(outputs_logits.data, 1)
            total += labels_int.size(0)
            correct += (predicted == labels_int.squeeze()).sum().item()
    if total == 0: return 0.0
    return correct / total

def main(args, full_train_dataset_outer=None, train_loader_outer=None, val_loader_outer=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    set_seed(args.seed)

    test_dir_name = os.path.basename(os.path.dirname(args.test_path)) if args.test_path else "unknown_test_set"
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, f"training_{args.gnn_type}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("\n" + "="*60 + f"\n>>> Starting main script for GNN type: {args.gnn_type}\n" + "="*60 + "\n")
    logging.info(f"• Device     : {device}")
    logging.info(f"• Input args : {args}\n")

    node_feat_dim_initial = args.node_feat_dim_initial
    logging.info(f"• Initial node feature dimension (for AddNodeFeatures): {node_feat_dim_initial}")
    node_feature_creator = AddNodeFeatures(node_feature_dim=node_feat_dim_initial)

    num_dataset_classes = None
    num_edge_features_resolved = None

    full_train_dataset = full_train_dataset_outer
    if args.train_path:
        if full_train_dataset is None:
            logging.info(f">>> Loading training dataset from: {args.train_path}")
            full_train_dataset = GraphDataset(args.train_path, add_zeros_transform=node_feature_creator)

        if len(full_train_dataset) > 0:
            num_dataset_classes = full_train_dataset.num_classes
            num_edge_features_resolved = full_train_dataset.num_edge_features
            if hasattr(full_train_dataset[0], 'x') and full_train_dataset[0].x is not None:
                actual_node_feat_dim = full_train_dataset[0].x.size(1)
                if actual_node_feat_dim != node_feat_dim_initial:
                    logging.warning(f"Mismatch: AddNodeFeatures intended dim {node_feat_dim_initial}, but got {actual_node_feat_dim}.")
            else:
                logging.error("Node features 'x' not found in the first training sample after AddNodeFeatures.")
                return
            logging.info(">>> Parameters inferred/set for training dataset:")
            logging.info(f"  Num node features (input to GNN): {node_feat_dim_initial}")
            logging.info(f"  Num edge features                : {num_edge_features_resolved}")
            logging.info(f"  Num classes                      : {num_dataset_classes}")
        else:
            logging.error("Training dataset is empty. Cannot infer parameters.")
            return
    else:
        logging.warning(">>> No training path provided. Using parameters from args or defaults.")
        num_dataset_classes = args.num_classes_fallback
        num_edge_features_resolved = args.num_edge_features
        logging.info(">>> Parameters from args/defaults:")
        logging.info(f"  Num node features (input to GNN): {node_feat_dim_initial}")
        logging.info(f"  Num edge features                : {num_edge_features_resolved}")
        logging.info(f"  Num classes                      : {num_dataset_classes}")

    if num_dataset_classes is None or num_dataset_classes <= 0:
        logging.error(f"Number of classes ({num_dataset_classes}) is invalid. Exiting.")
        return
    if num_edge_features_resolved is None:
        logging.warning(f"Number of edge features undetermined. Assuming 0.")
        num_edge_features_resolved = 0

    logging.info(">>> Building the model...")
    model = None
    if args.gnn_type == 'gine':
        model = GINENet(in_channels=node_feat_dim_initial, hidden_channels=args.emb_dim, out_channels=num_dataset_classes,
                        num_layers=args.num_layer, edge_dim=num_edge_features_resolved, dropout_rate=args.drop_ratio,
                        graph_pooling=args.graph_pooling).to(device)
        logging.info("• GINENet Model Parameters:")
        logging.info(f"  Input node channels     : {node_feat_dim_initial}")
        logging.info(f"  Hidden channels (emb_dim): {args.emb_dim}")
        logging.info(f"  Output channels (classes): {num_dataset_classes}")
        logging.info(f"  Number of GIN layers    : {args.num_layer}")
        logging.info(f"  Edge feature dimension  : {num_edge_features_resolved}")
        logging.info(f"  Dropout rate            : {args.drop_ratio}")
        logging.info(f"  Graph pooling           : {args.graph_pooling}")
    elif args.gnn_type == 'transformer':
        model = GNN(gnn_type=args.gnn_type, num_class=num_dataset_classes, num_layer=args.num_layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, JK=args.jk_mode, graph_pooling=args.graph_pooling,
                    num_edge_features=num_edge_features_resolved, transformer_heads=args.transformer_heads,
                    residual=not args.no_residual).to(device)
        logging.info("• Transformer GNN Model Parameters:")
    else:
        raise ValueError(f"Unsupported GNN type: {args.gnn_type}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"• Total trainable parameters: {total_params:,}\n")

    optimizer_model = optim.AdamW(model.parameters(), lr=args.lr_model, weight_decay=args.weight_decay)

    lr_scheduler_model = ReduceLROnPlateau(optimizer_model, mode='max', factor=args.lr_scheduler_factor,
                                           patience=args.lr_scheduler_patience, verbose=True, min_lr=args.lr_scheduler_min_lr)

    checkpoint_path_best = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_{args.gnn_type}_best.pth")
    checkpoints_folder_epochs = os.path.join(script_dir, "checkpoints", test_dir_name, args.gnn_type)
    os.makedirs(checkpoints_folder_epochs, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path_best), exist_ok=True)

    if os.path.exists(checkpoint_path_best) and not args.train_path:
        logging.info(f"Loading pre-trained model from {checkpoint_path_best}")
        try:
            model.load_state_dict(torch.load(checkpoint_path_best, map_location=device))
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model checkpoint: {e}.")

    if args.num_checkpoints is not None and args.num_checkpoints > 0:
        if args.num_checkpoints == 1: checkpoint_intervals = [args.epochs]
        else: checkpoint_intervals = [int((i + 1) * args.epochs / args.num_checkpoints) for i in range(args.num_checkpoints)]
    else: checkpoint_intervals = []

    train_loader = train_loader_outer
    val_loader = val_loader_outer

    if args.train_path:
        if full_train_dataset is None:
            logging.error("full_train_dataset is None.")
            return

        logging.info(">>> Preparing train and validation datasets...")
        labels_for_split = []
        indices_with_valid_labels = []
        for i, data in enumerate(full_train_dataset):
            if data.y is not None:
                try:
                    label_item = data.y.item() if isinstance(data.y, torch.Tensor) else data.y
                    if isinstance(label_item, (int, float, np.integer)):
                        labels_for_split.append(label_item)
                        indices_with_valid_labels.append(i)
                except Exception: pass

        dataset_for_split = torch.utils.data.Subset(full_train_dataset, indices_with_valid_labels)
        train_indices_subset, val_indices_subset = [], []
        if args.val_split_ratio > 0 and len(dataset_for_split) > 0:
            if len(set(labels_for_split)) < 2 or len(labels_for_split) < 2:
                logging.warning("Not enough unique labels/samples for stratified split. Using non-stratified split.")
                train_indices_subset, val_indices_subset = train_test_split(range(len(dataset_for_split)), test_size=args.val_split_ratio, shuffle=True, random_state=args.seed)
            else:
                try:
                    train_indices_subset, val_indices_subset = train_test_split(range(len(dataset_for_split)), test_size=args.val_split_ratio, shuffle=True, stratify=labels_for_split, random_state=args.seed)
                except ValueError as e:
                    logging.warning(f"Stratified split failed: {e}. Falling back to non-stratified split.")
                    train_indices_subset, val_indices_subset = train_test_split(range(len(dataset_for_split)), test_size=args.val_split_ratio, shuffle=True, random_state=args.seed)
        elif len(dataset_for_split) > 0:
            train_indices_subset = list(range(len(dataset_for_split)))
        else:
            logging.error("No samples with valid labels for training/validation split.")
            return

        original_train_indices = [indices_with_valid_labels[i] for i in train_indices_subset]
        original_val_indices = [indices_with_valid_labels[i] for i in val_indices_subset]
        train_dataset_actual = torch.utils.data.Subset(full_train_dataset, original_train_indices)
        val_dataset_actual = torch.utils.data.Subset(full_train_dataset, original_val_indices) if original_val_indices else None

        logging.info("\n" + "-"*60 + "\n>> Dataset: split in training/validation\n" + "-"*60 + "\n")
        logging.info(f"• Full dataset size (original)  : {len(full_train_dataset)}")
        logging.info(f"• Samples w/ valid labels       : {len(dataset_for_split)}")
        logging.info(f"• Training set (actual)         : {len(train_dataset_actual)} campioni")
        if val_dataset_actual: logging.info(f"• Validation set (actual)       : {len(val_dataset_actual)} campioni\n")
        else: logging.info(f"• Validation set (actual)       : 0 campioni\n")

        if train_loader is None and len(train_dataset_actual) > 0:
            train_loader = DataLoader(train_dataset_actual, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        elif len(train_dataset_actual) == 0:
            logging.error("Training dataset empty after split.")
            return
        if val_loader is None and val_dataset_actual and len(val_dataset_actual) > 0:
            val_loader = DataLoader(val_dataset_actual, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        elif val_loader is None: val_loader = None

        full_train_loader_for_atrain = None
        if args.criterion == "gcod" and len(train_dataset_actual) > 0:
            logging.info(">>> Preparing current training subset loader for atrain calculation (GCOD)...")
            full_train_loader_for_atrain = DataLoader(train_dataset_actual, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        criterion_obj = None
        optimizer_loss_params = None
        criterion_obj_ce = None

        if args.criterion == "ce":
            criterion_obj = LabelSmoothingCrossEntropy(classes=num_dataset_classes, smoothing=args.label_smoothing).to(device)
        elif args.criterion == "gcod":
            y_values_numpy_for_gcod = np.array([(data.y.item() if isinstance(data.y, torch.Tensor) else data.y) for data in full_train_dataset if data.y is not None])
            num_examp_for_gcod = len(y_values_numpy_for_gcod)
            logging.info("\n" + "-"*60 + "\n>> Initializing Loss Function (GCOD)\n" + "-"*60 + "\n")
            criterion_obj = gcodLoss(sample_labels_numpy=y_values_numpy_for_gcod, device=device, num_examp=num_examp_for_gcod,
                                     num_classes=num_dataset_classes, gnn_embedding_dim=args.emb_dim, total_epochs=args.epochs).to(device)
            optimizer_loss_params = optim.SGD(criterion_obj.parameters(), lr=args.lr_u)
            if args.epoch_boost > 0:
                criterion_obj_ce = LabelSmoothingCrossEntropy(classes=num_dataset_classes, smoothing=args.label_smoothing).to(device)
        else: raise ValueError(f"Unsupported criterion: {args.criterion}")
        logging.info(">>> Loss function initialized.\n")

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

            current_criterion_obj = criterion_obj
            current_criterion_type = args.criterion
            if args.criterion == "gcod" and args.epoch_boost > 0 and epoch < args.epoch_boost:
                current_criterion_obj = criterion_obj_ce
                current_criterion_type = "ce_boost"

            current_optimizer_loss_params = optimizer_loss_params
            if current_criterion_type == "ce" or current_criterion_type == "ce_boost":
                current_optimizer_loss_params = None

            avg_train_loss, avg_train_accuracy, train_f1 = train_epoch(
                model=model, loader=train_loader, optimizer_model=optimizer_model, device=device,
                criterion_obj=current_criterion_obj, criterion_type=current_criterion_type,
                optimizer_loss_params=current_optimizer_loss_params, num_classes_dataset=num_dataset_classes,
                lambda_l3_weight=args.lambda_l3_weight, current_epoch=epoch, atrain_global_value=atrain_global,
                save_checkpoints=(epoch + 1 in checkpoint_intervals), checkpoint_path=os.path.join(checkpoints_folder_epochs, f"model_epoch_{epoch+1}"),
                epoch_boost=args.epoch_boost, gradient_clipping_norm=args.gradient_clipping)

            train_losses_history.append(avg_train_loss)
            train_f1_history.append(train_f1)
            train_accuracy_history.append(avg_train_accuracy)

            avg_val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
            if val_loader:
                eval_output = evaluate_model(
                    model=model, loader=val_loader, device=device, criterion_obj=current_criterion_obj,
                    criterion_type=current_criterion_type, num_classes_dataset=num_dataset_classes,
                    lambda_l3_weight=args.lambda_l3_weight, current_epoch_for_gcod=epoch,
                    atrain_for_gcod=atrain_global, is_validation=True)
                if isinstance(eval_output, tuple) and len(eval_output) >=3:
                    avg_val_loss, val_acc, val_f1 = eval_output[0], eval_output[1], eval_output[2]
                else: # Fallback o errore se evaluate_model non restituisce il formato atteso
                    logging.error(f"evaluate_model ha restituito un formato inatteso: {eval_output}")

            val_losses_history.append(avg_val_loss)
            val_f1_history.append(val_f1)
            val_accuracy_history.append(val_acc)

            epoch_duration = time.time() - epoch_start_time
            atrain_log_str = f"{atrain_global:.4f}" if args.criterion == 'gcod' and full_train_loader_for_atrain is not None else 'N/A'
            logging.info(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_accuracy*100:.2f}%, F1: {train_f1*100:.2f}% | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1*100:.2f}% | Time: {epoch_duration:.2f}s | a_train: {atrain_log_str}")

            if val_loader: lr_scheduler_model.step(val_f1)
            else: logging.warning("No validation loader. ReduceLROnPlateau scheduler not stepping.")
            current_lr_model = optimizer_model.param_groups[0]['lr']
            logging.info(f"Epoch {epoch+1}: Current model LR: {current_lr_model:.2e}")

            current_val_metric_for_best = val_f1 if val_loader else train_f1
            if current_val_metric_for_best > best_val_f1:
                best_val_f1 = current_val_metric_for_best
                if os.path.exists(os.path.dirname(checkpoint_path_best)):
                    torch.save(model.state_dict(), checkpoint_path_best)
                    logging.info(f"Best model updated (Val F1: {best_val_f1*100:.2f}%). Saved to {checkpoint_path_best}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                logging.info(f"Early stopping after {args.early_stopping_patience} epochs with no improvement.")
                break

        total_training_time = time.time() - start_time_train
        logging.info("\n" + "="*60 + f"\n>>> Training completed in {total_training_time:.2f} seconds\n" + "="*60 + "\n")
        if os.path.exists(logs_folder):
            plot_training_progress({"train_loss": train_losses_history, "val_loss": val_losses_history},
                                   {"train_acc": train_accuracy_history, "train_f1": train_f1_history, "val_acc": val_accuracy_history, "val_f1": val_f1_history},
                                   os.path.join(logs_folder, f"training_plots_{args.gnn_type}"))

    if args.predict == 1:
        if not args.test_path or not os.path.exists(args.test_path):
            logging.error(f"Test path {args.test_path} not found or not specified.")
            return
        checkpoint_to_load_path = checkpoint_path_best
        if not os.path.exists(checkpoint_to_load_path):
            logging.error(f"No best model checkpoint at {checkpoint_to_load_path}.")
            if checkpoint_intervals and os.path.exists(os.path.join(checkpoints_folder_epochs, f"model_epoch_{args.epochs}.pth")):
                checkpoint_to_load_path = os.path.join(checkpoints_folder_epochs, f"model_epoch_{args.epochs}.pth")
                logging.warning(f"Trying last epoch checkpoint: {checkpoint_to_load_path}")
            else: return

        logging.info(f">>> Loading model from {checkpoint_to_load_path} for prediction.")
        try:
            model.load_state_dict(torch.load(checkpoint_to_load_path, map_location=device))
            logging.info("Model loaded for prediction.")
        except Exception as e:
            logging.error(f"Error loading model for prediction: {e}.")
            return

        logging.info(">>> Preparing test dataset...")
        test_dataset = GraphDataset(args.test_path, add_zeros_transform=node_feature_creator)
        if len(test_dataset) == 0:
            logging.error("Test dataset empty.")
            return
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        logging.info(">>> Generating predictions for the test set...")
        test_output_tuple = evaluate_model(model=model, loader=test_loader, device=device, criterion_obj=None, criterion_type="ce",
                                           is_validation=False, num_classes_dataset=num_dataset_classes)

        test_predictions = None
        if isinstance(test_output_tuple, tuple) and len(test_output_tuple) >= 4:
            test_predictions = test_output_tuple[3]
            test_acc = test_output_tuple[1]
            test_f1 = test_output_tuple[2]
            logging.info(f"Test Set Metrics: Accuracy: {test_acc*100:.2f}%, F1-score: {test_f1*100:.2f}%")
        else:
            logging.error("evaluate_model did not return expected tuple for test set.")
            return

        if test_predictions is not None:
            submission_folder = os.path.join(script_dir, "submissions")
            os.makedirs(submission_folder, exist_ok=True)
            submission_filename = f"predictions_{test_dir_name}_{args.gnn_type}.txt"
            submission_path = os.path.join(submission_folder, submission_filename)
            save_predictions(test_predictions, submission_path)
            logging.info(f"Predictions saved to {submission_path}")
        else: logging.warning("No predictions generated for test set.")

    logging.info("Main script finished <<<")
    return full_train_dataset, train_loader, val_loader

TEST_PATH = "../datasets/B/test.json.gz"
TRAIN_PATH = "../datasets/B/train.json.gz"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, default=TRAIN_PATH)
    parser.add_argument("--test_path", type=str, default=TEST_PATH)
    parser.add_argument("--num_checkpoints", type=int, default=1)
    parser.add_argument('--gnn_type', type=str, default='gine', choices=['transformer','gine'])
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--drop_ratio', type=float, default=0.3)
    parser.add_argument('--graph_pooling', type=str, default="mean", choices=["sum", "mean", "max", "add"])
    parser.add_argument('--transformer_heads', type=int, default=4)
    parser.add_argument('--jk_mode', type=str, default="last")
    parser.add_argument('--no_residual', action='store_true')
    parser.add_argument('--node_feat_dim_initial', type=int, default=1)
    parser.add_argument('--num_edge_features', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_model', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    parser.add_argument('--val_split_ratio', type=float, default=0.15)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument("--criterion", type=str, default="ce", choices=["ce", "gcod"])
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument("--lr_u", type=float, default=0.01)
    parser.add_argument("--lambda_l3_weight", type=float, default=0.7)
    parser.add_argument('--epoch_boost', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--predict", type=int, default=1, choices=[0,1])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_classes_fallback', type=int, default=2)
    parser.add_argument('--lr_scheduler_metric', type=str, default='val_f1', choices=['val_loss', 'val_f1', 'val_acc'], help="Metric for LR scheduler (always val_f1 for max F1).")
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help="Factor for LR reduction.")
    parser.add_argument('--lr_scheduler_patience', type=int, default=10, help="Patience for LR reduction.")
    parser.add_argument('--lr_scheduler_min_lr', type=float, default=1e-6, help="Minimum learning rate.")

    args = parser.parse_args()

    if args.gnn_type == 'gine' and (args.emb_dim <= 0 or args.num_layer <=0):
        logging.error("For GINENet, emb_dim and num_layer must be positive.")
        exit()
    if args.train_path is None and args.num_edge_features is None:
        logging.warning("No train_path and num_edge_features not set. Assuming 0.")
        args.num_edge_features = 0

    args.lr_scheduler_metric = 'val_f1' # Forza il monitoraggio di val_f1

    main(args)