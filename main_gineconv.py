import argparse
import logging
import os
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader # Corretto
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import time

from source.evaluation import evaluate_model
from source.statistics import save_predictions, plot_training_progress
from source.train import train_epoch
from source.loadData import GraphDataset
from source.loss import gcodLoss, LabelSmoothingCrossEntropy
from source.models import GNN, GINENet # Assicurati che GINENet sia qui
from source.dataLoader import AddNodeFeatures # Per creare feature dei nodi
from source.utils import set_seed

# set_seed() # Meglio chiamarlo all'inizio del main o dopo aver parsato args.seed

def calculate_global_train_accuracy(model, full_train_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch in tqdm(full_train_loader, desc="Calculating global train accuracy (atrain)", unit="batch", leave=False, disable=True):
            graphs = data_batch.to(device)
            if graphs.y is None: continue # Salta se non ci sono etichette
            labels_int = graphs.y.to(device)
            outputs_logits = model(graphs)
            if outputs_logits.shape[0] == 0: continue # Se il modello restituisce output vuoto (es. grafo vuoto)
            _, predicted = torch.max(outputs_logits.data, 1)
            total += labels_int.size(0)
            correct += (predicted == labels_int.squeeze()).sum().item()
    if total == 0: return 0.0
    return correct / total

def main(args, full_train_dataset_outer=None, train_loader_outer=None, val_loader_outer=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")

    set_seed(args.seed) # Chiamato qui con l'argomento del parser

    test_dir_name = os.path.basename(os.path.dirname(args.test_path)) if args.test_path else "unknown_test_set"
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, f"training_{args.gnn_type}.log") # Log specifico per GNN type
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', # Aggiunto levelname
        force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler()) # Mostra log anche in console
    logging.info("\n" + "="*60 + f"\n>>> Starting main script for GNN type: {args.gnn_type}\n" + "="*60 + "\n")
    logging.info(f"• Device     : {device}")
    logging.info(f"• Input args : {args}\n")

    # --- Preparazione Dataset e Inferenza Parametri ---
    node_feat_dim_initial = 1 # Default per AddNodeFeatures (es. vettore di uni)
    if hasattr(args, 'node_feat_dim_initial') and args.node_feat_dim_initial is not None:
        node_feat_dim_initial = args.node_feat_dim_initial # Permetti override da args
    logging.info(f"• Initial node feature dimension (for AddNodeFeatures): {node_feat_dim_initial}")
    node_feature_creator = AddNodeFeatures(node_feature_dim=node_feat_dim_initial)

    num_dataset_classes = None
    num_edge_features_resolved = None
    # num_node_features_initial è ora node_feat_dim_initial, non serve inferirlo se lo fissiamo

    full_train_dataset = full_train_dataset_outer
    if args.train_path:
        if full_train_dataset is None:
            logging.info(f">>> Loading training dataset from: {args.train_path}")
            full_train_dataset = GraphDataset(args.train_path, add_zeros_transform=node_feature_creator) # add_zeros_transform rinominato per coerenza

        if len(full_train_dataset) > 0:
            num_dataset_classes = full_train_dataset.num_classes
            num_edge_features_resolved = full_train_dataset.num_edge_features
            # Verifica la dimensione effettiva delle feature dei nodi dopo la trasformazione
            # Questo conferma che AddNodeFeatures ha funzionato come previsto
            if hasattr(full_train_dataset[0], 'x') and full_train_dataset[0].x is not None:
                actual_node_feat_dim = full_train_dataset[0].x.size(1)
                if actual_node_feat_dim != node_feat_dim_initial:
                    logging.warning(f"Mismatch: AddNodeFeatures intended dim {node_feat_dim_initial}, but got {actual_node_feat_dim}. Using actual.")
                    # Potresti voler aggiornare node_feat_dim_initial qui, ma GINENet usa in_channels
                    # che sarà impostato a node_feat_dim_initial. È cruciale che AddNodeFeatures
                    # rispetti node_feat_dim_initial.
            else:
                logging.error("Node features 'x' not found in the first training sample after AddNodeFeatures. Check GraphDataset and AddNodeFeatures.")
                return

            logging.info(">>> Parameters inferred/set for training dataset:")
            logging.info(f"  Num node features (input to GINENet): {node_feat_dim_initial}")
            logging.info(f"  Num edge features                   : {num_edge_features_resolved}")
            logging.info(f"  Num classes                         : {num_dataset_classes}")
        else:
            logging.error("Training dataset is empty. Cannot infer parameters.")
            return
    else:
        logging.warning(">>> No training path provided. Using parameters from args or defaults.")
        num_dataset_classes = args.num_classes_fallback
        num_edge_features_resolved = args.num_edge_features # Prende da args
        # node_feat_dim_initial è già impostato
        logging.info(">>> Parameters from args/defaults:")
        logging.info(f"  Num node features (input to GINENet): {node_feat_dim_initial}")
        logging.info(f"  Num edge features                   : {num_edge_features_resolved}")
        logging.info(f"  Num classes                         : {num_dataset_classes}")

    if num_dataset_classes is None or num_dataset_classes <=0 : # Aggiunto <=0
        logging.error(f"Number of classes ({num_dataset_classes}) could not be determined or is invalid. Exiting.")
        return
    if num_edge_features_resolved is None: # Può essere 0, ma non None
        logging.warning(f"Number of edge features could not be determined. Assuming 0.")
        num_edge_features_resolved = 0


    # --- Costruzione Modello ---
    logging.info(">>> Building the model...")
    model = None
    if args.gnn_type == 'gine':
        model = GINENet(
            in_channels=node_feat_dim_initial, # Dimensione delle feature create da AddNodeFeatures
            hidden_channels=args.emb_dim,
            out_channels=num_dataset_classes,
            num_layers=args.num_layer,
            edge_dim=num_edge_features_resolved, # num_edge_features inferito
            dropout_rate=args.drop_ratio,
            graph_pooling=args.graph_pooling
        ).to(device)
        logging.info("• GINENet Model Parameters:")
        logging.info(f"  Input node channels     : {node_feat_dim_initial}")
        logging.info(f"  Hidden channels (emb_dim): {args.emb_dim}")
        logging.info(f"  Output channels (classes): {num_dataset_classes}")
        logging.info(f"  Number of GIN layers    : {args.num_layer}")
        logging.info(f"  Edge feature dimension  : {num_edge_features_resolved}")
        logging.info(f"  Dropout rate            : {args.drop_ratio}")
        logging.info(f"  Graph pooling           : {args.graph_pooling}")

    elif args.gnn_type == 'transformer': # Mantenuto se vuoi ancora questa opzione
        model = GNN( # Assicurati che la classe GNN sia definita e funzionante
            gnn_type=args.gnn_type,
            num_class=num_dataset_classes,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim, # Questo potrebbe essere diverso dall'in_channels del primo layer
            drop_ratio=args.drop_ratio,
            JK=args.jk_mode,
            graph_pooling=args.graph_pooling, # Potrebbe non essere lo stesso di GINENet
            num_edge_features=num_edge_features_resolved,
            transformer_heads=args.transformer_heads,
            residual=not args.no_residual # Per GNN generico
        ).to(device)
        # Logging specifico per il modello Transformer
        logging.info("• Transformer GNN Model Parameters (assicurati che GNN sia implementato):")
        logging.info(f"  Num layers        : {args.num_layer}")
        logging.info(f"  Embedding dimension : {args.emb_dim}")
        # ... altri log specifici per GNN/Transformer
    else:
        raise ValueError(f"Unsupported GNN type: {args.gnn_type}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"• Total trainable parameters: {total_params:,}\n")

    optimizer_model = torch.optim.AdamW(model.parameters(), lr=args.lr_model, weight_decay=args.weight_decay)

    checkpoint_path_best = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_{args.gnn_type}_best.pth")
    checkpoints_folder_epochs = os.path.join(script_dir, "checkpoints", test_dir_name, args.gnn_type)
    os.makedirs(checkpoints_folder_epochs, exist_ok=True) # Crea anche la dir base "checkpoints"
    os.makedirs(os.path.dirname(checkpoint_path_best), exist_ok=True)


    if os.path.exists(checkpoint_path_best) and not args.train_path:
        logging.info(f"Loading pre-trained model from {checkpoint_path_best}")
        try:
            model.load_state_dict(torch.load(checkpoint_path_best, map_location=device))
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model checkpoint: {e}. Proceeding without pre-trained weights.")


    if args.num_checkpoints is not None and args.num_checkpoints > 0:
        if args.num_checkpoints == 1: checkpoint_intervals = [args.epochs]
        else: checkpoint_intervals = [int((i + 1) * args.epochs / args.num_checkpoints) for i in range(args.num_checkpoints)]
    else: checkpoint_intervals = []

    train_loader = train_loader_outer
    val_loader = val_loader_outer

    # --- Training ---
    if args.train_path:
        if full_train_dataset is None:
            logging.error("full_train_dataset is None, though args.train_path was provided. This is an issue.")
            return

        logging.info(">>> Preparing train and validation datasets...")
        # Estrai etichette per lo split stratificato
        labels_for_split = []
        indices_with_valid_labels = []
        for i, data in enumerate(full_train_dataset):
            if data.y is not None:
                try:
                    label_item = data.y.item() if isinstance(data.y, torch.Tensor) else data.y
                    if isinstance(label_item, (int, float, np.integer)): # Accetta tipi numerici numpy
                        labels_for_split.append(label_item)
                        indices_with_valid_labels.append(i)
                    else:
                        logging.warning(f"Graph at index {i} has non-numeric label {data.y} of type {type(data.y)}. Excluding from stratified split if problematic.")
                except Exception as e:
                    logging.warning(f"Could not extract label for graph at index {i} (y={data.y}): {e}. Excluding from stratified split if problematic.")
            else:
                logging.warning(f"Graph at index {i} has y=None. Excluding from stratified split.")

        # Filtra il dataset per usare solo campioni con etichette valide per lo split
        # Questo è importante se alcuni grafi non hanno etichette (es. test set mascherato in un train set)
        dataset_for_split = torch.utils.data.Subset(full_train_dataset, indices_with_valid_labels)
        # labels_for_split ora corrisponde a dataset_for_split

        train_indices_subset, val_indices_subset = [], [] # Indici relativi a dataset_for_split
        if args.val_split_ratio > 0 and len(dataset_for_split) > 0:
            if len(set(labels_for_split)) < 2 or len(labels_for_split) < 2 : # Poche classi o campioni per stratificare
                logging.warning("Not enough unique labels or samples for stratified split. Using non-stratified split on samples with labels.")
                train_indices_subset, val_indices_subset = train_test_split(
                    range(len(dataset_for_split)), # Indici relativi a dataset_for_split
                    test_size=args.val_split_ratio,
                    shuffle=True,
                    random_state=args.seed
                )
            else:
                try:
                    train_indices_subset, val_indices_subset = train_test_split(
                        range(len(dataset_for_split)), # Indici relativi a dataset_for_split
                        test_size=args.val_split_ratio,
                        shuffle=True,
                        stratify=labels_for_split,
                        random_state=args.seed
                    )
                except ValueError as e:
                    logging.warning(f"Stratified split failed: {e}. Falling back to non-stratified split.")
                    train_indices_subset, val_indices_subset = train_test_split(
                        range(len(dataset_for_split)),
                        test_size=args.val_split_ratio,
                        shuffle=True,
                        random_state=args.seed
                    )
        elif len(dataset_for_split) > 0: # No validation split, usa tutto dataset_for_split per training
            train_indices_subset = list(range(len(dataset_for_split)))
        else:
            logging.error("No samples with valid labels available for training/validation split.")
            return


        # Mappa gli indici del subset (train_indices_subset, val_indices_subset)
        # agli indici originali di full_train_dataset (indices_with_valid_labels)
        original_train_indices = [indices_with_valid_labels[i] for i in train_indices_subset]
        original_val_indices = [indices_with_valid_labels[i] for i in val_indices_subset]

        train_dataset_actual = torch.utils.data.Subset(full_train_dataset, original_train_indices)
        val_dataset_actual = torch.utils.data.Subset(full_train_dataset, original_val_indices) if original_val_indices else None


        logging.info("\n" + "-"*60 + "\n>> Dataset: split in training/validation\n" + "-"*60 + "\n")
        logging.info(f"• Full dataset size (original)  : {len(full_train_dataset)}")
        logging.info(f"• Samples w/ valid labels       : {len(dataset_for_split)}")
        logging.info(f"• Training set (actual)         : {len(train_dataset_actual)} campioni")
        if val_dataset_actual:
            logging.info(f"• Validation set (actual)       : {len(val_dataset_actual)} campioni\n")
        else:
            logging.info(f"• Validation set (actual)       : 0 campioni (val_split_ratio might be 0 or no valid labels for val)\n")


        if train_loader is None and len(train_dataset_actual) > 0:
            train_loader = DataLoader(train_dataset_actual, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        elif len(train_dataset_actual) == 0:
            logging.error("Training dataset is empty after split. Cannot proceed.")
            return

        if val_loader is None and val_dataset_actual and len(val_dataset_actual) > 0:
            val_loader = DataLoader(val_dataset_actual, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        elif val_loader is None : # val_dataset_actual è None o vuoto
            val_loader = None


        full_train_loader_for_atrain = None
        if args.criterion == "gcod" and len(train_dataset_actual) > 0:
            logging.info(">>> Preparing current training subset loader for atrain calculation (used by GCOD)...")
            # Usa train_dataset_actual per atrain, che è il set di training effettivo
            full_train_loader_for_atrain = DataLoader(train_dataset_actual, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


        criterion_obj = None
        optimizer_loss_params = None
        criterion_obj_ce = None

        if args.criterion == "ce":
            criterion_obj = LabelSmoothingCrossEntropy(classes=num_dataset_classes, smoothing=args.label_smoothing).to(device)
        elif args.criterion == "gcod":
            # Per GCOD, le etichette dovrebbero essere quelle dell'INTERO dataset di training originale
            # su cui GCOD deve operare, non solo il subset di training corrente dopo lo split.
            # La classe GraphDataset memorizza `original_idx`.
            # gcodLoss internamente mapperà questi indici.
            y_values_numpy_for_gcod = np.array([
                (data.y.item() if isinstance(data.y, torch.Tensor) else data.y)
                for data in full_train_dataset # USA L'INTERO full_train_dataset
                if data.y is not None # Solo campioni con etichetta
            ])
            # Conta solo i campioni con etichetta per num_examp
            num_examp_for_gcod = len(y_values_numpy_for_gcod)

            logging.info("\n" + "-"*60 + "\n>> Initializing Loss Function (GCOD)\n" + "-"*60 + "\n")
            logging.info(f"• Num classes           : {num_dataset_classes}")
            logging.info(f"• Embeddings size       : {args.emb_dim}")
            logging.info(f"• Total labeled samples : {num_examp_for_gcod} (for GCOD matrix)")
            logging.info(f"• Total epochs          : {args.epochs}")

            criterion_obj = gcodLoss(
                sample_labels_numpy=y_values_numpy_for_gcod,
                device=device,
                num_examp=num_examp_for_gcod,
                num_classes=num_dataset_classes,
                gnn_embedding_dim=args.emb_dim, # GINENet usa hidden_channels che è args.emb_dim
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

            if args.criterion == "gcod" and full_train_loader_for_atrain is not None:
                atrain_global = calculate_global_train_accuracy(model, full_train_loader_for_atrain, device)

            current_criterion_obj = criterion_obj
            current_criterion_type = args.criterion
            if args.criterion == "gcod" and args.epoch_boost > 0 and epoch < args.epoch_boost:
                current_criterion_obj = criterion_obj_ce
                current_criterion_type = "ce"
                logging.info(f"Epoch {epoch+1}: Using CE for boost.")


            current_optimizer_loss_params = optimizer_loss_params
            if current_criterion_type == "ce":
                current_optimizer_loss_params = None

            avg_train_loss, avg_train_accuracy, train_f1 = train_epoch(
                model=model, loader=train_loader, optimizer_model=optimizer_model, device=device,
                criterion_obj=current_criterion_obj, criterion_type=current_criterion_type,
                optimizer_loss_params=current_optimizer_loss_params, num_classes_dataset=num_dataset_classes,
                lambda_l3_weight=args.lambda_l3_weight, current_epoch=epoch,
                atrain_global_value=atrain_global,
                save_checkpoints=(epoch + 1 in checkpoint_intervals), checkpoint_path=os.path.join(checkpoints_folder_epochs, f"model_epoch_{epoch+1}"),
                epoch_boost=args.epoch_boost,
                gradient_clipping_norm=args.gradient_clipping
            )
            train_losses_history.append(avg_train_loss)
            train_f1_history.append(train_f1)
            train_accuracy_history.append(avg_train_accuracy)

            avg_val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
            if val_loader: # Solo se esiste un validation loader
                avg_val_loss, val_acc, val_f1 = evaluate_model( # Assumo restituisca (loss, acc, f1, preds, labels)
                    model=model, loader=val_loader, device=device,
                    criterion_obj=current_criterion_obj,
                    criterion_type=current_criterion_type,
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
                if os.path.exists(os.path.dirname(checkpoint_path_best)): # Controlla se la dir esiste
                    torch.save(model.state_dict(), checkpoint_path_best)
                    logging.info(f"Best model updated and saved to {checkpoint_path_best} (Val F1: {best_val_metric*100:.2f}%)")
                else:
                    logging.warning(f"Checkpoint directory {os.path.dirname(checkpoint_path_best)} does not exist. Cannot save best model.")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                # logging.info(f"No improvement for {epochs_no_improve} epoch(s). Best Val F1: {best_val_metric*100:.2f}%") # Meno verboso

            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                logging.info(f"Early stopping triggered after {args.early_stopping_patience} epochs with no improvement.")
                break

        total_training_time = time.time() - start_time_train
        logging.info("\n" + "="*60 + f"\n>>> Training completed in {total_training_time:.2f} seconds\n" + "="*60 + "\n")
        if os.path.exists(logs_folder):
            plot_training_progress({"train_loss": train_losses_history, "val_loss": val_losses_history},
                                   {"train_acc": train_accuracy_history, "train_f1": train_f1_history, "val_acc": val_accuracy_history, "val_f1": val_f1_history},
                                   os.path.join(logs_folder, f"training_plots_{args.gnn_type}")) # Nome file specifico per GNN
        else:
            logging.warning(f"Logs folder {logs_folder} does not exist. Cannot save training plots.")


    # --- Predizione sul Test Set ---
    if args.predict == 1:
        if not args.test_path or not os.path.exists(args.test_path):
            logging.error(f"Test path {args.test_path} does not exist or not specified.")
            return

        checkpoint_to_load_path = checkpoint_path_best
        if not os.path.exists(checkpoint_to_load_path):
            logging.error(f"No model checkpoint found at {checkpoint_to_load_path} for GNN type {args.gnn_type}. Cannot make predictions.")
            # Prova a caricare l'ultimo checkpoint di epoca se il "best" non c'è e ci sono checkpoint salvati
            if checkpoint_intervals and os.path.exists(os.path.join(checkpoints_folder_epochs, f"model_epoch_{args.epochs}.pth")):
                checkpoint_to_load_path = os.path.join(checkpoints_folder_epochs, f"model_epoch_{args.epochs}.pth")
                logging.warning(f"Best model checkpoint not found. Trying last epoch checkpoint: {checkpoint_to_load_path}")
            else:
                return


        logging.info(f">>> Loading model from {checkpoint_to_load_path} for prediction.")
        try:
            model.load_state_dict(torch.load(checkpoint_to_load_path, map_location=device))
            logging.info("Model loaded successfully for prediction.")
        except Exception as e:
            logging.error(f"Error loading model for prediction: {e}. Cannot make predictions.")
            return


        logging.info(">>> Preparing test dataset...")
        test_dataset = GraphDataset(args.test_path, add_zeros_transform=node_feature_creator) # Usa lo stesso creator
        if len(test_dataset) == 0:
            logging.error("Test dataset is empty. Cannot make predictions.")
            return
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        logging.info(">>> Generating predictions for the test set...")
        # Assumiamo che evaluate_model restituisca (avg_loss, accuracy, f1_score, all_preds, all_labels)
        # e che per il test (is_validation=False) non calcoli loss se criterion_obj=None
        test_output_tuple = evaluate_model(
            model=model, loader=test_loader, device=device,
            criterion_obj=None, # Non calcolare loss
            criterion_type="ce", # Irrilevante se criterion_obj è None
            is_validation=False, # Modalità test
            num_classes_dataset=num_dataset_classes # Necessario per metriche se calcolate
        )

        test_predictions = None
        if isinstance(test_output_tuple, tuple) and len(test_output_tuple) >= 4:
            test_predictions = test_output_tuple[3] # Assumendo che le predizioni siano il 4° elemento
            test_true_labels = test_output_tuple[4] # E le etichette vere il 5°
            # Qui potresti calcolare e loggare le metriche del test set se hai le etichette vere
            test_acc = test_output_tuple[1]
            test_f1 = test_output_tuple[2]
            logging.info(f"Test Set Metrics: Accuracy: {test_acc*100:.2f}%, F1-score: {test_f1*100:.2f}%")
        else:
            logging.error("evaluate_model did not return the expected tuple with predictions for the test set.")
            return # Non possiamo salvare le predizioni se non le abbiamo

        if test_predictions is not None:
            submission_folder = os.path.join(script_dir, "submissions")
            os.makedirs(submission_folder, exist_ok=True)
            submission_filename = f"predictions_{test_dir_name}_{args.gnn_type}.txt"
            submission_path = os.path.join(submission_folder, submission_filename)

            save_predictions(test_predictions, submission_path)
            logging.info(f"Predictions saved to {submission_path}")
        else:
            logging.warning("No predictions were generated for the test set.")


    logging.info("Main script finished <<<")
    return full_train_dataset, train_loader, val_loader


TEST_PATH = "../datasets/B/test.json.gz" # Esempio
TRAIN_PATH = "../datasets/B/train.json.gz" # Esempio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")

    parser.add_argument("--train_path", type=str, default=TRAIN_PATH, help="Path to the training dataset.")
    parser.add_argument("--test_path", type=str, default=TEST_PATH, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, default=1, help="Number of checkpoints during training. 0 to disable. 1 saves only at the end.") # Modificato default

    parser.add_argument('--gnn_type', type=str, default='gine', choices=['transformer','gine'], help='GNN architecture.')
    parser.add_argument('--num_layer', type=int, default=3, help='Number of GNN layers.')
    parser.add_argument('--emb_dim', type=int, default=128, help='Dimensionality of hidden GNN units (hidden_channels for GINENet).')
    parser.add_argument('--drop_ratio', type=float, default=0.3, help='Dropout ratio.') # Aumentato un po'
    parser.add_argument('--graph_pooling', type=str, default="mean", choices=["sum", "mean", "max", "add"], help="Graph pooling method (for GINENet or other GNNs).")

    # Argomenti specifici per Transformer (se gnn_type == 'transformer')
    parser.add_argument('--transformer_heads', type=int, default=4, help='Num heads for TransformerConv.')
    parser.add_argument('--jk_mode', type=str, default="last", help="Jumping Knowledge mode for Transformer GNN.")
    parser.add_argument('--no_residual', action='store_true', help='Disable residual connections for Transformer GNN.')

    # Argomento per la dimensione iniziale delle feature dei nodi se AddNodeFeatures lo richiede
    parser.add_argument('--node_feat_dim_initial', type=int, default=1, help='Initial dimension for node features created by AddNodeFeatures.')
    parser.add_argument('--num_edge_features', type=int, default=None, help='Dimensionality of edge features (used if not inferred, e.g. no train_path).') # Default a None

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr_model', type=float, default=1e-3, help='Learning rate for GNN model.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for AdamW.')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Max norm for gradient clipping (0 to disable).')
    parser.add_argument('--val_split_ratio', type=float, default=0.15, help='Fraction for validation split.')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping. 0 to disable.')

    parser.add_argument("--criterion", type=str, default="ce", choices=["ce", "gcod"], help="Loss function.")
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing for CE loss (0 for no smoothing).') # Default a 0
    parser.add_argument("--lr_u", type=float, default=0.01, help="Learning rate for 'u' in GCOD.")
    parser.add_argument("--lambda_l3_weight", type=float, default=0.7, help="Weight for L3 in GCOD.")
    parser.add_argument('--epoch_boost', type=int, default=0, help='Initial epochs with CE before GCOD (0 to disable boost).')

    parser.add_argument('--device', type=int, default=0, help='GPU device ID (-1 for CPU).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.') # Cambiato seed per varietà
    parser.add_argument("--predict", type=int, default=1, choices=[0,1], help="Generate predictions (1) or not (0).")
    parser.add_argument('--num_workers', type=int, default=0, help='Dataloader workers (set >0 for multiprocessing, e.g., 2 or 4).')

    parser.add_argument('--num_classes_fallback', type=int, default=2, help='Fallback for num_classes if not inferred.') # Default più generico

    args = parser.parse_args()

    # Validazione di alcuni argomenti
    if args.gnn_type == 'gine' and (args.emb_dim <= 0 or args.num_layer <=0):
        logging.error("For GINENet, emb_dim and num_layer must be positive.")
        exit()
    if args.train_path is None and args.num_edge_features is None:
        logging.warning("No train_path and num_edge_features not set. Assuming 0 edge features if needed.")
        args.num_edge_features = 0 # Fallback se non specificato e non inferibile

    main(args)