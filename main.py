import argparse
import logging
import os
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import time

from source.evaluation import evaluate_model
from source.statistics import save_predictions, plot_training_progress
from source.train import train_epoch
# from source.dataLoader import add_zeros # Rimosso, useremo AddNodeFeatures
from source.loadData import GraphDataset
from source.loss import gcodLoss, LabelSmoothingCrossEntropy
# from source.models import GNN # Rimosso se usi solo GINE, o modificato se GINENet è lì
from source.models import GNN,GINENet # Assumendo GINENet in source/model.py
from source.dataLoader import AddNodeFeatures # Per creare feature dei nodi
from source.utils import set_seed

set_seed()


def calculate_global_train_accuracy(model, full_train_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch in tqdm(full_train_loader, desc="Calculating global train accuracy (atrain)", unit="batch", leave=False, disable=True):
            graphs = data_batch.to(device)
            labels_int = graphs.y.to(device)
            outputs_logits = model(graphs) # Modificato: GINENet restituisce solo logits
            _, predicted = torch.max(outputs_logits.data, 1)
            total += labels_int.size(0)
            correct += (predicted == labels_int.squeeze()).sum().item()
    if total == 0: return 0.0
    return correct / total

def main(args, full_train_dataset_outer=None, train_loader_outer=None, val_loader_outer=None): # Rinominate per evitare shadowing
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

    # --- Preparazione Dataset e Inferenza Parametri ---
    # Utilizza AddNodeFeatures per creare feature dei nodi se non esistono
    # La dimensione iniziale (es. 1 per un vettore di uni) sarà proiettata da GINENet.node_encoder
    node_feat_dim_initial = 1 # Esempio: usa 1 come dimensione iniziale delle feature (es. vettori di uni)
    node_feature_creator = AddNodeFeatures(node_feature_dim=node_feat_dim_initial)

    num_dataset_classes = None
    num_edge_features_resolved = None
    num_node_features_initial = node_feat_dim_initial # Predefinito, verrà confermato dal dataset

    # Carica il dataset di training (se specificato) per inferire i parametri
    full_train_dataset = full_train_dataset_outer # Usa quello passato se disponibile
    if args.train_path:
        if full_train_dataset is None:
            logging.info(f">>> Loading training dataset from: {args.train_path}")
            full_train_dataset = GraphDataset(args.train_path, add_zeros_transform=node_feature_creator)

        if len(full_train_dataset) > 0:
            num_dataset_classes = full_train_dataset.num_classes
            num_edge_features_resolved = full_train_dataset.num_edge_features
            if hasattr(full_train_dataset[0], 'x') and full_train_dataset[0].x is not None:
                num_node_features_initial = full_train_dataset[0].x.size(1)
            else: # Se AddNodeFeatures non è stato applicato o x è ancora None
                num_node_features_initial = node_feat_dim_initial # Fallback
            logging.info(">>> Parameters inferred from training dataset:")
            logging.info(f"  Num node features (initial): {num_node_features_initial}")
            logging.info(f"  Num edge features          : {num_edge_features_resolved}")
            logging.info(f"  Num classes                : {num_dataset_classes}")
        else:
            logging.error("Training dataset is empty. Cannot infer parameters.")
            return
    else: # Se non c'è un dataset di training, usa args o valori di default (potrebbe essere problematico)
        logging.warning(">>> No training path provided. Using parameters from args or defaults.")
        num_dataset_classes = args.num_classes_fallback # Aggiungi questo arg se necessario
        num_edge_features_resolved = args.num_edge_features
        num_node_features_initial = node_feat_dim_initial # O un altro arg
        logging.info(">>> Parameters from args/defaults:")
        logging.info(f"  Num node features (initial): {num_node_features_initial}")
        logging.info(f"  Num edge features          : {num_edge_features_resolved}")
        logging.info(f"  Num classes                : {num_dataset_classes}")

    if num_dataset_classes is None:
        logging.error("Number of classes could not be determined. Exiting.")
        return

    # --- Costruzione Modello ---
    logging.info(">>> Building the model...")
    if args.gnn_type == 'gine':
        model = GINENet(
            in_channels=num_node_features_initial,
            hidden_channels=args.emb_dim,
            out_channels=num_dataset_classes,
            num_layers=args.num_layer,
            edge_dim=num_edge_features_resolved,
            train_eps=True, # Puoi renderlo un argomento del parser se necessario
            dropout_rate=args.drop_ratio # Passa il tipo di pooling
        ).to(device)
    elif args.gnn_type == 'transformer':
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
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"• Model architecture      : {args.gnn_type}")
        logging.info(f"• Number of layers        : {args.num_layer}")
        logging.info(f"• Embedding dimension     : {args.emb_dim}")
        logging.info(f"• Dropout                 : {args.drop_ratio}")
        logging.info(f"• Pooling                 : {args.graph_pooling}")
        logging.info(f"• Residual                : {not args.no_residual}")
        logging.info(f"• Total parameters        : {total_params:,}\n")
    else:
        raise ValueError(f"Unsupported GNN type: {args.gnn_type}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"• Model architecture      : {args.gnn_type}")
    logging.info(f"• Number of layers        : {args.num_layer}")
    logging.info(f"• Embedding dimension     : {args.emb_dim}")
    logging.info(f"• Dropout                 : {args.drop_ratio}")
    logging.info(f"• Graph Pooling           : {args.graph_pooling if args.gnn_type == 'gine' else 'N/A for GINENet as implemented here'}")
    # logging.info(f"• Residual                : {not args.no_residual}") # Non rilevante per GINENet base
    logging.info(f"• Total parameters        : {total_params:,}\n")

    optimizer_model = torch.optim.AdamW(model.parameters(), lr=args.lr_model, weight_decay=args.weight_decay)

    checkpoint_path_best = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_{args.gnn_type}_best.pth")
    checkpoints_folder_epochs = os.path.join(script_dir, "checkpoints", test_dir_name, args.gnn_type)
    os.makedirs(checkpoints_folder_epochs, exist_ok=True)

    if os.path.exists(checkpoint_path_best) and not args.train_path: # Solo se non si sta per fare training
        logging.info(f"Loading pre-trained model from {checkpoint_path_best}")
        model.load_state_dict(torch.load(checkpoint_path_best, map_location=device))
        logging.info("Model loaded successfully.")


    if args.num_checkpoints is not None and args.num_checkpoints > 0:
        if args.num_checkpoints == 1: checkpoint_intervals = [args.epochs]
        else: checkpoint_intervals = [int((i + 1) * args.epochs / args.num_checkpoints) for i in range(args.num_checkpoints)]
    else: checkpoint_intervals = []

    train_loader = train_loader_outer
    val_loader = val_loader_outer

    # --- Training ---
    if args.train_path:
        if full_train_dataset is None: # Dovrebbe essere già caricato, ma per sicurezza
            logging.error("full_train_dataset is None, though args.train_path was provided. This is an issue.")
            return

        logging.info(">>> Preparing train and validation datasets...")
        try:
            # Assicurati che le etichette siano effettivamente dei tensori scalari o interi
            labels_for_split = [data.y.item() if isinstance(data.y, torch.Tensor) else data.y for data in full_train_dataset if data.y is not None]
            if not all(isinstance(label, (int, float)) for label in labels_for_split): # np.int64 è ok
                raise ValueError("Labels for split must be numeric.")
        except Exception as e:
            logging.error(f"Could not extract labels for stratified split: {e}. Ensure dataset elements have a .y attribute containing a single numeric label.")
            # Fallback a split non stratificato se le etichette sono problematiche
            logging.warning("Falling back to non-stratified split due to issues with labels.")
            train_indices, val_indices = train_test_split(
                range(len(full_train_dataset)),
                test_size=args.val_split_ratio,
                shuffle=True,
                random_state=args.seed
            )
        else: # Try-else, se l'estrazione delle etichette ha successo
            if len(set(labels_for_split)) < 2 and args.val_split_ratio > 0: # Poche classi per stratificare
                logging.warning("Not enough unique labels for stratified split. Using non-stratified split.")
                train_indices, val_indices = train_test_split(
                    range(len(full_train_dataset)),
                    test_size=args.val_split_ratio,
                    shuffle=True,
                    random_state=args.seed)
            elif args.val_split_ratio > 0:
                train_indices, val_indices = train_test_split(
                    range(len(full_train_dataset)),
                    test_size=args.val_split_ratio,
                    shuffle=True,
                    stratify=labels_for_split,
                    random_state=args.seed
                )
            else: # No validation split
                train_indices = list(range(len(full_train_dataset)))
                val_indices = []


        train_dataset_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
        if val_indices:
            val_dataset_subset = torch.utils.data.Subset(full_train_dataset, val_indices)
        else:
            val_dataset_subset = None

        logging.info("\n" + "-"*60 + "\n>> Dataset: split in training/validation\n" + "-"*60 + "\n")
        logging.info(f"• Full dataset size       : {len(full_train_dataset)}")
        logging.info(f"• Training set            : {len(train_dataset_subset)} campioni")
        if val_dataset_subset:
            logging.info(f"• Validation set          : {len(val_dataset_subset)} campioni\n")
        else:
            logging.info(f"• Validation set          : 0 campioni (val_split_ratio might be 0)\n")


        if train_loader is None:
            train_loader = DataLoader(train_dataset_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        if val_loader is None and val_dataset_subset:
            val_loader = DataLoader(val_dataset_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        elif val_loader is None and not val_dataset_subset:
            val_loader = None # Esplicito

        full_train_loader_for_atrain = None
        if args.criterion == "gcod":
            logging.info(">>> Preparing current training subset loader for atrain calculation (used by GCOD)...")
            full_train_loader_for_atrain = DataLoader(train_dataset_subset, batch_size=args.batch_size, shuffle=False)


        criterion_obj = None
        optimizer_loss_params = None
        criterion_obj_ce = None # Per epoch_boost con gcod

        if args.criterion == "ce":
            criterion_obj = LabelSmoothingCrossEntropy(classes=num_dataset_classes, smoothing=args.label_smoothing).to(device)
        elif args.criterion == "gcod":
            y_values_numpy_for_gcod = np.array([
                data.y.item() if isinstance(data.y, torch.Tensor) else data.y
                for data in full_train_dataset # Usa il dataset completo per le etichette originali
                if data.y is not None
            ])
            # original_idx è cruciale per GCOD, GraphDataset lo aggiunge
            # Assicurati che gli indici passati a gcodLoss (tramite data.original_idx in train_epoch)
            # corrispondano a come y_values_numpy_for_gcod è ordinato.
            # Se full_train_dataset è una lista di oggetti Data, l'ordine è preservato.

            logging.info("\n" + "-"*60 + "\n>> Initializing Loss Function\n" + "-"*60 + "\n")
            logging.info(f"• Loss function used    : GCODLoss")
            logging.info(f"• Number of classes     : {num_dataset_classes}")
            logging.info(f"• Embeddings size       : {args.emb_dim}") # gnn_embedding_dim per gcodLoss
            logging.info(f"• Total train samples   : {len(y_values_numpy_for_gcod)}") # o len(full_train_dataset)
            logging.info(f"• Total epochs          : {args.epochs}")
            logging.info(f"• Device                : {device}\n")

            criterion_obj = gcodLoss(
                sample_labels_numpy=y_values_numpy_for_gcod, # Etichette di tutto il dataset di training originale
                device=device,
                num_examp=len(y_values_numpy_for_gcod), # Num campioni per cui abbiamo etichette
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

        best_val_metric = 0.0 # Può essere F1 o Accuratezza
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

            # Determina quale criterion usare per questo epoch (per GCOD con epoch_boost)
            current_criterion_obj = criterion_obj
            current_criterion_type = args.criterion
            if args.criterion == "gcod" and args.epoch_boost > 0 and epoch < args.epoch_boost:
                current_criterion_obj = criterion_obj_ce
                current_criterion_type = "ce"

            current_optimizer_loss_params = optimizer_loss_params
            if current_criterion_type == "ce": # Non serve optimizer per CE loss standard
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
            if val_loader:
                avg_val_loss, val_acc, val_f1 = evaluate_model(
                    model=model, loader=val_loader, device=device,
                    criterion_obj=current_criterion_obj, # Usa lo stesso criterion del training per consistenza
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

            # Usa val_f1 per early stopping e best model, ma potrebbe essere val_acc
            current_val_metric = val_f1 if val_loader else train_f1 # Se non c'è val, usa train F1

            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                torch.save(model.state_dict(), checkpoint_path_best)
                logging.info(f"Best model updated and saved to {checkpoint_path_best} (Val F1: {best_val_metric*100:.2f}%)")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement for {epochs_no_improve} epoch(s). Best Val F1: {best_val_metric*100:.2f}%")

            # Early stopping (opzionale)
            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                logging.info(f"Early stopping triggered after {args.early_stopping_patience} epochs with no improvement.")
                break

        total_training_time = time.time() - start_time_train
        logging.info("\n" + "="*60 + f"\n>>> Training completed in {total_training_time:.2f} seconds\n" + "="*60 + "\n")
        plot_training_progress({"train_loss": train_losses_history, "val_loss": val_losses_history},
                               {"train_acc": train_accuracy_history, "train_f1": train_f1_history, "val_acc": val_accuracy_history, "val_f1": val_f1_history},
                               os.path.join(logs_folder, f"training_plots_{args.gnn_type}"))

    # --- Predizione sul Test Set ---
    if args.predict == 1:
        if not os.path.exists(args.test_path):
            logging.error(f"Test path {args.test_path} does not exist.")
            return

        checkpoint_to_load_path = checkpoint_path_best
        if not os.path.exists(checkpoint_to_load_path):
            logging.error(f"No model checkpoint found at {checkpoint_to_load_path} for GNN type {args.gnn_type}. Cannot make predictions.")
            return

        logging.info(f">>> Loading model from {checkpoint_to_load_path} for prediction.")
        model.load_state_dict(torch.load(checkpoint_to_load_path, map_location=device))
        logging.info("Model loaded successfully for prediction.")

        logging.info(">>> Preparing test dataset...")
        # Usa node_feature_creator anche per il test_dataset per coerenza
        test_dataset = GraphDataset(args.test_path, add_zeros_transform=node_feature_creator)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        logging.info(">>> Generating predictions for the test set...")
        # evaluate_model per il test set di solito non calcola la loss, ma solo predizioni/metriche
        # Assicurati che `evaluate_model` possa gestire `criterion_obj=None` o un flag per la modalità test
        # La tua chiamata corrente non passa criterion_obj, il che è buono per il test.
        test_predictions_output = evaluate_model( # Modificato per catturare l'output, che dovrebbe essere le predizioni
            model=model, loader=test_loader, device=device,
            criterion_obj=None, # Non calcolare la loss per il test
            criterion_type="ce", # o il tuo tipo, ma non dovrebbe essere usato se criterion_obj è None
            is_validation=False, # Indica che è test, non validazione
            num_classes_dataset=num_dataset_classes # Necessario per alcune metriche
        )

        # Assumiamo che evaluate_model in modalità test (is_validation=False, criterion_obj=None)
        # restituisca una lista/array di predizioni.
        # Se restituisce (loss, acc, f1, predictions_list), estrai predictions_list.
        # Per ora assumo che test_predictions_output sia la lista di predizioni.
        # Se evaluate_model restituisce (avg_loss, accuracy, f1_score, all_preds, all_labels)
        # allora test_predictions = test_predictions_output[3] (o simile)
        # Adatta in base a cosa restituisce evaluate_model
        if isinstance(test_predictions_output, tuple) and len(test_predictions_output) >= 4:
            test_predictions = test_predictions_output[3] # Esempio se le predizioni sono il 4° elemento
        else: # Se evaluate_model restituisce solo le predizioni
            test_predictions = test_predictions_output

        submission_folder = os.path.join(script_dir, "submissions")
        os.makedirs(submission_folder, exist_ok=True)
        # Modifica il nome del file di submission per includere il tipo di GNN e il dataset
        submission_filename = f"predictions_{test_dir_name}_{args.gnn_type}.txt"
        submission_path = os.path.join(submission_folder, submission_filename)

        save_predictions(test_predictions, submission_path) # Passa il path completo
        logging.info(f"Predictions saved to {submission_path}")

    logging.info("Main script finished <<<")
    return full_train_dataset, train_loader, val_loader


TEST_PATH = "../datasets/B/test.json.gz"
TRAIN_PATH = "../datasets/B/train.json.gz"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")

    parser.add_argument("--train_path", type=str, default=TRAIN_PATH, help="Path to the training dataset.")
    parser.add_argument("--test_path", type=str, default=TEST_PATH, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints during training. 0 to disable.")

    parser.add_argument('--gnn_type', type=str, default='gine', choices=['transformer','gine'], help='GNN architecture.')
    parser.add_argument('--num_layer', type=int, default=3, help='Number of GNN layers.')
    parser.add_argument('--emb_dim', type=int, default=128, help='Dimensionality of hidden units.') # Era 204, standardizzato a 128
    parser.add_argument('--drop_ratio', type=float, default=0.1, help='Dropout ratio.')
    parser.add_argument('--transformer_heads', type=int, default=4, help='Num heads for TransformerConv.') # Specifico per Transformer
    parser.add_argument('--num_edge_features', type=int, default=7, help='Dimensionality of edge features (used if not inferred).')
    parser.add_argument('--jk_mode', type=str, default="last", help="Jumping Knowledge mode.") # Non per GINE base
    parser.add_argument('--graph_pooling', type=str, default="add", choices=["sum", "mean", "max", "add"], help="Graph pooling method for GINE.") # "attention", "set2set" rimossi perché non in GINE base
    parser.add_argument('--no_residual', action='store_true', help='Disable residual connections.') # Non per GINE base

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs.') # Era 1, aumentato a 50
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr_model', type=float, default=1e-3, help='Learning rate for GNN model.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Max norm for gradient clipping (0 to disable).')
    parser.add_argument('--val_split_ratio', type=float, default=0.15, help='Fraction for validation split.')
    parser.add_argument('--early_stopping_patience', type=int, default=0, help='Patience for early stopping. 0 to disable.')


    parser.add_argument("--criterion", type=str, default="ce", choices=["ce", "gcod"], help="Loss function.") # Default a 'ce'
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing for CE loss.')
    parser.add_argument("--lr_u", type=float, default=0.01, help="Learning rate for 'u' in GCOD.")
    parser.add_argument("--lambda_l3_weight", type=float, default=0.7, help="Weight for L3 in GCOD.")
    parser.add_argument('--epoch_boost', type=int, default=0, help='Initial epochs with CE before GCOD.')

    parser.add_argument('--device', type=int, default=0, help='GPU device ID (-1 for CPU).')
    parser.add_argument('--seed', type=int, default=777, help='Random seed.')
    parser.add_argument("--predict", type=int, default=1, choices=[0,1], help="Generate predictions (1) or not (0).")
    parser.add_argument('--num_workers', type=int, default=0, help='Dataloader workers.')

    # Fallback per num_classes se non c'è train_path (dovrebbe essere evitato)
    parser.add_argument('--num_classes_fallback', type=int, default=6, help='Fallback for num_classes if not inferred from train_dataset.')


    args = parser.parse_args()
    main(args)



