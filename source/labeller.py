import argparse
import os
import torch
from torch_geometric.loader import DataLoader

from source.loadData import GraphDataset
from source.models import GINENetWithTransformer # <--- MODIFICATO QUI
from source.dataLoader import AddNodeFeatures
from source.evaluation import evaluate_model
from source.statistics import save_predictions
from source.utils import set_seed

def predict_with_model(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    print(f"Using device: {device}")

    num_node_features_initial = args.node_feat_dim_initial
    num_edge_features_resolved = args.num_edge_features_from_training
    num_dataset_classes = args.num_classes_from_training

    if num_edge_features_resolved is None:
        print("ATTENZIONE: num_edge_features_from_training non specificato. Assumo 0.")
        num_edge_features_resolved = 0

    if num_dataset_classes is None:
        raise ValueError("num_classes_from_training deve essere specificato.")

    print(f"Parametri per la costruzione del modello (DEVONO CORRISPONDERE AL TRAINING):")
    print(f"  Initial Node Feature Dim: {num_node_features_initial}")
    print(f"  Edge Feature Dim        : {num_edge_features_resolved}")
    print(f"  Number of Classes       : {num_dataset_classes}")
    print(f"  GIN Layers              : {args.num_layer}")
    print(f"  Embedding Dim           : {args.emb_dim}")
    if args.transformer_layers > 0 : # Logga i parametri del Transformer solo se usati
        print(f"  Transformer Layers      : {args.transformer_layers}")
        print(f"  Transformer Heads       : {args.transformer_heads}")
    print(f"  Dropout Rate            : {args.drop_ratio}")
    print(f"  Graph Pooling           : {args.graph_pooling}")
    print(f"  Residual Connections    : {'Yes' if not args.no_residual else 'No'}")
    print(f"  Jumping Knowledge       : {args.jk}")



    print("Building the model...")
    model = GINENetWithTransformer(
        in_channels=num_node_features_initial,
        hidden_channels=args.emb_dim,
        out_channels=num_dataset_classes,
        num_gin_layers=args.num_layer, # Numero di layer GIN
        no_residual=args.no_residual,
        jk=args.jk,
        edge_dim=num_edge_features_resolved,
        dropout_rate=args.drop_ratio,
        graph_pooling=args.graph_pooling,
        # Parametri del Transformer
        num_transformer_layers=args.transformer_layers, # Es. 2
        transformer_nhead=args.transformer_heads,           # Es. 4 (deve dividere emb_dim)
        transformer_dim_feedforward=(args.emb_dim * 2)
    ).to(device)
    print("Model built.")

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    model.eval()

    node_feature_creator = AddNodeFeatures(node_feature_dim=num_node_features_initial)
    print(f"Loading test dataset from: {args.test_path}")
    try:
        test_dataset = GraphDataset(args.test_path, add_zeros_transform=node_feature_creator)
        if len(test_dataset) == 0:
            print("ATTENZIONE: Test dataset is empty.")
            return
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("Test dataset loaded.")

    print("Generating predictions...")
    try:
        eval_results = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            criterion_obj=None,
            criterion_type=args.criterion,
            num_classes_dataset=num_dataset_classes,
            lambda_l3_weight=0.0,
            current_epoch_for_gcod=-1,
            atrain_for_gcod=0.0,
            is_validation=False
        )
    except Exception as e:
        print(f"Error during model evaluation (prediction): {e}")
        import traceback
        traceback.print_exc()
        return

    if not (isinstance(eval_results, tuple) and len(eval_results) >= 4):
        print("Errore: evaluate_model non ha restituito il formato atteso (tuple con almeno 4 elementi).")
        print(f"Ricevuto: {eval_results}")
        return

    all_predictions = eval_results[3]
    if eval_results[4] is not None and len(eval_results[4]) > 0 and len(eval_results) >= 5: # Assicura che ci sia il 5° elemento
        print(f"Test Set Metrics (se calcolate): Loss: {eval_results[0]:.4f}, Acc: {eval_results[1]:.4f}, F1: {eval_results[2]:.4f}")

    if all_predictions is None or len(all_predictions) == 0:
        print("Nessuna predizione generata.")
        return
    print(f"Generated {len(all_predictions)} predictions.")

    output_prediction_filename = f"predictions_{os.path.basename(args.test_path).replace('.json.gz', '')}_{args.gnn_type}"
    if args.transformer_layers > 0: # Aggiungi info sul transformer al nome del file
        output_prediction_filename += f"_tf{args.transformer_layers}"
    output_prediction_filename += ".txt"

    os.makedirs(args.output_dir, exist_ok=True)
    output_prediction_path = os.path.join(args.output_dir, output_prediction_filename)

    try:
        save_predictions(all_predictions, output_prediction_path)
        print(f"Predictions saved to: {output_prediction_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")

if __name__ == "__main__":
    # Stai creando args manualmente, quindi definisci qui TUTTI i parametri necessari
    # per GINENetWithTransformer che corrispondono al modello addestrato.
    args = argparse.Namespace(
        # Path
        checkpoint_path='/Users/giuseppedaddario/Downloads/checkpoints_Gcod/model_B_gine_best.pth', # << MODIFICA CON IL PATH AL TUO CHECKPOINT
        test_path='../datasets/A/test.json.gz',
        output_dir='/Users/giuseppedaddario/Desktop/submission', # << MODIFICA DIR OUTPUT

        # Architettura del modello (DEVE CORRISPONDERE AL CHECKPOINT)
        gnn_type='gine',
        num_layer=3,                                # Numero di layer GIN
        emb_dim=153,
        drop_ratio=0.1,                             # Valore usato nel training
        graph_pooling='attention',
        node_feat_dim_initial=1,
        num_edge_features_from_training=7,          # Valore usato nel training (es. 7 per dataset A, o 0)
        num_classes_from_training=6,                # Valore usato nel training (es. 6 per dataset A)
        no_residual=False,                          # Se il modello addestrato ha residual connections
        jk='last',                                  # Se il modello addestrato usa Jumping Knowledge (es. 'last' o 'cat')

        # Parametri del Transformer (DEVE CORRISPONDERE AL CHECKPOINT)
        transformer_layers=1,                   # Es. 1 se il checkpoint ha 1 layer Transformer
        transformer_heads=3,                        # Es. 4 (se emb_dim=128, 128/4=32 OK)           # Es.

        # Runtime
        batch_size=32,
        device=0,
        seed=777,
        num_workers=0, # Su macOS spesso 0 è più stabile per PyG se non hai configurato bene il multiprocessing

        # Criterio (passato a evaluate_model, anche se non per la loss in predizione)
        criterion='gcod' # O 'gcod' se evaluate_model lo usa per qualche logica interna anche in test
    )

    print("Argomenti per la predizione:")
    for k,v in vars(args).items():
        print(f"  {k}: {v}")

    predict_with_model(args)