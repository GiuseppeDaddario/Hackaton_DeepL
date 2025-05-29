# Import dai moduli esistenti
import argparse
# Imposta il seed
import os

import torch
from torch_geometric.loader import DataLoader

from source.dataLoader import add_zeros
## Modular imports
from source.evaluation import evaluate_model as evaluate
from source.loadData import GraphDataset
from source.models import GNN
from source.statistics import save_predictions
from source.utils import set_seed

set_seed()

def main(args):
    # Configurazione manuale (puoi modificarla o passare via argparse se vuoi)
    checkpoint_path = args.checkpoint_path  # checkpoint già salvato
    test_path = args.test_path                      # path al file di test
    batch_size = args.batch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Costruzione del modello (usando il costruttore esistente)
    model = GNN(
        gnn_type=args.gnn_type,
        num_class=6,
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
        JK=args.jk_mode,
        graph_pooling=args.graph_pooling,
        num_edge_features=args.num_edge_features,
        transformer_heads=args.transformer_heads,
        residual=not args.no_residual
    ).to(device)

    # Caricamento checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Caricamento del dataset di test
    test_dataset = GraphDataset(test_path, transform=add_zeros if "add_zeros" in globals() else None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Generazione predizioni
    print("Generating predictions...")
    predictions = evaluate(model,
                           test_loader,
                           device,
                           criterion_obj=None, # Può essere None se is_validation è False
                           criterion_type="gcod",
                           num_classes_dataset=None, # Necessario solo se is_validation e criterion_type="gcod"
                           lambda_l3_weight=args.lambda_l3_weight,   # Necessario solo se is_validation e criterion_type="gcod"
                           current_epoch_for_gcod=-1, # Necessario solo se is_validation e criterion_type="gcod"
                           atrain_for_gcod=0.0,       # Necessario solo se is_validation e criterion_type="gcod"
                           is_validation=False)

    # Salvataggio predizioni
    save_predictions(predictions, test_path)
    print("Predictions saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")

    parser.add_argument("--test_path", type=str, default="/Users/giuseppedaddario/Documents/Hackaton_DeepL/datasets/A/test.json.gz", help="Path to the test dataset.")
    parser.add_argument("--checkpoint_path", type=str, default="/Users/giuseppedaddario/Downloads/model_A_best_80.pth", help="Path to the model checkpoint to load.")

    # Model Architecture
    parser.add_argument('--gnn_type', type=str, default='transformer', choices=['transformer'], help='GNN architecture type (default: transformer)')
    parser.add_argument('--num_layer', type=int, default=1, help='Number of GNN message passing layers (default: 3)')
    parser.add_argument('--emb_dim', type=int, default=64, help='Dimensionality of hidden units in GNNs (default: 128)')
    parser.add_argument('--drop_ratio', type=float, default=0.1, help='Dropout ratio (default: 0.1)')
    parser.add_argument('--transformer_heads', type=int, default=1, help='Number of attention heads for TransformerConv (default: 4)')
    parser.add_argument('--num_edge_features', type=int, default=7, help='Dimensionality of edge features (default: 7, VERIFY FROM DATASET)')
    parser.add_argument('--jk_mode', type=str, default="last", choices=["last", "sum", "mean", "concat"], help="Jumping Knowledge mode (default: last)")
    parser.add_argument('--graph_pooling', type=str, default="mean", choices=["sum", "mean", "max", "attention", "set2set"], help="Graph pooling method (default: mean)")
    parser.add_argument('--no_residual', action='store_true', help='Disable residual connections in GNN layers.')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32)')
    parser.add_argument('--lr_model', type=float, default=1e-3, help='Learning rate for the GNN model (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty) for AdamW (default: 1e-5)')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Max norm for gradient clipping (0 to disable, default: 1.0)')
    parser.add_argument('--val_split_ratio', type=float, default=0.15, help='Fraction of training data to use for validation (default: 0.15)')

    # Loss Function and GCOD specific
    parser.add_argument("--criterion", type=str, default="gcod", choices=["ce", "gcod"], help="Type of loss to use (default: ce)")
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Amount of label smoothing for CE loss (default: 0.1, use 0 for no smoothing)')
    parser.add_argument("--lr_u", type=float, default=0.01, help="Learning rate for 'u' parameters in GCOD (default: 0.01)")
    parser.add_argument("--lambda_l3_weight", type=float, default=1.0, help="Weight for L3 component in GCOD loss (default: 0.7)")
    parser.add_argument('--epoch_boost', type=int, default=0, help='Number of initial epochs with CE loss before GCOD (default: 0)')

    # Runtime and Misc
    parser.add_argument('--device', type=int, default=0, help='GPU device ID to use if available (default: 0, -1 for CPU)')
    parser.add_argument('--seed', type=int, default=2177530, help='Random seed (default: 777)')
    parser.add_argument("--predict", type=int, default=1, choices=[0,1], help="Generate and save predictions on the test set (default: 1)")
    parser.add_argument('--num_workers', type=int, default=0, help='Number of Dataloader workers (default: 0 for main process, >0 for multiprocessing)')

    args = parser.parse_args()
    main(args)


