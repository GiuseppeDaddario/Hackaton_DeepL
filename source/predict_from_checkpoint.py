# Import dai moduli esistenti
import argparse
# Imposta il seed
import os

import torch
from torch_geometric.loader import DataLoader

from source.dataLoader import add_zeros
## Modular imports
from source.evaluation import evaluate
from source.loadData import GraphDataset
from source.models import GNN
from source.statistics import save_predictions
from source.utils import set_seed

set_seed()

def main(args):
    # Configurazione manuale (puoi modificarla o passare via argparse se vuoi)
    checkpoint_path = args.checkpoint_path  # checkpoint già salvato
    test_path = args.test_path                      # path al file di test
    gnn_type = "gin"                                   # o "gcn", "gin-virtual", etc.
    num_classes = 6
    num_layer = args.num_layer
    emb_dim = args.emb_dim
    drop_ratio = args.drop_ratio
    batch_size = args.batch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Costruzione del modello (usando il costruttore esistente)
    model = GNN(
        gnn_type = gnn_type if "virtual" not in gnn_type else gnn_type.split("-")[0],
        num_class = num_classes,
        num_layer = num_layer,
        emb_dim = emb_dim,
        drop_ratio = drop_ratio,
        virtual_node = "virtual" in gnn_type
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
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)

    # Salvataggio predizioni
    save_predictions(predictions, test_path)
    print("Predictions saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the testset with predictions starting from a checkpoint.")
    parser.add_argument('--checkpoint_path', type=int, required=True, help='path of the checkpoint to load')
    parser.add_argument('--test_path', type=int, required=True, help='path of the testset to load')
    parser.add_argument('--gnn', type=str, default='gin', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    args = parser.parse_args()
    main(args)
