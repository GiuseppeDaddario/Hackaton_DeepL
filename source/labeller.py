## Imports
import argparse
import os
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch_geometric.loader import DataLoader

## Modular imports
from source.evaluation import evaluate
from source.loadData import GraphDataset
from source.models import GNN
from source.statistics import save_predictions_1
from source.utils import set_seed
from source.dataLoader import add_zeros


#####################################     
###  PUT YOUR CONFIGURATION HERE  ###
#####################################

DATASET = "D"
WEIGHT_PATH = "../checkpoints/model_sequential_final.pth"  # The path to the checkpoint you want to load
GNN_TYPE = "gin"                                        # o "gcn", "gin-virtual", etc.
DROP_RATIO = 0.5                                        # Dropout ratio for the GNN
NUM_LAYER = 5                                           # Number of GNN layers
EMB_DIM = 300                                           # Dimensionality of hidden units in GNNs
BATCH_SIZE = 32                                         # Input batch size for training



#####
TEST_PATH = f"../../datasets/{DATASET}/test.json.gz"
OUTPUT_PATH = f"../submission/testset_{DATASET}.csv"
# Calcola il path assoluto rispetto a questo script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(CURRENT_DIR, WEIGHT_PATH)
OUTPUT_PATH = os.path.join(CURRENT_DIR, OUTPUT_PATH)
TEST_PATH = os.path.join(CURRENT_DIR, TEST_PATH)
############################################
############################################




set_seed()

def main(args):

    
    weight_path = args.weight_path  
    test_path = args.test_path     
    output_path = args.output_path                 
    gnn_type = "gin"                                  
    num_classes = 6
    num_layer = args.num_layer
    emb_dim = args.emb_dim
    drop_ratio = args.drop_ratio
    batch_size = args.batch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Definition of the model
    model = GNN(
        gnn_type = gnn_type if "virtual" not in gnn_type else gnn_type.split("-")[0],
        num_class = num_classes,
        num_layer = num_layer,
        emb_dim = emb_dim,
        drop_ratio = drop_ratio,
        virtual_node = "virtual" in gnn_type
    ).to(device)

    # Uploading weights
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Caricamento del dataset di test
    test_dataset = GraphDataset(test_path, transform=add_zeros if "add_zeros" in globals() else None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Generazione predizioni
    print("Generating predictions...")
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)

    # Salvataggio predizioni
    save_predictions_1(predictions, output_path)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the testset with predictions starting from a checkpoint.")
    parser.add_argument('--weight_path', type=str, default=WEIGHT_PATH, help='path of the checkpoint to load')
    parser.add_argument('--test_path', type=str, default=TEST_PATH, help='path of the testset to load')
    parser.add_argument('--gnn', type=str, default=GNN_TYPE, help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin)')
    parser.add_argument('--drop_ratio', type=float, default=DROP_RATIO, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=NUM_LAYER, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=EMB_DIM, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='input batch size for training (default: 32)')
    parser.add_argument('--output_path', type=str, default=OUTPUT_PATH, help='input batch size for training (default: 32)')
    args = parser.parse_args()
    
    
    
    
    main(args) ## RUNNING THE MAIN FUNCTION
