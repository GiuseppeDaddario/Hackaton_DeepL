## Imports

import argparse
import logging
import os

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

## Modular imports
from modular.evaluation import evaluate
from modular.statistics import save_predictions, plot_training_progress
from modular.train import train
from modular.dataLoader import add_zeros

from src.loadData import GraphDataset
from src.loss import ncodLoss
from src.models import GNN
from src.utils import set_seed

# Set the random seed
set_seed()



######################################################
##                                                  ##
##                   MAIN FUNCTION                  ##
##                                                  ##
######################################################

def main(args):
    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3

    print("Building the model...")
    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #criterion = torch.nn.CrossEntropyLoss()

    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    
    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well


    # Define checkpoint path relative to the script's directory
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # Load pre-trained model for inference
    if os.path.exists(checkpoint_path) and not args.train_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded best model from {checkpoint_path}")


    # If train_path is provided, train the model
    if args.train_path:
        print("Preparing train and valid datasets...")
        train_dataset = GraphDataset(args.train_path, transform=add_zeros)

        # Crea i DataLoader per training e validation
        print("Loading train dataset...")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


        print("Computing the ncod loss")
        y_values = torch.tensor([graph["y"][0] for graph in train_dataset.graphs_dicts])
        if args.criterion == "ncod":
            train_loss = ncodLoss(y_values, device, num_examp=len(train_dataset),
                                  num_classes=6,
                                  ratio_consistency=0, ratio_balance=0,
                                  encoder_features=300, total_epochs=args.epochs)
            if train_loss.USE_CUDA:
                train_loss.to(device)
        elif args.criterion == "ce":
            train_loss = torch.nn.CrossEntropyLoss()


        optimizer_overparametrization = optim.SGD([train_loss.u], lr=args.lr_u)

        # Training loop
        num_epochs = args.epochs
        best_accuracy = 0.0
        train_losses = []
        train_accuracies = []

        # Calculate intervals for saving checkpoints
        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        train_acc_cater = 0.0
        print("Starting training...")
        for epoch in range(num_epochs):
            train_acc, training_loss, train_acc_cater = train(
                train_acc_cater,
                train_loader,
                model,
                optimizer,
                device,
                optimizer_overparametrization,
                train_loss,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )
            #train_acc, _ = evaluate(train_loader, model, device, calculate_accuracy=True)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {training_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Salva i valori scalari in liste separate
            train_losses.append(training_loss)
            train_accuracies.append(train_acc)
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {training_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Save best model
            if train_acc > best_accuracy:
                best_accuracy = train_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")

        # Plot training progress in current directory
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))

    if args.predict == 1:
        train_dataset = 0
        train_loader = 0
        # Prepare test dataset and loader
        print("Preparing test dataset...")
        test_dataset = GraphDataset(args.test_path, transform=add_zeros)
        print("Loading test dataset...")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Generate predictions for the test set using the best model
        print("Generating predictions for the test set...")
        model.load_state_dict(torch.load(checkpoint_path))
        predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
        save_predictions(predictions, args.test_path)
        print("Predictions saved successfully.")


######################################################
##                                                  ##
##               END OF MAIN FUNCTION               ##
##                                                  ##
######################################################






if __name__ == "__main__":
    TEST_PATH = "datasets/B/test.json.gz"
    TRAIN_PATH = "datasets/B/train.json.gz"
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--criterion", type=str, default="ncod", help="Type of loss to use")
    parser.add_argument("--lr_u", type=float, default=1.0, help="lr u")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--predict", type=int, default=1, help="Save or not the predictions")
    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    
    args = parser.parse_args()
    main(args)
