import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from src.loadData import GraphDataset
from src.utils import set_seed

import logging




from src.models import GNN 
from src.conv import GINETransformerNet
from src.statistics import evaluate, evaluateTwo, save_predictions, plot_training_progress
from src.train import train

# Set the random seed
set_seed()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data









def main(args):
    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3
    
    

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    
    elif args.gnn == 'gine':
        model = GNN(gnn_type = 'gine', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gine-virtual':
        model = GNN(gnn_type = 'gine', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    
    elif args.gnn == 'gineTransformer':
        model = GINETransformerNet(num_layer=args.num_layer, emb_dim=args.emb_dim,num_classes=args.num_classes,gnn_type='gineTransformer',drop_ratio=args.drop_ratio,ff_dim=args.ff_dim,residual=args.residuals,num_heads=args.num_heads).to(device)

    
    # elif args.gnn == 'gineTrans-virtual':
    #    model = GNN(gnn_type = 'gineTransformer', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, ff_dim = args.ff_dim ,residual = args.residual, num_heads = args.num_heads, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

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

    
    
    ####################################
    ###      UPLOADING DATASETS      ### 
    ####################################

    from torch.utils.data import random_split

    print("-------- Loading datasets --------")
    train_dataset = GraphDataset(args.train_path, transform=add_zeros)
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)  # TestSet to be labelled
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) # TestSet to be labelled
    print("-------- Datasets loaded --------")

    # Split dataset: 80% train, 20% validation
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])

    # DataLoaders for training and validation
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False)

    #####################################
    #####################################





    # TRAINING:
    if args.train_path:
        #train_dataset = GraphDataset(args.train_path, transform=add_zeros)
        #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)



        # Training loop
        num_epochs = args.epochs
        best_accuracy = 0.0
        train_losses = []
        train_accuracies = []
        valid_losses = []
        valid_accuracies = []

        # Calculate intervals for saving checkpoints
        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

            
        for epoch in range(num_epochs):
            train_loss, train_acc, train_f1 = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )
            
            
            # EVALUATION
            valid_acc, valid_loss, valid_f1 = evaluateTwo(valid_loader, model, device, criterion = criterion)

            #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Stacking losses and accuracies for plotting
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_acc)
            
            logging.info(f"|Epoch {epoch + 1}/{num_epochs} | T-Loss: {train_loss:.4f} | T-Acc: {train_acc:.4f} V-Loss: {valid_loss:.4f} |V-Acc: {valid_acc:.4f}| T-f1: {train_f1:.4f} | V-f1: {valid_f1:.4f} " )

            # Save best model
            if train_acc > best_accuracy:
                best_accuracy = train_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")

        # Plot training progress in current directory
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))

    # Generate predictions for the test set using the best model
    model.load_state_dict(torch.load(checkpoint_path))
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    save_predictions(predictions, args.test_path)


TRAIN_PATH = r"C:\Users\Lorenzo\Desktop\Progetto\datasets\A\train_part_1.json.gz"
TEST_PATH = r"C:\Users\Lorenzo\Desktop\Progetto\datasets\A\train_part_2.json.gz"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, default=TRAIN_PATH, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, default=TEST_PATH, help="Path to the test dataset.")
    
    ## GENERIC
    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    
    ## GNN-specific parameters
    parser.add_argument('--gnn', type=str, default='gine', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=3, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=128, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--residual', action='store_true', help='Use residual connections in GNN layers')
    
    # Transformer-specific parameters
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in Transformer (default: 4)')
    parser.add_argument('--ff_dim', type=int, default=256, help='Dimensionality of feedforward layer in Transformer (default: 256)')



    args = parser.parse_args()
    main(args)
