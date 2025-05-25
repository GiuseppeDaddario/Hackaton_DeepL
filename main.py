import argparse
import os
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from src.loadData import GraphDataset
from src.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import torch.nn.functional as F

from src.models import GNN
from src.loss import ncodLoss



# Set the random seed
set_seed()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(train_acc_cater,data_loader, model,model_sp, optimizer, device,optimizer_overparametrization,train_loss, train_dataset, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    model_sp.train()

    total_loss = 0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(tqdm(data_loader, desc="Iterating training graphs", unit="batch")):
        inputs, labels = data[0].to(device), data[0].y.to(device)
        target = torch.zeros(len(labels), 6).to(device).scatter_(1, labels.view(-1, 1).long(), 1)
        index_run = [train_dataset.indices[int(key)] for key in data[1]]

        outs_sp, _ = model_sp(inputs)
        prediction = F.softmax(outs_sp, dim=1)
        prediction = torch.sum((prediction * target), dim=1)
        train_loss.weight[index_run] = (prediction.detach()).view(-1, 1)

        optimizer.zero_grad()
        optimizer_overparametrization.zero_grad()
        outputs, emb, _ = model(inputs)
        loss = train_loss(index_run, outputs, target, emb, i, current_epoch,train_acc_cater)
        loss.backward()
        optimizer.step()
        optimizer_overparametrization.step()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels.squeeze()).sum().item()
        total_loss += loss.item()

    train_acc_cater = correct_train / total_train

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    loss_return = (correct_train / total_train) * 100, (total_loss / len(data_loader))
    return loss_return, train_acc_cater

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions

def save_predictions(predictions, test_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()

def main(args):
    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3

    print("Building the model...")
    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
        model_sp = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
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

    # Prepare test dataset and loader
    print("Preparing test dataset...")
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    print("Loading test dataset...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # If train_path is provided, train the model
    if args.train_path:
        print("Preparing train and valid datasets...")
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)

        # Calcola le dimensioni per il training e la validation
        train_size = int(0.8 * len(full_dataset))  # 80% per il training
        val_size = len(full_dataset) - train_size  # 20% per la validation

        # Dividi il dataset
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        # Crea i DataLoader per training e validation
        print("Loading train dataset...")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        print("Loading validation dataset...")
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        classbins = []
        print("Preparing class bins for training dataset...")
        for i in range(6):
            indices = []
            for idx, graph in enumerate(full_dataset.graphs_dicts):
                if graph["y"][0] == i:  # Controlla se la classe del grafo corrisponde a `i`
                    indices.append(idx)
            classbins.append(indices)

        print("Computing the ncod loss")
        y_values = torch.tensor([train_dataset.dataset.get(idx).y for idx in train_dataset.indices])
        train_loss = ncodLoss(y_values, device, num_examp=len(train_dataset.indices),
                              num_classes=6,
                              ratio_consistency=0, ratio_balance=0,
                              encoder_features=300, total_epochs=args.epochs)
        if train_loss.USE_CUDA:
            train_loss.to(device)

        pureIndices = []
        noisyIndices = []
        for x,z in zip(classbins, train_loss.shuffledbins):
            noisyIndices.append(list(set(z) - set(x)))
            pureIndices.append(list(set(z) - (set(z) - set(x))))

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
            training_loss, train_acc_cater = train(train_acc_cater,
                train_loader, model, model_sp, optimizer, device, optimizer_overparametrization, train_loss, train_dataset,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )
            train_acc, _ = evaluate(train_loader, model, device, calculate_accuracy=True)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {training_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Save logs for training progress
            train_losses.append(training_loss)
            train_accuracies.append(train_acc)
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {training_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Save best model
            if train_acc > best_accuracy:
                best_accuracy = train_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")

        # Plot training progress in current directory
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))

    # Generate predictions for the test set using the best model
    print("Generating predictions for the test set...")
    model.load_state_dict(torch.load(checkpoint_path))
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    save_predictions(predictions, args.test_path)
    print("Predictions saved successfully.")

if __name__ == "__main__":
    TEST_PATH = "datasets/B/test.json.gz"
    TRAIN_PATH = "datasets/B/train.json.gz"
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, default=TRAIN_PATH, help="Path to the training dataset (optional).")
    parser.add_argument("--lr_u", type=str, default=1, help="lr u")
    parser.add_argument("--test_path", type=str, default=TEST_PATH, help="Path to the test dataset.")
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
