import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np
import random
from sklearn.model_selection import train_test_split
from source.loadData import GraphDataset
from source.statistics import save_predictions

# ------------------- HYPERPARAMETERS -------------------
TRAIN_PATH = "../datasets/B/train.json.gz"   # <-- path al dataset completo
TEST_PATH = "../datasets/B/test.json.gz"     # <-- path al test set
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 0.001
EMB_DIM = 300
NUM_LAYERS = 5
NUM_CLASSES = 6
NOISE_RATE = 0.2  # 20% di label noise
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_NUM_NODE_FEATURES = 7  # Imposta qui il numero fisso di feature per nodo
# -------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ----------- MODEL DEFINITION -----------
class GINE_Net(nn.Module):
    def __init__(self, num_features, emb_dim, num_layers, num_classes):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_dim = num_features if i == 0 else emb_dim
            nn_gine = nn.Sequential(
                nn.Linear(in_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim)
            )
            self.convs.append(GINEConv(nn_gine, edge_dim=num_features))
            self.bns.append(nn.BatchNorm1d(emb_dim))
        self.lin = nn.Linear(emb_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)

# ----------- LABEL NOISE UTILITY -----------
def add_label_noise(dataset, noise_rate, num_classes):
    noisy_labels = 0
    for data in dataset:
        if hasattr(data, "y") and data.y is not None and random.random() < noise_rate:
            old = data.y.item()
            new = random.choice([i for i in range(num_classes) if i != old])
            data.y = torch.tensor(new, dtype=torch.long)
            noisy_labels += 1
    print(f"Injected noise in {noisy_labels} samples.")
    return dataset

# ----------- TRAIN & EVAL UTILS -----------
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
        #print(pred, data.y)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for data in loader:
        data = data.to(DEVICE)
        out = model(data)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    return total_loss / total, correct / total

@torch.no_grad()
def predict(model, loader):
    model.eval()
    predictions = []
    for data in loader:
        data = data.to(DEVICE)
        out = model(data)
        pred = out.argmax(dim=1)
        predictions.extend(pred.cpu().numpy())
    return predictions




def ensure_graph_shapes(dataset):
    filtered = []
    for data in dataset:
        # Salta grafi senza nodi
        if not hasattr(data, "num_nodes") or data.num_nodes < 1:
            continue
        # x deve essere 2D
        if not hasattr(data, "x") or data.x is None or data.x.numel() == 0:
            data.x = torch.eye(data.num_nodes, dtype=torch.float)
        elif data.x.dim() == 1:
            data.x = data.x.unsqueeze(1)
        # edge_index deve essere 2D
        if hasattr(data, "edge_index") and data.edge_index is not None and data.edge_index.dim() == 1:
            data.edge_index = data.edge_index.unsqueeze(1)
        # edge_attr, se presente, deve essere 2D
        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.dim() == 1:
            data.edge_attr = data.edge_attr.unsqueeze(1)
        filtered.append(data)
    return filtered

# ----------- MAIN PIPELINE -----------
def main():
    # Carica tutto il dataset train/val
    full_dataset = GraphDataset(TRAIN_PATH)
    # Inizializza x come one-hot se non presente
    
    full_dataset.processed_data_list = [d for d in full_dataset.processed_data_list if hasattr(d, "num_nodes") and d.num_nodes > 0]

    for data in full_dataset.processed_data_list:
        if not hasattr(data, "x") or data.x is None or data.x.numel() == 0:
            # Inizializza tutti i nodi con one-hot sulla prima colonna
            data.x = torch.zeros(data.num_nodes, FIXED_NUM_NODE_FEATURES)
            if data.num_nodes > 0:
                data.x[:, 0] = 1.0
        elif data.x.dim() == 1:
            data.x = data.x.unsqueeze(1)
        # Se data.x.shape[1] != FIXED_NUM_NODE_FEATURES, fai padding o taglia
        elif data.x.shape[1] < FIXED_NUM_NODE_FEATURES:
            pad = FIXED_NUM_NODE_FEATURES - data.x.shape[1]
            data.x = torch.cat([data.x, torch.zeros(data.x.shape[0], pad)], dim=1)
        elif data.x.shape[1] > FIXED_NUM_NODE_FEATURES:
            data.x = data.x[:, :FIXED_NUM_NODE_FEATURES]
        # Ora sistema data.edge_attr
        if not hasattr(data, "edge_attr") or data.edge_attr is None or data.edge_attr.numel() == 0:
            # Se mancano edge_attr, metti zeri
            data.edge_attr = torch.zeros(data.edge_index.size(1), FIXED_NUM_NODE_FEATURES)
        elif data.edge_attr.dim() == 1:
            data.edge_attr = data.edge_attr.unsqueeze(1)
        if data.edge_attr.shape[1] < FIXED_NUM_NODE_FEATURES:
            pad = FIXED_NUM_NODE_FEATURES - data.edge_attr.shape[1]
            data.edge_attr = torch.cat([data.edge_attr, torch.zeros(data.edge_attr.shape[0], pad)], dim=1)
        elif data.edge_attr.shape[1] > FIXED_NUM_NODE_FEATURES:
            data.edge_attr = data.edge_attr[:, :FIXED_NUM_NODE_FEATURES]
    # Split 80/20
    idxs = list(range(len(full_dataset)))
    train_idxs, val_idxs = train_test_split(idxs, test_size=0.2, random_state=SEED, shuffle=True)
    
    
    
    
    
    train_dataset = [full_dataset[i] for i in train_idxs]
    val_dataset = [full_dataset[i] for i in val_idxs]

    train_dataset = ensure_graph_shapes(train_dataset)
    val_dataset = ensure_graph_shapes(val_dataset)
    
    # Applica noise alle label del train
    #train_dataset = add_label_noise(train_dataset, NOISE_RATE, NUM_CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Inferisci feature dimension
    sample = train_dataset[0]
    num_features = FIXED_NUM_NODE_FEATURES

    model = GINE_Net(num_features, EMB_DIM, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_gine_model.pth")
            print("  (saved best model)")

    # ----------- INFERENZA SUL TEST SET -----------
    print("\n--- INFERENZA SUL TEST SET ---")

    # 1. Carica il test set
    test_dataset = GraphDataset(TEST_PATH)

    # 2. Sistema le shape e le feature dei nodi/archi come per il train
    test_dataset.processed_data_list = ensure_graph_shapes(test_dataset.processed_data_list)
    for data in test_dataset.processed_data_list:
        if not hasattr(data, "x") or data.x is None or data.x.numel() == 0:
            # Usa degree come feature se non hai altro
            deg = torch.bincount(data.edge_index[0], minlength=data.num_nodes).float().unsqueeze(1)
            if deg.shape[1] < FIXED_NUM_NODE_FEATURES:
                pad = FIXED_NUM_NODE_FEATURES - deg.shape[1]
                data.x = torch.cat([deg, torch.zeros(data.num_nodes, pad)], dim=1)
            else:
                data.x = deg[:, :FIXED_NUM_NODE_FEATURES]
        elif data.x.dim() == 1:
            data.x = data.x.unsqueeze(1)
        elif data.x.shape[1] < FIXED_NUM_NODE_FEATURES:
            pad = FIXED_NUM_NODE_FEATURES - data.x.shape[1]
            data.x = torch.cat([data.x, torch.zeros(data.x.shape[0], pad)], dim=1)
        elif data.x.shape[1] > FIXED_NUM_NODE_FEATURES:
            data.x = data.x[:, :FIXED_NUM_NODE_FEATURES]
        # Edge_attr come prima
        if not hasattr(data, "edge_attr") or data.edge_attr is None or data.edge_attr.numel() == 0:
            data.edge_attr = torch.zeros(data.edge_index.size(1), FIXED_NUM_NODE_FEATURES)
        elif data.edge_attr.dim() == 1:
            data.edge_attr = data.edge_attr.unsqueeze(1)
        if data.edge_attr.shape[1] < FIXED_NUM_NODE_FEATURES:
            pad = FIXED_NUM_NODE_FEATURES - data.edge_attr.shape[1]
            data.edge_attr = torch.cat([data.edge_attr, torch.zeros(data.edge_attr.shape[0], pad)], dim=1)
        elif data.edge_attr.shape[1] > FIXED_NUM_NODE_FEATURES:
            data.edge_attr = data.edge_attr[:, :FIXED_NUM_NODE_FEATURES]

    test_dataset.processed_data_list = ensure_graph_shapes(test_dataset.processed_data_list)

    # 3. Crea il DataLoader per il test set
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Carica il modello migliore
    model.load_state_dict(torch.load("best_gine_model.pth", map_location=DEVICE))

    # 5. Fai le predizioni
    predictions = predict(model, test_loader)

    # 6. Salva le predizioni in CSV
    save_predictions(predictions, TEST_PATH)

if __name__ == "__main__":
    main()