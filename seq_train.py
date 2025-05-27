import os
from main import main
import torch
from source.models import GNN

def sequential_training(args):
    model = None
    train_dataset = None
    train_loader = None

    datasets_paths = [
        "../datasets/A/train.json.gz",
        "../datasets/B/train.json.gz",
        "../datasets/C/train.json.gz",
        "../datasets/D/train.json.gz",
    ]

    args.ret = "all"


    ## loading checkpoint if exists
    checkpoint_path = os.path.join("checkpoints", "model_sequential_final.pth")
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = GNN(
            gnn_type=args.gnn,
            num_class=6,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node=('virtual' in args.gnn)
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model = None

    ## End of loading checkpoint


    for i, train_path in enumerate(datasets_paths):
        print(f"\n--- Training on dataset {chr(ord('A') + i)}: {train_path} ---")
        args.train_path = train_path
        args.test_path = train_path
        train_dataset, model, train_loader = main(
            args,
            train_dataset=train_dataset,
            train_loader_for_batches=train_loader,
            model=model
        )
    print("Sequential training complete!")

    # Salva modello finale
    save_path = os.path.join("checkpoints", "model_sequential_final.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")


    def log_training_info(args, epochs_done, info_path="submission/info.txt"):
        os.makedirs(os.path.dirname(info_path), exist_ok=True)

        # Se modello e info.txt non esistono => nuovo file
        model_path = os.path.join("checkpoints", "model_sequential_final.pth")
        new_file = not os.path.exists(info_path) or not os.path.exists(model_path)

        mode = 'w' if new_file else 'a'

        with open(info_path, mode) as f:
            f.write(f"Training session:\n")
            f.write(f"Epochs: {epochs_done}\n")
            f.write(f"Parameters:\n")
            f.write(f" - Criterion: {args.criterion}\n")
            f.write(f" - LR model: {args.lr_model}\n")
            f.write(f" - LR u: {args.lr_u}\n")
            f.write(f" - Lambda L3 Weight: {args.lambda_l3_weight}\n")
            f.write(f" - GNN type: {args.gnn}\n")
            f.write(f" - Dropout ratio: {args.drop_ratio}\n")
            f.write(f" - Num layers: {args.num_layer}\n")
            f.write(f" - Embedding dim: {args.emb_dim}\n")
            f.write(f" - Batch size: {args.batch_size}\n")
            f.write(f" - Epoch boost: {args.epoch_boost}\n")
            f.write(f"{'-'*40}\n")

    # Nel main sequential_training, dopo aver salvato il modello finale:
    log_training_info(args, epochs_done=args.epochs)
    print(f"Training info logged in submission/info.txt")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sequential training over datasets A-D")
    parser.add_argument("--device", type=int, default=0)
    # aggiungi altri parametri se vuoi
    args = parser.parse_args()

    # Set default params
    args.criterion = "gcod"
    args.lr_model = 0.001
    args.lr_u = 0.01
    args.lambda_l3_weight = 0.7
    args.num_checkpoints = 3
    args.gnn = "gin"
    args.drop_ratio = 0.5
    args.num_layer = 5
    args.emb_dim = 300
    args.batch_size = 32
    args.epochs = 1
    args.epoch_boost = 0

    sequential_training(args)
