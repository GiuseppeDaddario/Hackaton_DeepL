## Imports
import argparse
import logging
import os

import numpy as np
import torch.optim as optim
# Importazioni per TPU
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_geometric.loader import DataLoader
from tqdm import tqdm

## Modular imports
from source.evaluation import evaluate
# MODIFICA: Assicurati che add_zeros sia importato dalla sua posizione corretta.
# Se add_zeros è definito in source.loadData, importa da lì.
from source.loadData import GraphDataset, add_zeros  # Assumendo che add_zeros sia definito in source.loadData
from source.loss import ncodLoss, gcodLoss
from source.models import GNN  # Assicurati che GNN accetti 'train' in forward
from source.statistics import save_predictions, plot_training_progress
from source.train import train
from source.utils import set_seed

# Set the random seed
set_seed()

from torch_geometric.data import Batch, Data
from torch.utils.data.dataloader import default_collate
import torch # Assicurati che torch sia importato

def pyg_data_list_to_dict_collate(data_list):
    if not data_list:
        return {}

    valid_items = [item for item in data_list if isinstance(item, (Data, Batch)) and hasattr(item, 'keys')]
    if not valid_items:
        print(f"Warning: pyg_data_list_to_dict_collate riceve dati non validi o vuoti: {data_list}")
        return default_collate(data_list) # O {} se data_list era vuota ma valid_items no.

    try:
        temp_cpu_batch = Batch.from_data_list(valid_items)
    except Exception as e:
        print(f"Errore in Batch.from_data_list: {e}. Dati: {valid_items}")
        raise e

    final_batch_dict = {}

    attributes_to_check = ['x', 'edge_index', 'edge_attr', 'y', 'pos', 'face', 'original_idx']

    for attr_name in attributes_to_check:
        if hasattr(temp_cpu_batch, attr_name):
            value = getattr(temp_cpu_batch, attr_name)
            if torch.is_tensor(value):
                final_batch_dict[attr_name] = value

    if hasattr(temp_cpu_batch, 'batch') and torch.is_tensor(temp_cpu_batch.batch):
        final_batch_dict['batch'] = temp_cpu_batch.batch

    if hasattr(temp_cpu_batch, 'ptr') and torch.is_tensor(temp_cpu_batch.ptr):
        final_batch_dict['ptr'] = temp_cpu_batch.ptr

    # Aggiungi _num_graphs (intero)
    final_batch_dict['_num_graphs'] = temp_cpu_batch.num_graphs

    # Verifica se mancano attributi essenziali che dovrebbero essere tensori
    if 'x' not in final_batch_dict and temp_cpu_batch.num_nodes > 0: # Se ci sono nodi, ci si aspetta x
        print(f"Warning: 'x' (node features) non trovato o non è un tensore nel batch per MpDeviceLoader.")
        pass
    if 'edge_index' not in final_batch_dict:
        print(f"Warning: 'edge_index' non trovato o non è un tensore nel batch per MpDeviceLoader.")
        pass

    return final_batch_dict
def calculate_global_train_accuracy(model, full_train_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # MODIFICA: Rinomina data_batch a data_dict per chiarezza
        for data_dict in tqdm(full_train_loader, desc="Calculating global train accuracy (atrain)", unit="batch", leave=False, disable=not xm.is_master_ordinal()):
            if not data_dict: continue # Salta batch vuoti

            # MODIFICA: Estrai _num_graphs prima di creare il Batch
            num_graphs_val = data_dict.pop('_num_graphs', None)
            graphs = Batch(**data_dict) # I tensori sono già su device XLA
            if num_graphs_val is not None:
                graphs.num_graphs = num_graphs_val

            labels_int = graphs.y
            # MODIFICA: Passa train=False al modello
            outputs_logits, _, _ = model(graphs, train=False)
            _, predicted = torch.max(outputs_logits.data, 1)
            total += labels_int.size(0)
            correct += (predicted == labels_int.squeeze()).sum().item()
            xm.mark_step()
    model.train()
    if total == 0: return 0.0
    return correct / total


def _run_on_tpu(rank, args):
    device = xm.xla_device()
    # MODIFICA: is_master per operazioni I/O
    is_master = xm.is_master_ordinal() # True se rank == 0 o se xm.xrt_world_size() == 1

    # num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3 # Già gestito nell'if più avanti
    num_dataset_classes = 6 # !! CONFIGURA QUESTO VALORE CORRETTAMENTE !!
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # MODIFICA: Stampa e logging solo dal master
    if is_master:
        print(f"Master (Rank {rank}): Building the model...")
        logging.info(f"Master (Rank {rank}): Building the model...")

    # Definizione Modello (invariata, ma assicurati che GNN e le sue sub-classi siano corrette)
    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = num_dataset_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = num_dataset_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = num_dataset_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = num_dataset_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        # Questo errore dovrebbe essere sollevato da tutti i processi
        raise ValueError(f'Rank {rank}: Invalid GNN type: {args.gnn}')

    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr_model, weight_decay=1e-4)

    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)

    # MODIFICA: Configurazione logging solo per il master
    if is_master:
        os.makedirs(logs_folder, exist_ok=True)
        log_file_master = os.path.join(logs_folder, "training_master.log") # Log principale per il master
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            filename=log_file_master,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s', # Formato più dettagliato
            force=True
        )
        # Aggiungi StreamHandler per output console solo per il master
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
        logging.info(f"Logging configured for master. Script arguments: {args}")
        logging.info(f"PyTorch XLA version: {torch_xla.__version__}")

    checkpoint_path_best = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder_epochs = os.path.join(script_dir, "checkpoints", test_dir_name)
    if is_master:
        os.makedirs(checkpoints_folder_epochs, exist_ok=True)

    if os.path.exists(checkpoint_path_best) and not args.train_path:
        if is_master: logging.info(f"Loading pre-trained best model from {checkpoint_path_best}")
        model.load_state_dict(torch.load(checkpoint_path_best, map_location="cpu"))
        model = model.to(device) # Sposta il modello sul device XLA
        if is_master: logging.info("Loaded best model to XLA device.")

    if args.train_path:
        if is_master: logging.info("Preparing train dataset...")
        # MODIFICA: Passa emb_dim a add_zeros se necessario per definire la dimensione delle feature
        # Assicurati che GraphDataset e add_zeros gestiscano correttamente la creazione di `data.x`
        train_dataset = GraphDataset(args.train_path,
                                     transform=lambda data: add_zeros(data, node_feature_dim=args.emb_dim), args_emb_dim=args.emb_dim)

        if is_master: logging.info(f"Train dataset length: {len(train_dataset)}")
        if len(train_dataset) == 0:
            # Questo errore dovrebbe essere sollevato da tutti i processi o almeno dal master per fermare l'esecuzione
            logging.error("Train dataset is empty. Check the path and data.")
            raise ValueError("Train dataset is empty.")

        if is_master: logging.info("Loading train dataset into DataLoader...")
        # MODIFICA: Aggiunto num_workers e drop_last per consistenza e distributed training
        # drop_last è importante se il numero di campioni non è divisibile per (batch_size * world_size)
        train_loader_for_batches = pl.MpDeviceLoader(
            DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                       collate_fn=pyg_data_list_to_dict_collate,
                       num_workers=args.num_workers,
                       drop_last=True if xm.xrt_world_size() > 1 else False),
            device
        )

        full_train_loader_for_atrain = None
        if args.criterion in ["ncod", "gcod"]:
            if is_master: logging.info("Preparing full train loader for atrain calculation...")
            full_train_loader_for_atrain = pl.MpDeviceLoader(
                DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=pyg_data_list_to_dict_collate,
                           num_workers=args.num_workers,
                           drop_last=True if xm.xrt_world_size() > 1 else False),
                device
            )

        if is_master: logging.info(f"Initializing loss function: {args.criterion}")

        # MODIFICA: Estrazione y_values_numpy più robusta e solo dal master (se necessario per inizializzazione loss)
        # Se NCOD/GCOD sono inizializzati allo stesso modo su tutti i rank, non serve is_master qui.
        # Se la loss deve avere gli stessi parametri iniziali (es. 'u'), considera di inizializzarla
        # sul master e poi distribuire i parametri, o assicurati che l'inizializzazione sia deterministica.
        y_values_list = []
        if args.criterion in ["ncod", "gcod"]: # Solo se serve per queste loss
            if is_master: logging.info("Extracting y_values for NCOD/GCOD initialization...")
            for i in range(len(train_dataset)):
                # Ottieni l'oggetto Data base, la trasformazione è già applicata da GraphDataset
                # o dal wrapper lambda nel DataLoader.
                # Per coerenza, è meglio che GraphDataset.get() restituisca l'oggetto Data già trasformato.
                # Altrimenti, applica la transform qui se non è già fatto.
                data_item = train_dataset.get(i) # get() dovrebbe restituire un Data object.

                # Se train_dataset.get() non applica la transform, applicala:
                # if train_dataset.transform:
                #    data_item = train_dataset.transform(data_item)

                if data_item.y is not None and data_item.y.numel() > 0:
                    y_values_list.append(data_item.y.item()) # Assumendo y scalare
                else:
                    if is_master: logging.warning(f"Sample {i} in train_dataset has no valid y label. Skipping for NCOD/GCOD y_values.")

            y_values_numpy = np.array(y_values_list)
            if is_master and len(y_values_numpy) == 0 :
                logging.error("No valid y labels found. NCOD/GCOD cannot be initialized correctly without sample_labels.")
                # Potrebbe essere un errore fatale per NCOD/GCOD
                # raise ValueError("No y_labels for NCOD/GCOD.")
            if is_master and len(y_values_numpy) != len(y_values_list): # Verifica la coerenza se alcuni sono stati saltati
                logging.warning(f"NCOD/GCOD: Using {len(y_values_numpy)} labels out of {len(train_dataset)} total samples due to missing y.")
        else:
            y_values_numpy = np.array([]) # Array vuoto se non serve

        loss_function_obj = None
        optimizer_loss_params = None

        # Inizializzazione Loss
        if args.criterion == "ncod":
            # Assicurati che num_examp corrisponda a y_values_numpy se la loss lo usa
            loss_function_obj = ncodLoss(
                sample_labels=y_values_numpy, device=device, num_examp=len(y_values_numpy),
                num_classes=num_dataset_classes, ratio_consistency=0, ratio_balance=0,
                encoder_features=args.emb_dim, total_epochs=args.epochs
            )
            # Controlla se la loss ha parametri ottimizzabili
            if list(loss_function_obj.parameters()):
                optimizer_loss_params = optim.SGD(loss_function_obj.parameters(), lr=args.lr_u)
        elif args.criterion == "gcod":
            loss_function_obj = gcodLoss(
                sample_labels_numpy=y_values_numpy, device=device, num_examp=len(y_values_numpy),
                num_classes=num_dataset_classes, gnn_embedding_dim=args.emb_dim, total_epochs=args.epochs
            )
            if list(loss_function_obj.parameters()):
                optimizer_loss_params = optim.SGD(loss_function_obj.parameters(), lr=args.lr_u)
        elif args.criterion == "ce":
            loss_function_obj = torch.nn.CrossEntropyLoss() # .to(device) sarà fatto dopo
            # optimizer_loss_params rimane None
        else:
            raise ValueError(f"Unsupported criterion: {args.criterion}")

        # Sposta la loss sul device (se non è già un modulo nn.Module come CE che viene spostato con il modello)
        if hasattr(loss_function_obj, 'to') and callable(getattr(loss_function_obj, 'to')):
            loss_function_obj = loss_function_obj.to(device)


        num_epochs = args.epochs
        best_train_accuracy = 0.0 # Tracciato solo dal master
        train_losses_history = [] # Tracciato solo dal master
        train_accuracies_history = [] # Tracciato solo dal master

        # MODIFICA: Calcolo intervalli checkpoint più robusto
        if args.num_checkpoints is not None and args.num_checkpoints > 0:
            if args.num_checkpoints == 1:
                checkpoint_intervals = [num_epochs] # Salva solo alla fine
            else:
                # Arrotonda per evitare problemi con float
                checkpoint_intervals = [int(round((i + 1) * num_epochs / args.num_checkpoints)) for i in range(args.num_checkpoints)]
        else:
            checkpoint_intervals = [] # Nessun checkpoint intermedio

        atrain_global = 0.0 # Calcolato solo dal master o aggregato
        if is_master: logging.info("Starting training...")

        # MODIFICA: Sposta loss_fn_ce per il boost sul device
        loss_fn_ce_for_boost = torch.nn.CrossEntropyLoss().to(device)

        for epoch in range(num_epochs):
            if is_master and epoch < args.epoch_boost:
                logging.info(f"Epoch {epoch+1}/{num_epochs}: Boosting with CE loss.")

            # Calcolo atrain_global (potrebbe essere intensivo, valuta la frequenza)
            # Se in modalità distribuita (nprocs > 1), questo dovrebbe essere calcolato solo dal master
            # o i risultati aggregati. Con nprocs=1, è corretto.
            if args.criterion in ["ncod", "gcod"] and full_train_loader_for_atrain is not None:
                if is_master or xm.xrt_world_size() == 1 : # Solo il master calcola atrain
                    atrain_global = calculate_global_train_accuracy(model, full_train_loader_for_atrain, device)
                # Se nprocs > 1, atrain_global andrebbe trasmesso agli altri rank
                # Per semplicità con nprocs=1, questo è ok.
                # if xm.xrt_world_size() > 1:
                #    atrain_tensor = torch.tensor(atrain_global if is_master else 0.0, device=device)
                #    xm.broadcast_master_param(atrain_tensor) # xm.all_reduce con op='SUM' e poi dividi per world_size se calcolato da tutti
                #    atrain_global = atrain_tensor.item()


            # La funzione train gestisce xm.optimizer_step e xm.mark_step
            avg_batch_acc_epoch, epoch_loss_avg = train(
                atrain_global_value=atrain_global, train_loader=train_loader_for_batches,
                model=model, optimizer_model=optimizer_model, device=device, # device è ridondante se train_loader è MpDeviceLoader
                optimizer_loss_params=optimizer_loss_params, loss_function_obj=loss_function_obj,
                save_checkpoints=(is_master and (epoch + 1) in checkpoint_intervals), # Salva solo dal master
                checkpoint_path=os.path.join(checkpoints_folder_epochs, f"model_{test_dir_name}"), # Path base
                current_epoch=epoch, criterion_type=args.criterion,
                num_classes_dataset=num_dataset_classes,
                lambda_l3_weight=args.lambda_l3_weight if args.criterion == "gcod" else 0.0,
                epoch_boost=args.epoch_boost, loss_fn_ce=loss_fn_ce_for_boost, is_tpu=True
            )

            # MODIFICA: Logging, stampa e aggiornamento storico solo dal master
            if is_master:
                atrain_log_str = f"{atrain_global:.4f}" if args.criterion in ['ncod', 'gcod'] else 'N/A'
                epoch_info_msg = (f"Epoch {epoch + 1}/{num_epochs}, Avg Batch Train Acc: {avg_batch_acc_epoch:.2f}%, "
                                  f"Epoch Train Loss: {epoch_loss_avg:.4f}, Atrain: {atrain_log_str}")
                logging.info(epoch_info_msg)
                # print(epoch_info_msg) # Potrebbe essere ridondante se il logger ha StreamHandler

                train_losses_history.append(epoch_loss_avg)
                train_accuracies_history.append(avg_batch_acc_epoch)

                if avg_batch_acc_epoch > best_train_accuracy:
                    best_train_accuracy = avg_batch_acc_epoch
                    # Usa xm.save per TPU, fatto solo dal master
                    xm.save(model.state_dict(), checkpoint_path_best)
                    logging.info(f"Best model (avg batch train acc: {best_train_accuracy:.2f}%) updated and saved: {checkpoint_path_best}")

            # Barriera per sincronizzare tutti i processi alla fine di ogni epoca se si usa il training distribuito
            if xm.xrt_world_size() > 1:
                xm.rendezvous(f'epoch_{epoch+1}_completed_لاک') # Usare una stringa univoca


        if is_master:
            plot_training_progress(train_losses_history, train_accuracies_history, os.path.join(logs_folder, "plots"))
            logging.info("Training completed.")

    # Barriera prima della predizione se il training è avvenuto e si usa il training distribuito
    if args.train_path and xm.xrt_world_size() > 1:
        xm.rendezvous('training_لاک_completed_before_predict')

    # MODIFICA: Sezione predict eseguita solo dal master
    if args.predict == 1 and is_master:
        if not os.path.exists(checkpoint_path_best):
            logging.error(f"Best model checkpoint not found at {checkpoint_path_best}. Cannot perform prediction.")
            return # Esce solo il master

        logging.info("Preparing test dataset for prediction...")
        # MODIFICA: Applica la trasformazione anche al test_dataset
        test_dataset = GraphDataset(args.test_path,
                                    transform=lambda data: add_zeros(data, node_feature_dim=args.emb_dim), args_emb_dim=args.emb_dim)

        if len(test_dataset) == 0:
            logging.error("Test dataset is empty. Cannot perform prediction.")
            return

        logging.info("Loading test dataset into DataLoader...")
        # MODIFICA: Aggiungi collate_fn e num_workers anche al test_loader
        test_loader = pl.MpDeviceLoader(
            DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                       collate_fn=pyg_data_list_to_dict_collate,
                       num_workers=args.num_workers),
            device
        )

        logging.info("Generating predictions for the test set...")
        model.load_state_dict(torch.load(checkpoint_path_best, map_location="cpu"))
        model = model.to(device) # Sposta il modello sul device XLA

        # evaluate dovrebbe gestire internamente la ricostruzione del Batch e xm.mark_step()
        predictions = evaluate(test_loader, model, device, calculate_accuracy=False, is_tpu=True)

        # MODIFICA: Salva le predizioni in un file specifico, non sovrascrivendo args.test_path
        predictions_output_file = os.path.join(logs_folder, f"predictions_{test_dir_name}.txt")
        save_predictions(predictions, predictions_output_file)
        logging.info(f"Predictions saved to {predictions_output_file}")


def main(args):
    # MODIFICA: Logica per nprocs basata su args.tpu_cores e ambiente XLA
    n_procs_to_spawn = 1 # Default a 1
    if 'XRT_TPU_CONFIG' in os.environ: # Siamo su un ambiente TPU
        available_tpu_cores = xm.xrt_world_size()
        if args.tpu_cores > available_tpu_cores:
            print(f"Warning: Requested tpu_cores ({args.tpu_cores}) > available XLA devices ({available_tpu_cores}). Using {available_tpu_cores} cores.")
            n_procs_to_spawn = available_tpu_cores
        elif args.tpu_cores <= 0: # Se 0 o negativo, usa tutti i disponibili
            print(f"Info: tpu_cores <= 0, using all available XLA devices: {available_tpu_cores}.")
            n_procs_to_spawn = available_tpu_cores
        else: # Usa il numero specificato se è valido
            n_procs_to_spawn = args.tpu_cores
    else: # Non siamo su TPU (es. esecuzione locale CPU/GPU)
        if args.tpu_cores > 1:
            print(f"Warning: tpu_cores={args.tpu_cores} but not on XLA environment. Forcing nprocs=1 for non-TPU execution.")
        n_procs_to_spawn = 1 # Su CPU/GPU, solitamente si usa 1 processo per xmp.spawn (o non si usa xmp affatto)

    # Stampa solo una volta, non per ogni potenziale processo spawnato
    print(f"Attempting to spawn {n_procs_to_spawn} XLA process(es). Master ordinal will handle most I/O.")

    # Se nprocs_to_spawn è 1, MpDeviceLoader gestirà l'uso dei core per il caricamento dati.
    # Se nprocs_to_spawn > 1, si attiva il data parallelism (ogni processo ha una copia del modello).
    xmp.spawn(_run_on_tpu, args=(args,), nprocs=n_procs_to_spawn, start_method='fork')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--criterion", type=str, default="gcod", choices=["ce", "ncod", "gcod"], help="Type of loss to use (ce, ncod, gcod)")
    parser.add_argument("--lr_model", type=float, default=0.001, help="Learning rate for GNN model (default: 0.001)")
    parser.add_argument("--lr_u", type=float, default=0.01, help="Learning rate for u parameters in NCOD/GCOD (default: 0.01)") # Coerente con codice
    parser.add_argument("--lambda_l3_weight", type=float, default=0.7, help="Weight for L3 in GCOD (default: 0.7)") # Coerente con codice
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--predict", type=int, default=1, choices=[0,1], help="Save predictions (1=yes, 0=no)")
    parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of intermediate checkpoints (0 for none, 1 for end only).")
    # parser.add_argument('--device', type=int, default=0, help='unused for TPU') # Correttamente commentato
    parser.add_argument('--gnn', type=str, default='gin', help='GNN type (gin, gin-virtual, gcn, gcn-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='Dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='Number of GNN layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='Dimensionality of GNN hidden units / node features (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--epoch_boost', type=int, default=0, help='Epochs for CE loss before NCOD/GCOD (default: 0)')
    # MODIFICA: Aggiunto num_workers, default a 0 per TPU può essere più stabile
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers. Default 0 for TPU (main process loading).')
    parser.add_argument('--tpu_cores', type=int, default=1, help='Number of TPU cores for xmp.spawn (1=single XLA process, >1 for data parallel). Use 0 or -1 to use all available on TPU.')

    args = parser.parse_args()

    # MODIFICA: Commentato il rilevamento automatico di tpu_cores, lasciato alla logica in main()
    # if 'XRT_TPU_CONFIG' in os.environ:
    #    args.tpu_cores = xm.xrt_world_size()
    # else:
    #    args.tpu_cores = 1

    # Validazione num_workers su TPU (opzionale, ma può prevenire problemi)
    if 'XRT_TPU_CONFIG' in os.environ and args.num_workers > 0:
        print(f"Warning: num_workers ({args.num_workers}) > 0 on TPU. This can sometimes lead to issues with 'fork'. "
              "If errors occur, try setting --num_workers 0.")
        # Per forzare: args.num_workers = 0

    main(args)