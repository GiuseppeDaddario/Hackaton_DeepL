import argparse
import os
import torch
from torch_geometric.loader import DataLoader

# Importa le tue classi e funzioni custom
from source.loadData import GraphDataset
from source.models import GINENet # Assicurati che sia il GINENet corretto
from source.dataLoader import AddNodeFeatures # Per creare feature dei nodi se necessario
from source.statistics import save_predictions
from source.evaluation import evaluate_model # Assicurati che questa funzione sia corretta
from source.utils import set_seed

def predict_with_model(args):
    set_seed(args.seed) # Imposta il seed all'inizio
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    print(f"Using device: {device}")

    # --- 1. Inferenza Parametri per il Modello ---
    # Per la predizione, abbiamo bisogno di sapere num_classes e potenzialmente num_edge_features
    # e node_feat_dim_initial per costruire il modello esattamente come durante il training.
    # Questi valori DEVONO corrispondere a quelli usati per addestrare il checkpoint.

    # Se il checkpoint è stato addestrato con main_gineconv.py, questi sono i valori
    # che sarebbero stati usati o inferiti. Li prendiamo dagli args.
    num_node_features_initial = args.node_feat_dim_initial # Es. 1, se AddNodeFeatures crea feature da zero
    num_edge_features_resolved = args.num_edge_features_from_training # Nuovo arg per specificare questo
    num_dataset_classes = args.num_classes_from_training      # Nuovo arg per specificare questo

    if num_edge_features_resolved is None:
        print("ATTENZIONE: num_edge_features_from_training non specificato. Se il modello usa feature degli archi, questo potrebbe essere un problema.")
        print("Assumo 0 edge features se il modello lo permette.")
        num_edge_features_resolved = 0 # Fallback, ma è meglio specificarlo.

    if num_dataset_classes is None:
        raise ValueError("num_classes_from_training deve essere specificato per costruire il modello e per la valutazione.")

    print(f"Parametri per la costruzione del modello (DEVONO CORRISPONDERE AL TRAINING):")
    print(f"  Initial Node Feature Dim: {num_node_features_initial}")
    print(f"  Edge Feature Dim        : {num_edge_features_resolved}")
    print(f"  Number of Classes       : {num_dataset_classes}")

    # --- 2. Costruzione del Modello ---
    print("Building the model...")
    # Assicurati che i parametri passati a GINENet qui siano ESATTAMENTE
    # quelli con cui il modello nel checkpoint è stato creato.
    model = GINENet(
        in_channels=num_node_features_initial,
        hidden_channels=args.emb_dim,             # Deve corrispondere al training
        out_channels=num_dataset_classes,         # Deve corrispondere al training
        num_layers=args.num_layer,                # Deve corrispondere al training
        edge_dim=num_edge_features_resolved,      # Deve corrispondere al training
        dropout_rate=args.drop_ratio,             # Solitamente 0 per eval, ma il modello salvato ha la sua config.
        # Qui passiamo il valore usato nel training per coerenza strutturale,
        # model.eval() disabiliterà il dropout.
        graph_pooling=args.graph_pooling          # Deve corrispondere al training
    ).to(device)
    print("Model built.")

    # --- 3. Caricamento Checkpoint ---
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Assicurati che l'architettura del modello corrisponda a quella del checkpoint.")
        return
    model.eval() # Metti il modello in modalità valutazione

    # --- 4. Caricamento del Dataset di Test ---
    # È importante usare la STESSA trasformazione (es. AddNodeFeatures) usata durante il training
    # per le feature dei nodi, se il modello se le aspetta create in quel modo.
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

    # --- 5. Generazione Predizioni ---
    print("Generating predictions...")
    # evaluate_model dovrebbe restituire (avg_loss, accuracy, f1_score, all_preds, all_labels)
    # Per il test, avg_loss potrebbe non essere significativo se criterion_obj è None.
    # Ci interessano principalmente all_preds.
    try:
        eval_results = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            criterion_obj=None,       # Non calcoliamo la loss per la predizione pura
            criterion_type=args.criterion, # Può essere utile per alcuni tipi di valutazione, ma non per la loss qui
            num_classes_dataset=num_dataset_classes, # Potrebbe essere usato da evaluate_model per metriche
            lambda_l3_weight=0.0,     # Non rilevante se criterion_obj è None
            current_epoch_for_gcod=-1,# Non rilevante
            atrain_for_gcod=0.0,      # Non rilevante
            is_validation=False       # Indica che siamo in modalità test/predizione
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

    all_predictions = eval_results[3] # Il quarto elemento dovrebbe essere la lista delle predizioni
    # Potresti anche voler usare le etichette vere (eval_results[4]) se disponibili nel test set per calcolare metriche
    if eval_results[4] is not None and len(eval_results[4]) > 0:
        print(f"Test Set Metrics (se calcolate da evaluate_model): Loss: {eval_results[0]:.4f}, Acc: {eval_results[1]:.4f}, F1: {eval_results[2]:.4f}")


    if all_predictions is None or len(all_predictions) == 0:
        print("Nessuna predizione generata o la lista delle predizioni è vuota.")
        return
    print(f"Generated {len(all_predictions)} predictions.")

    # --- 6. Salvataggio Predizioni ---
    # save_predictions si aspetta una lista/array di predizioni e un path di output
    # Il nome del file di output è gestito da save_predictions internamente o puoi specificarlo.
    # L'esempio originale di save_predictions usava test_path per derivare il nome del file di output.
    # Adattalo se necessario.
    output_prediction_filename = f"predictions_{os.path.basename(args.test_path).replace('.json.gz', '')}_{args.gnn_type}.txt"
    output_predictions_dir = args.output_dir if hasattr(args, 'output_dir') else "./submissions" # Aggiungi output_dir agli args
    os.makedirs(output_predictions_dir, exist_ok=True)
    output_prediction_path = os.path.join(output_predictions_dir, output_prediction_filename)

    try:
        save_predictions(all_predictions, output_prediction_path) # Passa il path completo del file di output
        print(f"Predictions saved to: {output_prediction_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")

if __name__ == "__main__":
    # Correzione: args = argparse.Namespace(...)
    args = argparse.Namespace(
        checkpoint_path='/Users/giuseppedaddario/Downloads/model_A_best.pth',
        test_path='../datasets/A/test.json.gz', # Assicurati che questo path sia corretto per il tuo ambiente
        output_dir='/Users/giuseppedaddario/Desktop/submission',

        # Parametri per la costruzione del modello (DEVONO CORRISPONDERE AL TRAINING)
        gnn_type='gine',                           # Coerente con GINENet
        num_layer=5,                               # Esempio, deve corrispondere al training
        emb_dim=300,                               # Esempio, deve corrispondere al training
        drop_ratio=0.2,                            # Esempio, deve corrispondere al training
        graph_pooling='mean',                      # Esempio, deve corrispondere al training
        node_feat_dim_initial=1,                   # Esempio, deve corrispondere al training
        num_edge_features_from_training=7,         # <<< CORRETTO: NOME e VALORE ESEMPIO (modifica 7 se necessario, o 0)
        num_classes_from_training=6,               # <<< CORRETTO: NOME e VALORE ESEMPIO (modifica 6 se necessario)

        # Parametri di runtime e dataloader
        batch_size=32,
        device=0, # o -1 per CPU
        seed=777,
        num_workers=0, # Metti a 0 se hai problemi, specialmente su Windows o macOS per debug semplice

        # Altri argomenti che potrebbero essere usati da evaluate_model o altre parti
        criterion='ce', # Passato a evaluate_model, anche se non per la loss

    )

    # La validazione ora dovrebbe funzionare
    if args.gnn_type != 'gine':
        # Se usi argparse.ArgumentParser, useresti parser.error()
        # Qui, dato che crei Namespace manualmente, stampiamo e usciamo
        print("ERRORE: Questo script è specificamente per GINENet. --gnn_type deve essere 'gine'.")
        exit(1)

    # Il controllo per num_edge_features_from_training ora accederà all'attributo corretto.
    # Se imposti num_edge_features_from_training a un valore (es. 7 o 0), questo if non sarà mai True.
    # Se lo imposti a None di proposito, allora questo blocco si attiverà.
    if args.num_edge_features_from_training is None:
        print("WARNING: num_edge_features_from_training è None. Se il modello usa feature degli archi, questo potrebbe portare a errori o risultati errati. Assumendo 0 se il modello lo permette.")
        args.num_edge_features_from_training = 0

    predict_with_model(args)