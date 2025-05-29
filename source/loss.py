import random
import torch.nn as nn

cross_entropy_val = nn.CrossEntropyLoss

mean = 1e-8
std  = 1e-9

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 1e-7


# ==================================================================================
# GCOD loss class
# ==================================================================================
class gcodLoss(nn.Module):
    def __init__(self, sample_labels_numpy, device, num_examp, num_classes, gnn_embedding_dim, total_epochs):
        super(gcodLoss, self).__init__()

        self.num_classes = num_classes
        self.device = device
        self.num_examp = num_examp
        self.total_epochs = total_epochs

        # Inizializzazione di u (come da tua ultima versione)
        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32, device=device))
        torch.nn.init.normal_(self.u, mean=0.0, std=1e-3) # Inizializzazione suggerita dal paper GCOD per N(0, 10^-3)
        with torch.no_grad():
            self.u.data.clamp_(min=eps, max=1.0 - eps) # Clampa subito dopo l'inizializzazione

        self.sample_labels_numpy = sample_labels_numpy
        self.gnn_embedding_dim = gnn_embedding_dim

        self.prev_gnn_embeddings = torch.rand((num_examp, gnn_embedding_dim), device=device) * 0.01
        self.class_centroids = torch.rand((num_classes, gnn_embedding_dim), device=device) * 0.01
        self.first_epoch_or_batch_init_centroid = True

        self.class_indices_bins = []
        for i in range(num_classes):
            self.class_indices_bins.append(np.where(self.sample_labels_numpy == i)[0])

    def _update_centroids(self):
        percent_cleanest = 50 # Puoi rendere questo un parametro
        with torch.no_grad():
            for i in range(self.num_classes):
                class_specific_indices = self.class_indices_bins[i]
                if len(class_specific_indices) == 0:
                    # Inizializzazione più stabile se una classe è vuota
                    self.class_centroids[i] = torch.zeros(self.gnn_embedding_dim, device=self.device)
                    continue

                valid_pytorch_indices = torch.tensor(class_specific_indices, device=self.device, dtype=torch.long)
                # Assicurati che gli indici siano validi
                if valid_pytorch_indices.max() >= self.u.shape[0]:
                    print(f"WARN: Indice {valid_pytorch_indices.max()} fuori range per u (shape {self.u.shape}) nella classe {i}. Salto aggiornamento centroide.")
                    self.class_centroids[i] = torch.zeros(self.gnn_embedding_dim, device=self.device) # o altra logica di fallback
                    continue


                class_u_values = self.u.detach()[valid_pytorch_indices].squeeze() # detach() è corretto qui
                class_embeddings = self.prev_gnn_embeddings[valid_pytorch_indices]

                if class_u_values.numel() == 0:
                    self.class_centroids[i] = torch.zeros(self.gnn_embedding_dim, device=self.device)
                    continue
                if class_u_values.ndim == 0:
                    class_u_values = class_u_values.unsqueeze(0)

                num_to_take = max(1, int(len(class_u_values) * (percent_cleanest / 100.0)))
                k_for_topk = min(num_to_take, len(class_u_values))

                if k_for_topk == 0:
                    self.class_centroids[i] = torch.zeros(self.gnn_embedding_dim, device=self.device)
                    continue

                # largest=False per prendere i più piccoli (più "puliti" se u è interpretato come rumore)
                _, top_k_indices_in_class_subset = torch.topk(class_u_values, k_for_topk, largest=False)
                clean_embeddings_for_class = class_embeddings[top_k_indices_in_class_subset]

                if clean_embeddings_for_class.numel() > 0:
                    self.class_centroids[i] = torch.mean(clean_embeddings_for_class, dim=0)
                elif class_embeddings.numel() > 0:
                    self.class_centroids[i] = torch.mean(class_embeddings, dim=0)
                else:
                    self.class_centroids[i] = torch.zeros(self.gnn_embedding_dim, device=self.device)


    def _get_soft_labels(self, current_gnn_embeddings_batch):
        with torch.no_grad():
            batch_emb_norm = current_gnn_embeddings_batch.norm(p=2, dim=1, keepdim=True)
            batch_emb_normalized = current_gnn_embeddings_batch / (batch_emb_norm + eps)

            centroids_norm = self.class_centroids.norm(p=2, dim=1, keepdim=True)
            # Gestisci il caso di centroidi a zero (norma zero) per evitare NaN
            safe_centroids_norm = torch.where(centroids_norm == 0, torch.ones_like(centroids_norm), centroids_norm)
            centroids_normalized = self.class_centroids / (safe_centroids_norm + eps)


            similarities = torch.mm(batch_emb_normalized, centroids_normalized.t())
            soft_labels = F.softmax(similarities, dim=1)
            return soft_labels


    def calculate_loss_components(self, batch_original_indices, gnn_logits_batch, true_labels_batch_one_hot,
                                  gnn_embeddings_batch, batch_iter_num, current_epoch, atrain_overall_accuracy):
        # Questo metodo calcola e restituisce L1, L2, L3. Il forward chiamerà questo.

        pytorch_batch_indices = torch.tensor(batch_original_indices, device=self.device, dtype=torch.long)

        with torch.no_grad():
            if torch.isnan(gnn_embeddings_batch).any() or torch.isinf(gnn_embeddings_batch).any():
                print(f"ERROR: NaN/Inf in gnn_embeddings_batch at Epoch {current_epoch}, Batch {batch_iter_num}")
                # Potresti voler restituire valori di loss non validi o zero per evitare ulteriori errori
            self.prev_gnn_embeddings[pytorch_batch_indices] = gnn_embeddings_batch.detach()

        if self.first_epoch_or_batch_init_centroid and current_epoch == 0 and batch_iter_num == 0:
            self._update_centroids()
            self.first_epoch_or_batch_init_centroid = False
        elif current_epoch > 0 and batch_iter_num == 0: # Aggiorna i centroidi all'inizio di ogni epoca (dopo la prima)
            self._update_centroids()

        # u_batch deve essere recuperato QUI, poiché self.u potrebbe essere stato aggiornato
        # nel ciclo di training precedente (se batch_iter_num > 0).
        # Se self.u è un Parameter, l'aggiornamento è gestito da optimizer_loss_params.
        u_batch = self.u[pytorch_batch_indices]

        # --- L1: Modified Cross-Entropy Loss (Eq 4) ---
        soft_labels_target_batch = self._get_soft_labels(gnn_embeddings_batch)
        # Rendi gnn_logits_batch un clone che richiede gradiente se non lo è già
        # Questo è importante se gnn_logits_batch viene da un model.eval() o with torch.no_grad()
        # Ma qui, gnn_logits_batch viene direttamente dal forward del modello in modalità train, quindi dovrebbe essere ok.
        logit_modification = atrain_overall_accuracy * u_batch * true_labels_batch_one_hot # u_batch qui traccia i gradienti per L1 rispetto a u
        modified_logits = gnn_logits_batch + logit_modification
        log_probs_l1 = F.log_softmax(modified_logits, dim=1)
        l1_loss_per_sample = -torch.sum(soft_labels_target_batch * log_probs_l1, dim=1)
        l1_loss = torch.mean(l1_loss_per_sample)

        # --- L2: Loss for updating 'u' (Eq 6) ---
        # Il paper dice ∇u L2, quindi L2 deve dipendere da u.
        # Il gradiente di (predicted_labels_one_hot - true_labels_batch_one_hot) rispetto a u è zero.
        # Il gradiente di (u_batch * true_labels_batch_one_hot) rispetto a u è true_labels_batch_one_hot.
        # Assicurati che gnn_logits_batch NON tracci gradienti per il calcolo di L2 rispetto a u.
        with torch.no_grad(): # Le predizioni del modello sono "fisse" per il calcolo di L2
            pred_indices_l2 = torch.argmax(gnn_logits_batch.detach(), dim=1)
            predicted_labels_one_hot_l2 = F.one_hot(pred_indices_l2, num_classes=self.num_classes).float()

        # u_batch è il Parameter, quindi i gradienti fluiranno correttamente.
        term_for_u_update = predicted_labels_one_hot_l2 + u_batch * true_labels_batch_one_hot - true_labels_batch_one_hot
        l2_loss = torch.mean(torch.sum(term_for_u_update**2, dim=1)) / self.num_classes

        # --- L3: Regularization (Eq 8) ---
        # Il paper dice ∇θ L3, quindi L3 deve dipendere da θ (gnn_logits_batch).
        # u_batch qui dovrebbe essere .detach() perché L3 non deve aggiornare u.
        with torch.no_grad(): # u_batch è considerato costante per L3 per l'aggiornamento di theta
            u_batch_detached_for_l3 = u_batch.detach()
            u_batch_squeezed_clamped_for_l3 = torch.clamp(u_batch_detached_for_l3.squeeze(dim=1), min=eps, max=1.0 - eps)
            log_of_clamped_u_for_l3 = torch.log(u_batch_squeezed_clamped_for_l3)
            u_transformed_for_l3 = torch.sigmoid(-log_of_clamped_u_for_l3)
            u_transformed_for_l3 = torch.clamp(u_transformed_for_l3, min=eps, max=1.0 - eps)

        # gnn_logits_batch qui traccia i gradienti per L3 rispetto a θ
        prob_true_class_from_logits = torch.sum(F.softmax(gnn_logits_batch, dim=1) * true_labels_batch_one_hot, dim=1)
        prob_true_class = torch.clamp(prob_true_class_from_logits, min=eps, max=1.0 - eps)

        log_prob_true_class = torch.log(prob_true_class)
        log_1_minus_prob_true_class = torch.log(1.0 - prob_true_class)

        log_u_transformed = torch.log(u_transformed_for_l3) # Usa la versione detached
        log_1_minus_u_transformed = torch.log(1.0 - u_transformed_for_l3) # Usa la versione detached

        term1_dkl = prob_true_class * (log_prob_true_class - log_u_transformed)
        term2_dkl = (1.0 - prob_true_class) * (log_1_minus_prob_true_class - log_1_minus_u_transformed)

        dkl_per_sample = term1_dkl + term2_dkl
        dkl_per_sample = torch.nan_to_num(dkl_per_sample, nan=0.0, posinf=0.0, neginf=0.0)
        l3_loss = torch.mean(dkl_per_sample) * (1.0 - atrain_overall_accuracy) # Ponderazione come da paper

        # --- DEBUG BLOCK (opzionale, ma utile) ---
        if torch.isnan(l1_loss).any() or torch.isinf(l1_loss).any() or \
                torch.isnan(l2_loss).any() or torch.isinf(l2_loss).any() or \
                torch.isnan(l3_loss).any() or torch.isinf(l3_loss).any():
            print(f"\n--- DEBUG GCOD LOSS COMPONENTS NaN/Inf Detected at Epoch {current_epoch}, Batch {batch_iter_num} ---")
            print(f"atrain_overall_accuracy: {atrain_overall_accuracy:.4f}")
            print(f"L1 Loss: {l1_loss.item() if not torch.isnan(l1_loss).any() else 'NaN/Inf'}")
            print(f"L2 Loss: {l2_loss.item() if not torch.isnan(l2_loss).any() else 'NaN/Inf'}")
            print(f"L3 Loss: {l3_loss.item() if not torch.isnan(l3_loss).any() else 'NaN/Inf'}")
            # ... (aggiungi altre stampe di debug se necessario)

        return l1_loss, l2_loss, l3_loss

    def forward(self, batch_original_indices, gnn_logits_batch, true_labels_batch_one_hot,
                gnn_embeddings_batch, batch_iter_num, current_epoch, atrain_overall_accuracy):
        # Il metodo forward ora è solo un wrapper se preferisci mantenere la chiamata standard
        # Oppure la funzione train chiamerà direttamente calculate_loss_components
        return self.calculate_loss_components(batch_original_indices, gnn_logits_batch, true_labels_batch_one_hot,
                                              gnn_embeddings_batch, batch_iter_num, current_epoch, atrain_overall_accuracy)



# ==================================================================================
# Label Smoothing Cross Entropy (args.criterion == 'ce')
# ==================================================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        if smoothing > 0:
            if classes <= 1:
                raise ValueError("Label smoothing requires num_classes > 1.")
            self.smooth_value = self.smoothing / (self.cls - 1)
        else:
            self.smooth_value = 0


    def forward(self, pred_logits, target_indices):
        """
        Args:
            pred_logits (Tensor): Logits dal modello (N, C).
            target_indices (Tensor): Etichette vere (N), indici interi.
        """
        pred_log_softmax = pred_logits.log_softmax(dim=self.dim) # (N, C)

        with torch.no_grad():
            # Crea la distribuzione target smooth
            true_dist = torch.zeros_like(pred_log_softmax) # (N, C)
            if self.smoothing > 0:
                true_dist.fill_(self.smooth_value)
            # Scatter assegna 'self.confidence' alle posizioni delle etichette vere
            true_dist.scatter_(1, target_indices.data.unsqueeze(1).long(), self.confidence)
            # Se smoothing è 0, confidence è 1.0 e smooth_value è 0, quindi è one-hot.

        # Calcola la KL divergence (o cross-entropy equivalente)
        # mean(sum(-true_dist * pred_log_softmax, dim))
        return torch.mean(torch.sum(-true_dist * pred_log_softmax, dim=self.dim))


    import numpy as np
from keras import backend as K
import tensorflow as tf


def cross_entropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def symmetric_cross_entropy(alpha, beta):
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        return alpha*tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.log(y_pred_1), axis = -1)) + beta*tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.log(y_true_2), axis = -1))
    return loss


def lsr(y_true, y_pred):
    epsilon = 0.1
    y_smoothed_true = y_true * (1 - epsilon - epsilon / 10.0)
    y_smoothed_true = y_smoothed_true + epsilon / 10.0

    y_pred_1 = tf.clip_by_value(y_pred, 1e-7, 1.0)

    return tf.reduce_mean(-tf.reduce_sum(y_smoothed_true * tf.log(y_pred_1), axis=-1))

def generalized_cross_entropy(y_true, y_pred):
    """
    2018 - nips - Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels.
    """
    q = 0.7
    t_loss = (1 - tf.pow(tf.reduce_sum(y_true * y_pred, axis=-1), q)) / q
    return tf.reduce_mean(t_loss)

def joint_optimization_loss(y_true, y_pred):
    """
    2018 - cvpr - Joint optimization framework for learning with noisy labels.
    """
    y_pred_avg = K.mean(y_pred, axis=0)
    p = np.ones(10, dtype=np.float32) / 10.
    l_p = - K.sum(K.log(y_pred_avg) * p)
    l_e = K.categorical_crossentropy(y_pred, y_pred)
    return K.categorical_crossentropy(y_true, y_pred) + 1.2 * l_p + 0.8 * l_e

def boot_soft(y_true, y_pred):
    """
    2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
    """
    beta = 0.95

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -K.sum((beta * y_true + (1. - beta) * y_pred) *
                  K.log(y_pred), axis=-1)


def boot_hard(y_true, y_pred):
    """
    2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
    """
    beta = 0.8

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    pred_labels = K.one_hot(K.argmax(y_pred, 1), num_classes=K.shape(y_true)[1])
    return -K.sum((beta * y_true + (1. - beta) * pred_labels) *
                  K.log(y_pred), axis=-1)

def forward(P):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach.
    """
    P = K.constant(P)
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        return -K.sum(y_true * K.log(K.dot(y_pred, P)), axis=-1)

    return loss

def backward(P):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach.
    """
    P_inv = K.constant(np.linalg.inv(P))

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        return -K.sum(K.dot(y_true, P_inv) * K.log(y_pred), axis=-1)

    return loss



class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=6, smoothing=0.0, device='cpu'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.smoothing = smoothing # Opzionale, per coerenza
        self.device = device

    def forward(self, y_pred_logits, y_true_indices):
        # y_pred_logits: (batch_size, num_classes) - raw outputs from model
        # y_true_indices: (batch_size) - integer class labels

        y_pred_probs = F.softmax(y_pred_logits, dim=-1)

        y_true_one_hot = F.one_hot(y_true_indices, num_classes=self.num_classes).float().to(self.device)

        if self.smoothing > 0:
            y_true_one_hot = y_true_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes

        # Termine 1: Alpha * CE(y_true, y_pred)
        y_pred_probs_clipped_1 = torch.clamp(y_pred_probs, 1e-7, 1.0)
        loss_ce = -torch.sum(y_true_one_hot * torch.log(y_pred_probs_clipped_1), dim=-1)

        # Termine 2: Beta * RCE(y_true, y_pred) = Beta * CE(y_pred, y_true_for_log)
        # La versione Keras clippa y_true_2 (le etichette one-hot) a [1e-4, 1.0] prima del log.
        # Questo serve per evitare log(0) e rende i target "morbidi" per il termine RCE.
        y_true_one_hot_clipped_for_log = torch.clamp(y_true_one_hot, 1e-4, 1.0) # Cruciale per replicare
        loss_rce = -torch.sum(y_pred_probs * torch.log(y_true_one_hot_clipped_for_log), dim=-1) # y_pred_probs non clippato qui

        return self.alpha * torch.mean(loss_ce) + self.beta * torch.mean(loss_rce)

class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, q=0.7, num_classes=6, smoothing=0.0, device='cpu'):
        super().__init__()
        self.q = q
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.device = device

    def forward(self, y_pred_logits, y_true_indices):
        y_pred_probs = F.softmax(y_pred_logits, dim=-1)
        y_true_one_hot = F.one_hot(y_true_indices, num_classes=self.num_classes).float().to(self.device)

        if self.smoothing > 0:
            y_true_one_hot = y_true_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes

        # tf.reduce_sum(y_true * y_pred, axis=-1) is the probability of the true class if y_true is one-hot
        # or more generally, the dot product if y_true is soft.
        # With y_true_one_hot, this is equivalent to gathering the predicted probability for the true class.
        pred_prob_for_true_class = torch.sum(y_true_one_hot * y_pred_probs, dim=-1)

        # Clamp to avoid issues with pow for very small pred_prob_for_true_class if q is high, or if it's exactly 0 or 1.
        pred_prob_for_true_class = torch.clamp(pred_prob_for_true_class, 1e-7, 1.0 - 1e-7)

        loss = (1 - torch.pow(pred_prob_for_true_class, self.q)) / self.q
        return torch.mean(loss)
