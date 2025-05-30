import random
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

cross_entropy_val = nn.CrossEntropyLoss
mean = 1e-8
std  = 1e-9
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
        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32, device=device))
        torch.nn.init.normal_(self.u, mean=0.0, std=1e-3)
        with torch.no_grad():
            self.u.data.clamp_(min=eps, max=1.0 - eps)

        self.sample_labels_numpy = sample_labels_numpy
        self.gnn_embedding_dim = gnn_embedding_dim

        self.prev_gnn_embeddings = torch.rand((num_examp, gnn_embedding_dim), device=device) * 0.01
        self.class_centroids = torch.rand((num_classes, gnn_embedding_dim), device=device) * 0.01
        self.first_epoch_or_batch_init_centroid = True

        self.class_indices_bins = []
        for i in range(num_classes):
            self.class_indices_bins.append(np.where(self.sample_labels_numpy == i)[0])

    def _update_centroids(self):
        percent_cleanest = 50
        with torch.no_grad():
            for i in range(self.num_classes):
                class_specific_indices = self.class_indices_bins[i]
                if len(class_specific_indices) == 0:
                    self.class_centroids[i] = torch.zeros(self.gnn_embedding_dim, device=self.device)
                    continue

                valid_pytorch_indices = torch.tensor(class_specific_indices, device=self.device, dtype=torch.long)
                if valid_pytorch_indices.max() >= self.u.shape[0]:
                    print(f"WARN: Indice {valid_pytorch_indices.max()} fuori range per u (shape {self.u.shape}) nella classe {i}. Salto aggiornamento centroide.")
                    self.class_centroids[i] = torch.zeros(self.gnn_embedding_dim, device=self.device)
                    continue


                class_u_values = self.u.detach()[valid_pytorch_indices].squeeze()
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
            safe_centroids_norm = torch.where(centroids_norm == 0, torch.ones_like(centroids_norm), centroids_norm)
            centroids_normalized = self.class_centroids / (safe_centroids_norm + eps)

            similarities = torch.mm(batch_emb_normalized, centroids_normalized.t())
            soft_labels = F.softmax(similarities, dim=1)
            return soft_labels


    def calculate_loss_components(self, batch_original_indices, gnn_logits_batch, true_labels_batch_one_hot,
                                  gnn_embeddings_batch, batch_iter_num, current_epoch, atrain_overall_accuracy):

        pytorch_batch_indices = torch.tensor(batch_original_indices, device=self.device, dtype=torch.long)

        with torch.no_grad():
            if torch.isnan(gnn_embeddings_batch).any() or torch.isinf(gnn_embeddings_batch).any():
                print(f"ERROR: NaN/Inf in gnn_embeddings_batch at Epoch {current_epoch}, Batch {batch_iter_num}")
            self.prev_gnn_embeddings[pytorch_batch_indices] = gnn_embeddings_batch.detach()

        if self.first_epoch_or_batch_init_centroid and current_epoch == 0 and batch_iter_num == 0:
            self._update_centroids()
            self.first_epoch_or_batch_init_centroid = False
        elif current_epoch > 0 and batch_iter_num == 0: # Aggiorna i centroidi all'inizio di ogni epoca (dopo la prima)
            self._update_centroids()

        u_batch = self.u[pytorch_batch_indices]

        # --- L1: Modified Cross-Entropy Loss ---
        soft_labels_target_batch = self._get_soft_labels(gnn_embeddings_batch)
        logit_modification = atrain_overall_accuracy * u_batch * true_labels_batch_one_hot
        modified_logits = gnn_logits_batch + logit_modification
        log_probs_l1 = F.log_softmax(modified_logits, dim=1)
        l1_loss_per_sample = -torch.sum(soft_labels_target_batch * log_probs_l1, dim=1)
        l1_loss = torch.mean(l1_loss_per_sample)

        # --- L2: Loss for updating 'u' ---
        with torch.no_grad():
            pred_indices_l2 = torch.argmax(gnn_logits_batch.detach(), dim=1)
            predicted_labels_one_hot_l2 = F.one_hot(pred_indices_l2, num_classes=self.num_classes).float()
        term_for_u_update = predicted_labels_one_hot_l2 + u_batch * true_labels_batch_one_hot - true_labels_batch_one_hot
        l2_loss = torch.mean(torch.sum(term_for_u_update**2, dim=1)) / self.num_classes

        # --- L3: Regularization ---
        with torch.no_grad():
            u_batch_detached_for_l3 = u_batch.detach()
            u_batch_squeezed_clamped_for_l3 = torch.clamp(u_batch_detached_for_l3.squeeze(dim=1), min=eps, max=1.0 - eps)
            log_of_clamped_u_for_l3 = torch.log(u_batch_squeezed_clamped_for_l3)
            u_transformed_for_l3 = torch.sigmoid(-log_of_clamped_u_for_l3)
            u_transformed_for_l3 = torch.clamp(u_transformed_for_l3, min=eps, max=1.0 - eps)

        prob_true_class_from_logits = torch.sum(F.softmax(gnn_logits_batch, dim=1) * true_labels_batch_one_hot, dim=1)
        prob_true_class = torch.clamp(prob_true_class_from_logits, min=eps, max=1.0 - eps)

        log_prob_true_class = torch.log(prob_true_class)
        log_1_minus_prob_true_class = torch.log(1.0 - prob_true_class)

        log_u_transformed = torch.log(u_transformed_for_l3)
        log_1_minus_u_transformed = torch.log(1.0 - u_transformed_for_l3)

        term1_dkl = prob_true_class * (log_prob_true_class - log_u_transformed)
        term2_dkl = (1.0 - prob_true_class) * (log_1_minus_prob_true_class - log_1_minus_u_transformed)

        dkl_per_sample = term1_dkl + term2_dkl
        dkl_per_sample = torch.nan_to_num(dkl_per_sample, nan=0.0, posinf=0.0, neginf=0.0)
        l3_loss = torch.mean(dkl_per_sample) * (1.0 - atrain_overall_accuracy)

        # --- DEBUG BLOCK ---
        if torch.isnan(l1_loss).any() or torch.isinf(l1_loss).any() or \
                torch.isnan(l2_loss).any() or torch.isinf(l2_loss).any() or \
                torch.isnan(l3_loss).any() or torch.isinf(l3_loss).any():
            print(f"\n--- DEBUG GCOD LOSS COMPONENTS NaN/Inf Detected at Epoch {current_epoch}, Batch {batch_iter_num} ---")
            print(f"atrain_overall_accuracy: {atrain_overall_accuracy:.4f}")
            print(f"L1 Loss: {l1_loss.item() if not torch.isnan(l1_loss).any() else 'NaN/Inf'}")
            print(f"L2 Loss: {l2_loss.item() if not torch.isnan(l2_loss).any() else 'NaN/Inf'}")
            print(f"L3 Loss: {l3_loss.item() if not torch.isnan(l3_loss).any() else 'NaN/Inf'}")

        return l1_loss, l2_loss, l3_loss

    def forward(self, batch_original_indices, gnn_logits_batch, true_labels_batch_one_hot,
                gnn_embeddings_batch, batch_iter_num, current_epoch, atrain_overall_accuracy):

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
        pred_log_softmax = pred_logits.log_softmax(dim=self.dim) # (N, C)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred_log_softmax) # (N, C)
            if self.smoothing > 0:
                true_dist.fill_(self.smooth_value)
            true_dist.scatter_(1, target_indices.data.unsqueeze(1).long(), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred_log_softmax, dim=self.dim))