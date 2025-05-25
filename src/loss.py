import torch
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cross_entropy_val = nn.CrossEntropyLoss

mean = 1e-8
std  = 1e-9
class ncodLoss(nn.Module):
    def __init__(self, sample_labels,device, num_examp=50000, num_classes=100, ratio_consistency=0, ratio_balance=0,encoder_features=64,total_epochs=100):
        super(ncodLoss, self).__init__()

        self.num_classes = num_classes
        self.device = device
        self.USE_CUDA = torch.cuda.is_available()
        self.num_examp = num_examp
        self.total_epochs = total_epochs
        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance


        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.init_param(mean=mean,std=std)

        self.beginning = True
        self.prevSimilarity = torch.rand((num_examp, encoder_features),device=device)
        self.masterVector = torch.rand((num_classes, encoder_features),device=device)
        self.take = torch.zeros((num_examp, 1), device=device)
        self.dirchilet = torch.zeros((num_examp, 1), device=device)
        self.sample_labels = sample_labels
        self.bins = []

        for i in range(0, num_classes):
            self.bins.append(np.where(self.sample_labels == i)[0])
        self.shuffledbins = copy.deepcopy(self.bins)
        for sublist in self.shuffledbins:
            random.shuffle(sublist)

    def init_param(self, mean= 1e-8, std= 1e-9):
        torch.nn.init.normal_(self.u, mean=mean, std=std)


    def forward(self, index, outputs, label, out, flag,train_acc_cater,_):

        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)
            out1, out2 = torch.chunk(out, 2)
        else:
            output = outputs
            out1 = out

        eps = 1e-4

        u = self.u[index]



        if (flag == 0):
            if self.beginning:
                # percent = math.ceil((50 - (50 / self.total_epochs) * epoch) + 50)
                percent= 100
                for i in range(0, len(self.bins)):
                    class_u = self.u.detach()[self.bins[i]]
                    bottomK = int((len(class_u) / 100) * percent)
                    important_indexs = torch.topk(class_u, bottomK, largest=False, dim=0)[1]
                    self.masterVector[i] = torch.mean(self.prevSimilarity[self.bins[i]][important_indexs.view(-1)],
                                                      dim=0)

            masterVector_norm = self.masterVector.norm(p=2, dim=1, keepdim=True)
            masterVector_normalized = self.masterVector.div(masterVector_norm)
            self.masterVector_transpose = torch.transpose(masterVector_normalized, 0, 1)
            self.beginning = True

        self.prevSimilarity[index] = out1.detach()

        prediction = F.softmax(output, dim=1)

        out_norm = out1.detach().norm(p=2, dim=1, keepdim=True)
        out_normalized = out1.detach().div(out_norm)

        similarity = torch.mm(out_normalized, self.masterVector_transpose)
        similarity = similarity * label
        sim_mask = (similarity > 0.000).type(torch.float32)
        similarity = similarity * sim_mask

        u = u * label
        #train_acc_class = torch.sum((label*train_acc_class),dim=1).view(-1,1)
        prediction = torch.clamp((prediction + (train_acc_cater*u.detach())), min=eps, max=1.0)
        #added by me for the test purpose only
        # loss = torch.mean(-torch.sum((label) * torch.log(prediction), dim=1))

        loss = torch.mean(-torch.sum((similarity) * torch.log(prediction), dim=1))

        label_one_hot = self.soft_to_hard(output.detach())

        MSE_loss = F.mse_loss((label_one_hot + u), label, reduction='sum') / len(label)
        loss += MSE_loss
        self.take[index] = torch.sum((label_one_hot * label), dim=1).view(-1, 1)


        kl_loss = F.kl_div(F.log_softmax(torch.sum((output * label),dim=1)),F.softmax(-torch.log(self.u[index].detach().view(-1))))
        # kl_loss = F.kl_div(F.log_softmax(-torch.log(self.u[index].detach().view(-1))),
        #                    F.softmax(torch.sum((output * label), dim=1))
        #                    )
        # print('the kl loss is',kl_loss.item())
        loss += (1-train_acc_cater)*kl_loss
        # west_loss = torch_wasserstein_loss(F.log_softmax(-torch.log(self.u[index].detach().view(-1))),
        #                                    F.softmax(torch.sum((output * label), dim=1)))
        # print('the w loss is',west_loss.item())
        # loss += west_loss


        if self.ratio_balance > 0:
            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)

            avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)

            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))

            loss += self.ratio_balance * balance_kl

        if (len(outputs) > len(index)) and (self.ratio_consistency > 0):
            consistency_loss = self.consistency_loss( output, output2)

            loss += self.ratio_consistency * torch.mean(consistency_loss)


        return loss


    def consistency_loss(self, output1, output2):
        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).to(self.device).scatter_(1, (x.argmax(dim=1)).view(-1, 1),1)




######################################################
##                                                  ##
##           GCOD LOSS IMPLEMENTATION               ##
##                                                  ##
######################################################



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Small epsilon to prevent log(0) or division by zero
# Un valore di eps più grande potrebbe aiutare con la stabilità dei log,
# ma potrebbe anche mascherare problemi o alterare leggermente i calcoli.
# Prova con 1e-7 o 1e-6 se 1e-8 continua a dare problemi con i log.
eps = 1e-7

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
            self.u.data.clamp_(min=eps, max=1.0-eps)

        self.sample_labels_numpy = sample_labels_numpy
        self.gnn_embedding_dim = gnn_embedding_dim

        # Inizializzazione prev_gnn_embeddings e class_centroids
        self.prev_gnn_embeddings = torch.rand((num_examp, gnn_embedding_dim), device=device) * 0.01 # Scala per valori più piccoli
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
                    self.class_centroids[i] = torch.randn(self.gnn_embedding_dim, device=self.device) * 0.01
                    continue

                valid_pytorch_indices = torch.tensor(class_specific_indices, device=self.device, dtype=torch.long)
                class_u_values = self.u.detach()[valid_pytorch_indices].squeeze()
                class_embeddings = self.prev_gnn_embeddings[valid_pytorch_indices]

                if class_u_values.numel() == 0:
                    self.class_centroids[i] = torch.randn(self.gnn_embedding_dim, device=self.device) * 0.01
                    continue
                if class_u_values.ndim == 0: # Se c'è un solo campione, squeeze lo rende scalare
                    class_u_values = class_u_values.unsqueeze(0)

                num_to_take = max(1, int(len(class_u_values) * (percent_cleanest / 100.0)))

                # Assicurati che k in topk non sia maggiore del numero di elementi
                k_for_topk = min(num_to_take, len(class_u_values))
                if k_for_topk == 0 : # Se non ci sono u_values
                    print("Centroids reinitialized to random values due to empty class_u_values.")
                    self.class_centroids[i] = torch.randn(self.gnn_embedding_dim, device=self.device) * 0.01
                    continue

                _, top_k_indices_in_class_subset = torch.topk(class_u_values, k_for_topk, largest=False)
                clean_embeddings_for_class = class_embeddings[top_k_indices_in_class_subset]

                if clean_embeddings_for_class.numel() > 0:
                    self.class_centroids[i] = torch.mean(clean_embeddings_for_class, dim=0)
                elif class_embeddings.numel() > 0 : # Fallback se non ci sono "clean" samples
                    self.class_centroids[i] = torch.mean(class_embeddings, dim=0)
                else: # Fallback estremo
                    self.class_centroids[i] = torch.randn(self.gnn_embedding_dim, device=self.device) * 0.01


    def _get_soft_labels(self, current_gnn_embeddings_batch):
        with torch.no_grad():
            # Normalizzazione per stabilità
            batch_emb_norm = current_gnn_embeddings_batch.norm(p=2, dim=1, keepdim=True)
            # Evita divisione per zero se la norma è zero (embedding tutti a zero)
            batch_emb_normalized = current_gnn_embeddings_batch / (batch_emb_norm + eps)

            centroids_norm = self.class_centroids.norm(p=2, dim=1, keepdim=True)
            centroids_normalized = self.class_centroids / (centroids_norm + eps)

            similarities = torch.mm(batch_emb_normalized, centroids_normalized.t())
            soft_labels = F.softmax(similarities, dim=1)
            return soft_labels


    def forward(self, batch_original_indices, gnn_logits_batch, true_labels_batch_one_hot,
                gnn_embeddings_batch, batch_iter_num, current_epoch, atrain_overall_accuracy):

        pytorch_batch_indices = torch.tensor(batch_original_indices, device=self.device, dtype=torch.long)

        with torch.no_grad():
            # Controlla se gli embedding del GNN sono validi
            if torch.isnan(gnn_embeddings_batch).any() or torch.isinf(gnn_embeddings_batch).any():
                print(f"ERROR: NaN/Inf in gnn_embeddings_batch at Epoch {current_epoch}, Batch {batch_iter_num}")
            self.prev_gnn_embeddings[pytorch_batch_indices] = gnn_embeddings_batch.detach()

        if self.first_epoch_or_batch_init_centroid and current_epoch == 0 and batch_iter_num == 0:
            self._update_centroids()
            self.first_epoch_or_batch_init_centroid = False
        elif current_epoch > 0 and batch_iter_num == 0:
            self._update_centroids()

        u_batch = self.u[pytorch_batch_indices]

        # --- L1: Modified Cross-Entropy Loss (Eq 4) ---
        soft_labels_target_batch = self._get_soft_labels(gnn_embeddings_batch)
        logit_modification = atrain_overall_accuracy * u_batch * true_labels_batch_one_hot
        modified_logits = gnn_logits_batch + logit_modification
        log_probs_l1 = F.log_softmax(modified_logits, dim=1)
        l1_loss_per_sample = -torch.sum(soft_labels_target_batch * log_probs_l1, dim=1)
        l1_loss = torch.mean(l1_loss_per_sample)

        # --- L2: Loss for updating 'u' (Eq 6) ---
        with torch.no_grad():
            pred_indices = torch.argmax(gnn_logits_batch, dim=1)
            predicted_labels_one_hot = F.one_hot(pred_indices, num_classes=self.num_classes).float()
        term_for_u_update = predicted_labels_one_hot + u_batch * true_labels_batch_one_hot - true_labels_batch_one_hot
        l2_loss = torch.mean(torch.sum(term_for_u_update**2, dim=1)) / self.num_classes

        # --- L3: Regularization for 'u' (Eq 8) ---
        with torch.no_grad(): # detach gnn_logits_batch per non influenzare i gradienti del modello tramite L3
            prob_true_class_from_logits = torch.sum(F.softmax(gnn_logits_batch.detach(), dim=1) * true_labels_batch_one_hot, dim=1)

        prob_true_class = torch.clamp(prob_true_class_from_logits, min=eps, max=1.0 - eps)

        # Clamp u_batch in [eps, 1.0 - eps] per stabilità dei log
        u_batch_squeezed_clamped_for_l3 = torch.clamp(u_batch.squeeze(dim=1), min=eps, max=1.0 - eps)

        # Calcolo di u_transformed secondo il paper: sigma(-log(u_B))
        # Se u_B è piccolo (vicino a eps), -log(u_B) è grande e positivo, sigma -> 1
        # Se u_B è grande (vicino a 1-eps), -log(u_B) è piccolo e positivo (vicino a -log(1)=0), sigma -> 0.5
        log_of_clamped_u = torch.log(u_batch_squeezed_clamped_for_l3)
        u_transformed = torch.sigmoid(-log_of_clamped_u)
        u_transformed = torch.clamp(u_transformed, min=eps, max=1.0 - eps) # Clamp anche il risultato

        # DKL(P||Q) = p*(log p - log q) + (1-p)*(log(1-p) - log(1-q))
        log_prob_true_class = torch.log(prob_true_class) # p
        log_1_minus_prob_true_class = torch.log(1.0 - prob_true_class) # 1-p

        log_u_transformed = torch.log(u_transformed) # q
        log_1_minus_u_transformed = torch.log(1.0 - u_transformed) # 1-q

        term1_dkl = prob_true_class * (log_prob_true_class - log_u_transformed)
        term2_dkl = (1.0 - prob_true_class) * (log_1_minus_prob_true_class - log_1_minus_u_transformed)

        dkl_per_sample = term1_dkl + term2_dkl
        dkl_per_sample = torch.nan_to_num(dkl_per_sample, nan=0.0, posinf=0.0, neginf=0.0) # Sostituisce NaN/Inf con 0.0

        l3_loss = torch.mean(dkl_per_sample) * (1.0 - atrain_overall_accuracy)

        lambda_l3 = 0.5
        total_loss = l1_loss + l2_loss + lambda_l3 * l3_loss

        # --- DEBUG BLOCK ---
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any() or \
                torch.isnan(l1_loss).any() or torch.isinf(l1_loss).any() or \
                torch.isnan(l2_loss).any() or torch.isinf(l2_loss).any() or \
                torch.isnan(l3_loss).any() or torch.isinf(l3_loss).any():
            print(f"\n--- DEBUG GCOD LOSS NaN/Inf Detected at Epoch {current_epoch}, Batch {batch_iter_num} ---")
            print(f"atrain_overall_accuracy: {atrain_overall_accuracy:.4f}")
            print(f"L1 Loss: {l1_loss.item():.4f}")
            print(f"L2 Loss: {l2_loss.item():.4f}")
            print(f"L3 Loss: {l3_loss.item():.4f}")
            print(f"Total Loss: {total_loss.item():.4f}")

            print(f"  Stats for u_batch (shape {u_batch.shape}): min={u_batch.min().item():.4e}, max={u_batch.max().item():.4e}, mean={u_batch.mean().item():.4e}, has_nan={torch.isnan(u_batch).any()}")
            print(f"  Stats for gnn_logits_batch (shape {gnn_logits_batch.shape}): min={gnn_logits_batch.min().item():.4f}, max={gnn_logits_batch.max().item():.4f}, mean={gnn_logits_batch.mean().item():.4f}, has_nan={torch.isnan(gnn_logits_batch).any()}")
            print(f"  Stats for prob_true_class (shape {prob_true_class.shape}): min={prob_true_class.min().item():.4f}, max={prob_true_class.max().item():.4f}, mean={prob_true_class.mean().item():.4f}, has_nan={torch.isnan(prob_true_class).any()}")
            print(f"  Stats for u_batch_squeezed_clamped_for_l3 (shape {u_batch_squeezed_clamped_for_l3.shape}): min={u_batch_squeezed_clamped_for_l3.min().item():.4f}, max={u_batch_squeezed_clamped_for_l3.max().item():.4f}, mean={u_batch_squeezed_clamped_for_l3.mean().item():.4f}, has_nan={torch.isnan(u_batch_squeezed_clamped_for_l3).any()}")
            print(f"  Stats for u_transformed (shape {u_transformed.shape}): min={u_transformed.min().item():.4f}, max={u_transformed.max().item():.4f}, mean={u_transformed.mean().item():.4f}, has_nan={torch.isnan(u_transformed).any()}")
            print(f"  Stats for dkl_per_sample (shape {dkl_per_sample.shape}): min={dkl_per_sample.min().item():.4f}, max={dkl_per_sample.max().item():.4f}, mean={dkl_per_sample.mean().item():.4f}, has_nan={torch.isnan(dkl_per_sample).any()}")

        # --- END DEBUG BLOCK ---

        return total_loss