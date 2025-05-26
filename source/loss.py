import random
import torch.nn as nn

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


# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 1e-7 # Mantieni il tuo valore di epsilon

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