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
import math

# Small epsilon to prevent log(0) or division by zero
eps = 1e-8

class gcodLoss(nn.Module):
    def __init__(self, sample_labels_numpy, device, num_examp, num_classes, gnn_embedding_dim, total_epochs):
        super(gcodLoss, self).__init__()

        self.num_classes = num_classes
        self.device = device
        self.num_examp = num_examp
        self.total_epochs = total_epochs

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32, device=device))
        torch.nn.init.normal_(self.u, mean=1e-8, std=1e-9)

        self.sample_labels_numpy = sample_labels_numpy # 1D numpy array of integer labels
        self.gnn_embedding_dim = gnn_embedding_dim

        self.prev_gnn_embeddings = torch.rand((num_examp, gnn_embedding_dim), device=device)
        self.class_centroids = torch.rand((num_classes, gnn_embedding_dim), device=device)

        self.first_epoch_or_batch_init_centroid = True

        self.class_indices_bins = []
        for i in range(num_classes):
            self.class_indices_bins.append(np.where(self.sample_labels_numpy == i)[0])

    def _update_centroids(self):
        # Using only "cleaner" samples (small u) for centroids.
        percent_cleanest = 50 # Use 50% cleanest samples per class

        with torch.no_grad():
            for i in range(self.num_classes):
                class_specific_indices = self.class_indices_bins[i]
                if len(class_specific_indices) == 0:
                    self.class_centroids[i] = torch.randn(self.gnn_embedding_dim, device=self.device) * 0.01
                    continue

                # Ensure indices are valid for self.u and self.prev_gnn_embeddings
                valid_pytorch_indices = torch.tensor(class_specific_indices, device=self.device, dtype=torch.long)

                class_u_values = self.u.detach()[valid_pytorch_indices].squeeze()
                class_embeddings = self.prev_gnn_embeddings[valid_pytorch_indices]

                if len(class_u_values.shape) == 0: # single sample in class
                    class_u_values = class_u_values.unsqueeze(0)

                if class_u_values.numel() == 0: # no samples for this class in current view
                    self.class_centroids[i] = torch.randn(self.gnn_embedding_dim, device=self.device) * 0.01
                    continue

                num_to_take = max(1, int(len(class_u_values) * (percent_cleanest / 100.0)))

                _, top_k_indices_in_class_subset = torch.topk(class_u_values, min(num_to_take, len(class_u_values)), largest=False)

                clean_embeddings_for_class = class_embeddings[top_k_indices_in_class_subset]

                if clean_embeddings_for_class.numel() > 0:
                    self.class_centroids[i] = torch.mean(clean_embeddings_for_class, dim=0)
                else:
                    self.class_centroids[i] = torch.mean(class_embeddings, dim=0) if class_embeddings.numel() > 0 else torch.randn(self.gnn_embedding_dim, device=self.device) * 0.01

    def _get_soft_labels(self, current_gnn_embeddings_batch):
        with torch.no_grad():
            batch_emb_norm = current_gnn_embeddings_batch.norm(p=2, dim=1, keepdim=True) + eps
            batch_emb_normalized = current_gnn_embeddings_batch / batch_emb_norm

            centroids_norm = self.class_centroids.norm(p=2, dim=1, keepdim=True) + eps
            centroids_normalized = self.class_centroids / centroids_norm

            similarities = torch.mm(batch_emb_normalized, centroids_normalized.t())
            soft_labels = F.softmax(similarities, dim=1)
            return soft_labels

    # Modifica della firma per corrispondere alla tua chiamata:
    # index_run, outputs, target, emb, i, current_epoch, train_acc_cater (che ora sarà atrain)
    def forward(self, batch_original_indices, gnn_logits_batch, true_labels_batch_one_hot,
                gnn_embeddings_batch, batch_iter_num, current_epoch, atrain_overall_accuracy):
        # batch_original_indices: il tuo 'index_run'
        # gnn_logits_batch: il tuo 'outputs'
        # true_labels_batch_one_hot: il tuo 'target'
        # gnn_embeddings_batch: il tuo 'emb'
        # batch_iter_num: il tuo 'i' (numero iterazione batch)
        # current_epoch: il tuo 'current_epoch'
        # atrain_overall_accuracy: il tuo 'train_acc_cater' (ma ora rappresenta l'accuratezza globale)

        if self.first_epoch_or_batch_init_centroid and current_epoch == 0 and batch_iter_num == 0:
            # Aggiorna i centroidi solo una volta all'inizio del training,
            # o più frequentemente se necessario (es. ogni epoca).
            # Per ora, lo facciamo solo all'inizio della prima epoca.
            # Assicurati che prev_gnn_embeddings sia popolato prima di chiamarlo
            # Questo implica che potresti dover fare un forward pass iniziale o usare valori random.
            # Data la logica, è meglio aggiornarli dopo che prev_gnn_embeddings è stato popolato.
            # Quindi, sposteremo l'aggiornamento dopo aver memorizzato gli embedding.
            pass


        # Store current embeddings for future centroid updates
        # Assicurati che batch_original_indices siano indici validi per self.prev_gnn_embeddings
        pytorch_batch_indices = torch.tensor(batch_original_indices, device=self.device, dtype=torch.long)
        with torch.no_grad():
            self.prev_gnn_embeddings[pytorch_batch_indices] = gnn_embeddings_batch.detach()

        # Aggiorna i centroidi
        if self.first_epoch_or_batch_init_centroid and current_epoch == 0 and batch_iter_num == 0:
            self._update_centroids() # Ora prev_gnn_embeddings ha i primi valori
            self.first_epoch_or_batch_init_centroid = False # Aggiorna solo una volta
        elif current_epoch > 0 and batch_iter_num == 0 : # Esempio: aggiorna all'inizio di ogni epoca successiva
            self._update_centroids()


        u_batch = self.u[pytorch_batch_indices]

        # --- L1: Modified Cross-Entropy Loss (Eq 4) ---
        soft_labels_target_batch = self._get_soft_labels(gnn_embeddings_batch)

        logit_modification = atrain_overall_accuracy * u_batch * true_labels_batch_one_hot
        modified_logits = gnn_logits_batch + logit_modification

        l1_loss_per_sample = -torch.sum(soft_labels_target_batch * F.log_softmax(modified_logits, dim=1), dim=1)
        l1_loss = torch.mean(l1_loss_per_sample)

        # --- L2: Loss for updating 'u' (Eq 6) ---
        with torch.no_grad():
            pred_indices = torch.argmax(gnn_logits_batch, dim=1)
            predicted_labels_one_hot = F.one_hot(pred_indices, num_classes=self.num_classes).float()

        term_for_u_update = predicted_labels_one_hot + u_batch * true_labels_batch_one_hot - true_labels_batch_one_hot
        l2_loss = torch.mean(torch.sum(term_for_u_update**2, dim=1)) / self.num_classes

        # --- L3: Regularization for 'u' (Eq 8) ---
        prob_true_class_from_logits = torch.sum(F.softmax(gnn_logits_batch.detach(), dim=1) * true_labels_batch_one_hot, dim=1)
        prob_true_class = torch.clamp(prob_true_class_from_logits, eps, 1.0 - eps)

        clamped_u_batch_squeezed = torch.clamp(u_batch.squeeze(dim=1), min=eps)
        u_transformed = torch.sigmoid(-torch.log(clamped_u_batch_squeezed))
        u_transformed = torch.clamp(u_transformed, eps, 1.0 - eps)

        dkl_per_sample = prob_true_class * torch.log(prob_true_class / u_transformed) + \
                         (1 - prob_true_class) * torch.log((1 - prob_true_class) / (1 - u_transformed))

        # Gestisci NaN in dkl_per_sample (può succedere se p o q sono esattamente 0 o 1 nonostante clamp)
        dkl_per_sample = torch.nan_to_num(dkl_per_sample, nan=0.0, posinf=0.0, neginf=0.0)

        l3_loss = torch.mean(dkl_per_sample) * (1.0 - atrain_overall_accuracy)

        total_loss = l1_loss + l2_loss + l3_loss

        return total_loss