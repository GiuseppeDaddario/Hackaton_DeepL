import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math
import copy
import random

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