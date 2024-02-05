import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, rep1, rep2, temperature=0.5):
        normalized_rep1 = F.normalize(rep1)
        normalized_rep2 = F.normalize(rep2)
        dis_matrix = torch.mm(normalized_rep1, normalized_rep2.T)/temperature

        pos = torch.diag(dis_matrix)
        dedominator = torch.sum(torch.exp(dis_matrix), dim=1)
        loss = (torch.log(dedominator)-pos).mean()
        
        return loss