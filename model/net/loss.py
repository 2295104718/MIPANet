import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SegmentationLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        super(SegmentationLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight) 

    def forward(self, *inputs):
        assert self.se_loss == False
        if self.aux:
            out_feats, target = inputs[0], inputs[-1]
            aux_feats, aux_loss = inputs[1:-1], []
            for aux in aux_feats:
                _, _, h, w = aux.size()
                aux_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w)).long().squeeze(1)
                aux_loss.append(super(SegmentationLoss, self).forward(aux, aux_target))
            loss1 = super(SegmentationLoss, self).forward(out_feats, target)
            loss2 = sum(aux_loss) / len(aux_loss)
            return loss1 + self.aux_weight * loss2
        else:
            out_feats, target = inputs[0], inputs[-1]
            return super(SegmentationLoss, self).forward(out_feats, target)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect