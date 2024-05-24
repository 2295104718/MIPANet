from torch import nn
from addict import Dict

from model.net import Encoder, Decoder

class PAFNet(nn.Module):

    def __init__(self, n_classes, config={}):
        super().__init__()

        self.config = Dict(config)
        
        self.encoder = Encoder(
            encoder=config.general.encoder,
            encoder_args=config.encoder_args
        )

        self.decoder = Decoder(
            n_classes=n_classes,
            fuse_feats=self.encoder.encoder.fuse_feats,
            feats=config.general.feats,
            aux=config.decoder_args.aux,
            final_aux=config.decoder_args.final_aux,
            lf_args=config.decoder_args.lf_args,
        )

    def forward(self, l, d):
        feats = self.encoder(l, d)    
        feats = self.decoder(feats)   
        return tuple(feats)

def get_pafnet(dataset='nyud', config={}):
    from .datasets import datasets
    model = PAFNet(datasets[dataset.lower()].NUM_CLASS, config=config)
    return model