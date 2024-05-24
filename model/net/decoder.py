
import torch.nn as nn

from .util import *
from .fuse import FUSE_MODULE_DICT

class Decoder(nn.Module):
    def __init__(self, n_classes, fuse_feats, feats='x', aux=False, final_aux=False, lf_args={}):
        super().__init__()

        self.aux = aux
        self.feats = feats
        self.final_aux = final_aux and aux
        
        decoder_feats = fuse_feats[-2:0:-1]

        # Refine Blocks
        for i in range(len(decoder_feats)):
            self.add_module('refine%d' % i,
                Level_Fuse_Module(in_feats=decoder_feats[i], **lf_args)
            )

        # Upsample Blocks
        for i in range(len(decoder_feats)):
            self.add_module('up%d' % i, 
                IRB_Up_Block(decoder_feats[i], aux=aux)
            )
        
        # Aux loss
        if aux:
            for i in range(len(decoder_feats)):
                self.add_module('aux%d' % i, 
                    nn.Conv2d(decoder_feats[i], n_classes, kernel_size=1, stride=1, padding=0, bias=True),
                )
        
        # Final fusion
        self.out_conv = nn.Sequential(
            nn.Conv2d(min(decoder_feats), n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.out_up = nn.Sequential(
            LearnedUpUnit(n_classes),
            LearnedUpUnit(n_classes)
        )

    def forward(self, in_feats):
        f1 = in_feats['%s1' % self.feats]
        f2 = in_feats['%s2' % self.feats]
        f3 = in_feats['%s3' % self.feats]
        f4 = in_feats['%s4' % self.feats]

        if self.aux:
            # Level Fuse
            feats, aux0 = self.up0(f4)
            feats = self.refine0(feats, f3)
            feats, aux1 = self.up1(feats)
            feats = self.refine1(feats, f2)
            feats, aux2 = self.up2(feats)
            feats = self.refine2(feats, f1)
            # Output
            aux3 = self.out_conv(feats)
            if self.final_aux:
                out_feats = [self.out_up(aux3), aux3, self.aux2(aux2), self.aux1(aux1), self.aux0(aux0)]
            else:
                out_feats = [self.out_up(aux3), self.aux2(aux2), self.aux1(aux1), self.aux0(aux0)]
            return out_feats
        else:
            # feats = self.refine0(self.up0(f4), f3)
            # feats = self.refine1(self.up1(feats), f2)
            # feats = self.refine2(self.up2(feats), f1)
            # feats = self.out_conv(feats)  #
            feats = self.up0(f4)
            feats = self.up1(feats)
            feats = self.up2(feats)
            feats = self.out_conv(feats)
            out_feats = [self.out_up(feats)]
            return out_feats

class Level_Fuse_Module(nn.Module):
    def __init__(self, in_feats, conv_flag=(True, False), lf_bb='irb[2->2]', fuse_args={}, fuse_module='mlgf'):
        super().__init__()
        self.conv_flag = conv_flag
        self.fuse = FUSE_MODULE_DICT[fuse_module](in_feats, **fuse_args)
        self.rfb0 = customized_module(lf_bb, in_feats) if conv_flag[0] else nn.Identity()
        self.rfb1 = customized_module(lf_bb, in_feats) if conv_flag[1] else nn.Identity()
    
    def forward(self, y, x):
        x = self.rfb0(x)    # Refine feats from backbone
        out, _, _ = self.fuse(y, x)
        return self.rfb1(out)

class IRB_Up_Block(nn.Module):
    def __init__(self, in_feats, aux=False):
        super().__init__()
        self.aux = aux
        self.conv_unit = nn.Sequential(
            IRB_Block(2*in_feats, 2*in_feats),
            IRB_Block(2*in_feats, 2*in_feats),
            IRB_Block(2*in_feats, in_feats)
        )
        self.up_unit = LearnedUpUnit(in_feats)

    def forward(self, x):
        feats = self.conv_unit(x)
        return (self.up_unit(feats), feats) if self.aux else self.up_unit(feats)