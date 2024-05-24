import torch
import torch.nn as nn
from addict import Dict

from .fuse import FUSE_MODULE_DICT
from .util import get_resnet18, get_resnet50, get_resnet101


class Encoder(nn.Module):
    def __init__(self, encoder='2b', encoder_args={}):
        super().__init__()
        encoder_dict = {
            'res18': Res18_Encoder,
            'res50': Res50_Encoder
        }
        self.encoder = encoder_dict[encoder](**encoder_args)
    
    def forward(self, l, d):
        return self.encoder(l, d)

class Res18_Encoder(nn.Module):
    def __init__(self, fuse_args, pass_rff=(True, False), fuse_module='fuse'):
        super().__init__()

        self.pass_rff = pass_rff

        self.rgb_base = get_resnet18(input_dim=3)
        self.dep_base = get_resnet18(input_dim=1)

        # Divide backbone into 5 parts
        self.rgb_layer0 = nn.Sequential(self.rgb_base.conv1,
                                        self.rgb_base.bn1,
                                        self.rgb_base.relu)
        self.dep_layer0 = nn.Sequential(self.dep_base.conv1, 
                                        self.dep_base.bn1,
                                        self.dep_base.relu)
        self.rgb_inpool = self.rgb_base.maxpool
        self.dep_inpool = self.dep_base.maxpool
        for i in range(1, 5):
            self.add_module('rgb_layer%d' % i, self.rgb_base.__getattr__('layer%d' % i))
            self.add_module('dep_layer%d' % i, self.dep_base.__getattr__('layer%d' % i))
        
        # Fuse Block
        self.fuse_feats = [64, 64, 128, 256, 512]
        for i in range(len(self.fuse_feats)):
            if i ==4:
                self.add_module(
                    'fuse%d' % i, FUSE_MODULE_DICT[fuse_module](self.fuse_feats[i], **fuse_args))
            else:
                self.add_module('fuse%d' % i, FUSE_MODULE_DICT['fuse'](self.fuse_feats[i]))

    def forward(self, l, d):

        # => [B, 64, h/2, w/2]
        l0 = self.rgb_layer0(l)
        d0 = self.dep_layer0(d)
        # x0, _, y0 = self.fuse0(l0, d0)

        # => [B, 64, h/4, w/4]
        l1 = self.rgb_inpool(l0 if self.pass_rff[0] else l0)
        d1 = self.dep_inpool(d0 if self.pass_rff[1] else d0)
        l1 = self.rgb_layer1(l1)  
        d1 = self.dep_layer1(d1)
        x1, r1, y1 = self.fuse1(l1,d1)

        # => [B, 128, h/8, w/8]
        l2 = self.rgb_layer2(r1 if self.pass_rff[0] else l1)
        d2 = self.dep_layer2(y1 if self.pass_rff[1] else d1)
        x2, r2 ,y2 = self.fuse2(l2,d2)

        # => [B, 256, h/16, w/16]
        l3 = self.rgb_layer3(r2 if self.pass_rff[0] else l2)
        d3 = self.dep_layer3(y2 if self.pass_rff[1] else d2)
        x3, r3 ,y3 = self.fuse3(l3,d3)

        # => [B, 512, h/32, w/32]
        l4 = self.rgb_layer4(r3 if self.pass_rff[0] else l3)
        d4 = self.dep_layer4(y3 if self.pass_rff[1] else d3)
        x4, _, y4 = self.fuse4(l4, d4)

        feats = Dict({
            'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4,
            'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4,
            'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
            'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4,
        })

        return feats

class Res50_Encoder(nn.Module):
    def __init__(self, fuse_args, pass_rff=(True, False), fuse_module='fuse'):
        super().__init__()

        self.pass_rff = pass_rff

        self.rgb_base = get_resnet50(input_dim=3)
        self.dep_base = get_resnet50(input_dim=1)

        # Divide backbone into 5 parts
        self.rgb_layer0 = nn.Sequential(self.rgb_base.conv1, self.rgb_base.bn1, self.rgb_base.relu,
                                        self.rgb_base.conv2, self.rgb_base.bn2, self.rgb_base.relu,
                                        self.rgb_base.conv3, self.rgb_base.bn3, self.rgb_base.relu,)
        self.dep_layer0 = nn.Sequential(self.dep_base.conv1, self.dep_base.bn1, self.dep_base.relu,
                                        self.dep_base.conv2, self.dep_base.bn2, self.dep_base.relu,
                                        self.dep_base.conv3, self.dep_base.bn3, self.dep_base.relu,)
        self.rgb_layer1 = nn.Sequential(self.rgb_base.maxpool, self.rgb_base.layer1)
        self.dep_layer1 = nn.Sequential(self.dep_base.maxpool, self.dep_base.layer1)
        for i in range(2, 5):
            self.add_module('rgb_layer%d' % i, self.rgb_base.__getattr__('layer%d' % i))
            self.add_module('dep_layer%d' % i, self.dep_base.__getattr__('layer%d' % i))
        
        # Fuse Block
        self.fuse_feats = [128, 256, 512, 1024, 2048]
        for i in range(len(self.fuse_feats)):
            if i == 4:
                self.add_module(
                    'fuse%d' % i, FUSE_MODULE_DICT[fuse_module](self.fuse_feats[i], **fuse_args))
            else:
                self.add_module('fuse%d' % i, FUSE_MODULE_DICT['fuse'](self.fuse_feats[i]))
    
    def forward(self, l, d):

        # => [B, 64, h/2, w/2]
        # d_3 = d.expand(-1, 3, -1, -1)  # [32,3,480,480]
        l0 = self.rgb_layer0(l)
        d0 = self.dep_layer0(d)
        # x0, _, y0 = self.fuse0(l0, d0)

        # => [B, 64, h/4, w/4]
        l1 = self.rgb_layer1(l0 if self.pass_rff[0] else l0)
        d1 = self.dep_layer1(d0 if self.pass_rff[1] else d0)
        x1, r1, y1 = self.fuse1(l1, d1)

        # => [B, 128, h/8, w/8]
        l2 = self.rgb_layer2(l1 if self.pass_rff[0] else l1)
        d2 = self.dep_layer2(d1 if self.pass_rff[1] else d1)
        x2, r2, y2 = self.fuse2(l2, d2)

        # => [B, 256, h/16, w/16]
        l3 = self.rgb_layer3(l2 if self.pass_rff[0] else l2)
        d3 = self.dep_layer3(d2 if self.pass_rff[1] else d2)
        x3, r3, y3 = self.fuse3(l3, d3)

        # => [B, 512, h/32, w/32]
        l4 = self.rgb_layer4(l3 if self.pass_rff[0] else l3)
        d4 = self.dep_layer4(d3 if self.pass_rff[1] else d3)
        x4, _, y4 = self.fuse4(l4, d4)

        feats = Dict({
            'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4,
            'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4,
            'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
            'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4,
        })

        return feats