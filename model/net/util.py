
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .resnet import resnet50

# Backbone
def get_resnet18(pretrained=True, input_dim=3, f_path='./model/utils/resnet18-5c106cde.pth'):
    assert input_dim in (1, 3, 4)
    model = models.resnet18(pretrained=False)

    if pretrained:
        # Check weights file
        if not os.path.exists(f_path):
            raise FileNotFoundError('The pretrained model cannot be found.')
        
        if input_dim != 3:
            model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            weights = torch.load(f_path)
            for k, v in weights.items():
                weights[k] = v.data
            conv1_ori = weights['conv1.weight']
            conv1_new = torch.zeros((64, input_dim, 7, 7), dtype=torch.float32)
            if input_dim == 4:
                conv1_new[:, :3, :, :] = conv1_ori
                conv1_new[:,  3, :, :] = conv1_ori[:,  1, :, :]
            else:
                conv1_new[:,  0, :, :] = conv1_ori[:,  1, :, :]
            weights['conv1.weight'] = conv1_new
            model.load_state_dict(weights, strict=False)
        else:
            model.load_state_dict(torch.load(f_path), strict=False)
    else:
        raise ValueError('Please use pretrained resnet18.')
    
    return model

def get_resnet50(pretrained=True, input_dim=3, f_path='./model/utils/resnet50_v2.pth'):
    assert input_dim in (1, 3, 4)
    model = resnet50(pretrained=False)

    if pretrained:
        # Check weights file
        if not os.path.exists(f_path):
            raise FileNotFoundError('The pretrained model cannot be found.')
        
        if input_dim != 3:
            model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1, bias=False)
            weights = torch.load(f_path)
            for k, v in weights.items():
                weights[k] = v.data
            conv1_ori = weights['conv1.weight']
            conv1_new = torch.zeros((64, input_dim, 3, 3), dtype=torch.float32)
            if input_dim == 4:
                conv1_new[:, :3, :, :] = conv1_ori
                conv1_new[:,  3, :, :] = conv1_ori[:,  1, :, :]
            else:
                conv1_new[:,  0, :, :] = conv1_ori[:,  1, :, :]
            weights['conv1.weight'] = conv1_new
            model.load_state_dict(weights, strict=False)
        else:
            model.load_state_dict(torch.load(f_path), strict=False)
    else:
        raise ValueError('Please use pretrained resnet18.')
    
    return model


def get_resnet101(pretrained=True, input_dim=3, f_path='./model/utils/resnet101-5d3b4d8f.pth'):
    assert input_dim in (1, 3, 4)
    model = resnet50(pretrained=False)

    if pretrained:
        # Check weights file
        if not os.path.exists(f_path):
            raise FileNotFoundError('The pretrained model cannot be found.')

        if input_dim != 3:
            model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            weights = torch.load(f_path)
            for k, v in weights.items():
                weights[k] = v.data
            conv1_ori = weights['conv1.weight']
            conv1_new = torch.zeros((64, input_dim, 3, 3), dtype=torch.float32)
            if input_dim == 4:
                conv1_new[:, :3, :, :] = conv1_ori
                conv1_new[:, 3, :, :] = conv1_ori[:, 1, :, :]
            else:
                conv1_new[:, 0, :, :] = conv1_ori[:, 1, :, :]
            weights['conv1.weight'] = conv1_new
            model.load_state_dict(weights, strict=False)
        else:
            model.load_state_dict(torch.load(f_path), strict=False)
    else:
        raise ValueError('Please use pretrained resnet18.')

    return model

# Aux modules
class ConvBnAct(nn.Sequential):
    def __init__(self, in_feats, out_feats, kernel=3, stride=1, pad=1, bias=False, conv_args = {},
                 norm_layer=nn.BatchNorm2d, act=True, act_layer=nn.ReLU(inplace=True)):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_feats, out_feats, kernel_size=kernel, stride=stride,
                                            padding=pad, bias=bias, **conv_args))
        self.add_module('bn', norm_layer(out_feats))
        self.add_module('act', act_layer if act else nn.Identity())

class ResidualBasicBlock(nn.Module):
    def __init__(self, in_feats, out_feats=None):
        super().__init__()
        self.conv_unit = nn.Sequential(
            ConvBnAct(in_feats, in_feats),
            ConvBnAct(in_feats, in_feats, act=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_unit(x)
        return self.relu(x + out)

class IRB_Block(nn.Module):
    def __init__(self, in_feats, out_feats=None, act='idt', expand_ratio=6):
        super().__init__()
        mid_feats = round(in_feats * expand_ratio)
        out_feats = in_feats if out_feats is None else out_feats
        act_layer = nn.Identity() if act == 'idt' else nn.ReLU6(inplace=True)
        self.idt = (in_feats == out_feats)
        self.irb = nn.Sequential(
                # point-wise conv
                nn.Conv2d(in_feats, mid_feats, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_feats),
                nn.ReLU6(inplace=True),
                # depth-wise conv
                nn.Conv2d(mid_feats, mid_feats, kernel_size=3, stride=1, padding=1, groups=mid_feats, bias=False),
                nn.BatchNorm2d(mid_feats),
                nn.ReLU6(inplace=True),
                # point-wise conv
                nn.Conv2d(mid_feats, out_feats, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_feats),
                act_layer
            )

    def forward(self, x):
        return (x + self.irb(x)) if self.idt else self.irb(x)

class LearnedUpUnit(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.dep_conv = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1, groups=in_feats, bias=False)

    def forward(self, x):
        x = self.up(x)
        x = self.dep_conv(x)
        return x

# Aux funtions
def interpolate(x, size, mode = 'nearest'):
    if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        return F.interpolate(x, size=size, mode=mode, align_corners=True)
    else:
        return F.interpolate(x, size=size, mode=mode)

# def init_conv(r, c=2, k=1, mode='a', decay1=-1, decay2=1e-2):
#     print('init conv: ', r, c, k, mode, decay2)
#     # kernel that mimics bilinear interpolation (from ESA Net)
#     w0 = torch.tensor([[[
#                 [0.0625, 0.1250, 0.0625],
#                 [0.1250, 0.2500, 0.1250],
#                 [0.0625, 0.1250, 0.0625]
#         ]]])
#     if mode == 'a':
#         w = torch.ones(1, 1, k, k) / (k * k)
#         return (w * (c ** decay1)).repeat(r, c, 1, 1)
#     elif mode == 'b':
#         w1 = torch.ones(1, k, k) / (k * k)
#         w2 = w1.repeat(c-1, 1, 1) * decay2
#         w = torch.cat((w1, w2), dim=0)
#         return w.view(1, c, k, k).repeat(r, 1, 1, 1)
#     elif mode == 'a3':
#         return (w0 * (c ** decay1)).repeat(r, c, 1, 1)
#     elif mode == 'b3':
#         w1 = w0.clone().repeat(r, 1, 1, 1)
#         w2 = (w0 * decay2).repeat(r, c-1, 1, 1)
#         return torch.cat((w1, w2), dim=1)
#     else:
#         raise NotImplementedError('Invalid init mode: .%s' % mode)

def customized_module(info, feats):
    module_dict = {
            'rbb': ResidualBasicBlock,
            'luu': LearnedUpUnit,
            'irb': IRB_Block,
    }
    # Format1: 'xxx[a->b]', i.e. 'module[in_feats->out_feats]'
    # Format2: 'xxx(a)', i.e. 'module(feats)'
    module_name = info[:3]
    assert module_name in module_dict
    # print(info)
    if info.find('(') != -1:
        return module_dict[module_name]((int(info[4]) * feats // 2))
    elif info.find('[') != -1:
        return module_dict[module_name]((int(info[4]) * feats // 2), (int(info[7]) * feats // 2))
    else:
        raise ValueError('Invalid customized module format: %s' % info)

def customized_module_seq(seq, feats):
    return nn.Sequential(*[customized_module(info, feats) for info in seq.split()])