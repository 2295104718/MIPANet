import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.nn import Softmax

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Pyramid-Context Guided Fusion Module
from model import model

#MIPANet
class MIPA_Module(nn.Module):
    def __init__(self, in_feats, pp_size=(1, 2, 4, 8), descriptor=8, mid_feats=16, sp_feats='u'):
        super().__init__()
        self.in_feats = in_feats
        # 调用交叉注意力CCNet
        # self.CC = CC_Module(in_feaIts)

        # 调用双输入自注意力Self_attention
        # self.Self_att = Self_attention(in_feats,in_feats,in_feats)

        # 调用单特征输入自注意力SelfAttention
        # self.SelfAtt = SelfAttention(in_feats)

        # 调用交叉注意力CrossAttention
        self.Corss = CrossAttention(in_feats)  # 对应函数中的dim参数

        self.sp_feats = sp_feats
        self.pp_size = pp_size
        self.feats_size = sum([(s ** 2) for s in self.pp_size])
        self.descriptor = descriptor

        # without dim reduction
        if (descriptor == -1) or (self.feats_size < descriptor):
            self.des = nn.Identity()
            self.fc = nn.Sequential(nn.Linear(in_feats * self.feats_size, mid_feats, bias=False),
                                    nn.BatchNorm1d(mid_feats),
                                    nn.ReLU(inplace=True))
        # with dim reduction
        else:
            self.des = nn.Conv2d(self.feats_size, self.descriptor, kernel_size=1)
            self.fc = nn.Sequential(nn.Linear(in_feats * descriptor, mid_feats, bias=False),
                                    nn.BatchNorm1d(mid_feats),
                                    nn.ReLU(inplace=True))

        self.fc_x = nn.Linear(mid_feats, in_feats)
        self.fc_y = nn.Linear(mid_feats, in_feats)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        batch_size, ch, _, _ = x.size()
        sp_dict = {'x': x, 'y': y, 'u': x + y}

        pooling_pyramid = []

        # CCNet方式融合后的特征
        # fuse_feats = self.CC(sp_dict[self.sp_feats])

        # 单特征输入SelfAttention方式融合后的特征
        # fuse_feats = self.SelfAtt(sp_dict[self.sp_feats])  #sp_dict[self.sp_feats]:  [16,64,240,240]

        # 交叉注意力融合
        out_1 = self.Corss(sp_dict['y'], sp_dict['x'])  # RGB
        out_2 = self.Corss(sp_dict['x'], sp_dict['y'])  # Dep

        return out_1 + out_2, out_1, out_2


# Multi-Level General Fusion Module
class MLGF_Module(nn.Module):
    def __init__(self, in_feats, fuse_setting={}, att_module='par', att_setting={}):
        super().__init__()
        module_dict = {
            'se': SE_Block,
            'par': PAR_Block,
            'idt': IDT_Block
        }
        self.att_module = att_module
        self.pre1 = module_dict[att_module](in_feats, **att_setting)
        self.pre2 = module_dict[att_module](in_feats, **att_setting)
        self.gcgf = General_Fuse_Block(in_feats, **fuse_setting)

    def forward(self, x, y):
        if self.att_module != 'idt':
            x = self.pre1(x)
            y = self.pre2(y)
        return self.gcgf(x, y), x, y


class General_Fuse_Block(nn.Module):
    def __init__(self, in_feats, merge_mode='grp', init=True, civ=1):
        super().__init__()
        merge_dict = {
            'add': Add_Merge(in_feats),
            'cc3': CC3_Merge(in_feats),
            'lma': LMA_Merge(in_feats),
            'grp': nn.Conv2d(2 * in_feats, in_feats, kernel_size=1, padding=0, groups=in_feats)
        }
        self.merge_mode = merge_mode
        self.merge = merge_dict[merge_mode]
        if init and isinstance(self.merge, nn.Conv2d):
            self.merge.weight.data.fill_(civ)

    def forward(self, x, y):
        if self.merge_mode != 'grp':
            return self.merge(x, y)
        b, c, h, w = x.size()
        feats = torch.cat((x, y), dim=-2).reshape(b, 2 * c, h, w)  # [b, c, 2h, w] => [b, 2c, h, w]
        return self.merge(feats)


# Attention Refinement Blocks

class SE_Block(nn.Module):
    def __init__(self, in_feats, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_feats, in_feats // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_feats // r, in_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(F.adaptive_avg_pool2d(x, 1))
        return w * x


# 局部区域增强
class PAR_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=4, descriptor=8, mid_feats=16):
        super().__init__()
        self.layer_size = pp_layer  # l: pyramid layer num
        self.feats_size = (4 ** pp_layer - 1) // 3  # f: feats for descritor
        self.descriptor = descriptor  # d: descriptor num (for one channel)

        self.des = nn.Conv2d(self.feats_size, descriptor, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(descriptor * in_feats, mid_feats, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_feats, in_feats),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        l, f, d = self.layer_size, self.feats_size, self.descriptor
        pooling_pyramid = []
        for i in range(l):
            pooling_pyramid.append(F.adaptive_avg_pool2d(x, 2 ** i).view(b, c, 1, -1))
        y = torch.cat(tuple(pooling_pyramid), dim=-1)  # [b,  c, 1, f]
        y = y.reshape(b * c, f, 1, 1)  # [bc, f, 1, 1]
        y = self.des(y).view(b, c * d)  # [bc, d, 1, 1] => [b, cd, 1, 1]
        w = self.mlp(y).view(b, c, 1, 1)  # [b,  c, 1, 1] => [b, c, 1, 1]
        return w * x


class IDT_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


# Merge Modules

class Add_Merge(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return x + y


class LMA_Merge(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lamb = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        return x + self.lamb * y


class CC3_Merge(nn.Module):
    def __init__(self, in_feats, *args, **kwargs):
        super().__init__()
        self.cc_block = nn.Sequential(
            nn.Conv2d(2 * in_feats, in_feats, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        return self.cc_block(torch.cat((x, y), dim=1))


class ADD_Module(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return x + y, x, y


# PAM
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):  # in_channels , head_dim :64
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        self.to_avg_pool = nn.AdaptiveAvgPool2d(2)  # 自适应平均池化到2x2
        self.to_max_pool = nn.MaxPool2d(kernel_size=2)  # 最大池化到1x1
        self.to_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.to_sigmoid = nn.Sigmoid()  # 可尝试其他激活函数

    def forward(self, x):
        # print('x.shape: ',x.shape)
        avg_pooled_x = self.to_avg_pool(x)
        max_pooled_x = self.to_max_pool(avg_pooled_x)
        weighted_x = self.to_conv(max_pooled_x)
        # print('weight_x: ',weighted_x)
        attention_x = self.to_sigmoid(weighted_x)
        # print('attention_x: ',attention_x)
        out_x = x * attention_x + x

        return out_x

#PAM
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spatial attention
        spatial_attention = self.sigmoid(self.conv1(x))

        # Multiply spatial attention with input
        out_x = x * self.relu(spatial_attention)

        return out_x

#PAM
class to_Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Self1 = ChannelAttention(in_channels)
        self.Self2 = SpatialAttention(in_channels)

    def forward(self, x, y):
        # out_1 = self.Self1(x)
        # out_2 = self.Self1(y)
        #
        # return out_1 + out_2, out_1, out_2

        out_1 = self.Self2(x)
        out_2 = self.Self1(y)
        result = out_1 + out_2
        #result = SelfAttention(CBR(result))

        return  result, out_1, out_2

#MIM
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):  # 0.
        super().__init__()
        inner_dim = dim_head * heads  # 计算每个注意力头的内部维度
        project_out = not (
                    heads == 1 and dim_head == dim)  # 这个布尔值表示是否需要将多头注意力的输出映射回原始维度。当注意力头数为1且每个头的维度等于输入特征的维度时，不需要映射，否则需要映射

        self.heads = heads
        self.dim = dim
        self.scale = dim_head ** -0.5  # 这是用于缩放注意力分数的因子，确保注意力分数不会过大

        # 分别是用于映射输入X、Y中的键、值和查询的线性层
        self.to_k = nn.Linear(dim, inner_dim, bias=False)  # dim = 128,inner_dim = 512
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        # 这是用于将多头注意力输出映射回原始维度的线性层，如果不需要映射，则使用nn.Identity()
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_q, x_kv):  # X Y     Y X
        b1, n1, height, width = x_q.size()
        b2, n2, height, width = x_q.size()
        out_kv = x_kv
        h = self.heads
        x_q = x_q.view(b1, n1, -1)
        x_kv = x_kv.view(b2, n2, -1)
        # print('x_q:',x_q.shape)         #[1,128,3600]
        # print('x_kv:',x_kv.shape)

        x_q = x_q.permute(0, 2, 1)  # [1,3600,128]
        x_kv = x_kv.permute(0, 2, 1)

        k = self.to_k(x_kv)  # 【1，3600，512】
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)  # [1,8,3600,64]

        v = self.to_v(x_kv)  # [1,3600,512]
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)  # [1,8,3600,64]

        q = self.to_q(x_q)  # [1,3600,512]
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)  # [1,8,3600,64]

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # [1,8,3600,3600]
        # print('dots:',dots.shape)

        attn = dots.softmax(dim=-1)

        # print('-------------得到注意力了-------------')
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # [1,8,3600,64]
        # print('out:',out.shape)
        out = rearrange(out, 'b h n d -> b n (h d)')  # [1,3600,512]
        # print('out_1:', out.shape)
        out = self.to_out(out)  # [1,3600,128]       最后应该要[1,128,60,60]
        # print('out_2:',out.shape)
        # 将注意力输出重塑回原始形状
        out = out.permute(0, 2, 1).view(b1, self.dim, height, width)
        # print('result:',out.shape)

        return out + out_kv

#RM
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # BxNXC'
        key = self.key(x).view(batch_size, -1, width * height)  # BxC'xN
        value = self.value(x).view(batch_size, -1, width * height)  # BxCxN

        attention = torch.bmm(query, key)  # BxNxN
        attention = torch.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))  # BxCxN
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x
        return out

FUSE_MODULE_DICT = {
    'add': ADD_Module,
    'mlgf': MLGF_Module,
    'MIPA': MIPA_Module,
    'fuse': to_Attention
}
