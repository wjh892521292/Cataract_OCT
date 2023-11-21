import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat

from collections import OrderedDict

'''
transformer融合
'''


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class BasicTransformer(nn.Module):
    def __init__(self, dim, depth, nhead, nhid, dropout):
        super().__init__()
        depth = depth
        heads = nhead
        assert nhid % nhead == 0
        dim_head = nhid // nhead

        self.pos_embedding = nn.Parameter(torch.randn(1, 64 + 2, dim))
        self.transformer_encoder = Transformer(dim, depth, heads, dim_head, dim * 4, dropout)
        self.dropout = nn.Dropout(dropout)

    # make some changes to the end layer contrast to the original resnet
    def forward(self, zz, x, y):
        p = torch.cat((zz, x, y), 1)

        p = self.transformer_encoder(p)
        zz = p[:, 0, :].contiguous()

        return zz



class MMF(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dp = args.dropout
        self.layers = args.resnet_layers
        self.patchsize = args.patchsize
        self.scale = 64 // self.patchsize
        resnet = getattr(torchvision.models, 'resnet{}'.format(self.layers))

        self.resnet_hor = resnet(pretrained=True)
        self.resnet_ver = resnet(pretrained=True)

        self.feature_hor = nn.Sequential(
            self.resnet_hor.conv1,
            self.resnet_hor.bn1,
            self.resnet_hor.relu,
            self.resnet_hor.maxpool,
            self.resnet_hor.layer1,
            self.resnet_hor.layer2,
            self.resnet_hor.layer3,
            self.resnet_hor.layer4)

        self.feature_ver = nn.Sequential(
            self.resnet_ver.conv1,
            self.resnet_ver.bn1,
            self.resnet_ver.relu,
            self.resnet_ver.maxpool,
            self.resnet_ver.layer1,
            self.resnet_ver.layer2,
            self.resnet_ver.layer3,
            self.resnet_ver.layer4)

        self.avgpool = nn.AdaptiveAvgPool2d((self.patchsize, self.patchsize))

        self.z_hor_encoder = nn.Linear(1, 128)
        self.z_ver_encoder = nn.Linear(1, 128)
        self.z_cva_encoder = nn.Linear(1, 128)

        self.reg_token = nn.Parameter(torch.randn(1, 1, 128))

        self.Transformer_layer = BasicTransformer(dim=128, depth=6, nhead=4, nhid=512, dropout=0.1)

        self.fc = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 128)
        )
        self.hor_feat_fc = nn.Linear(512, 128)
        self.ver_feat_fc = nn.Linear(512, 128)

        self.relu = nn.ReLU()
        self.top = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, y, z):
        b = x.shape[0]

        hor_feature = x
        ver_feature = y

        hor_feature = self.feature_hor(hor_feature)
        hor_feature = self.resnet_ver.avgpool(hor_feature)
        hor_feature = torch.flatten(hor_feature, 1).unsqueeze(1)
        hor_feature = self.hor_feat_fc(hor_feature)

        ver_feature = self.feature_ver(ver_feature)
        ver_feature = self.resnet_ver.avgpool(ver_feature)
        ver_feature = torch.flatten(ver_feature, 1).unsqueeze(1)
        ver_feature = self.ver_feat_fc(ver_feature)

        reg_token = repeat(self.reg_token, '() n d -> b n d', b=b)

        reg_token = self.Transformer_layer(reg_token, hor_feature, ver_feature)



        bcva_det = self.fc(reg_token)
        bcva_det = self.relu(bcva_det)
        bcva_det = self.top(bcva_det)

        bcva = self.sigmoid(bcva_det + z) * 1.5

        return bcva, bcva - z, bcva - z - 0.2

    def model_name(self):
        return 'MMF'
