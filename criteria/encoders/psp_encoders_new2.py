import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear

class GradualStyleBlock_combine(Module):
    def __init__(self, in_c, out_c, spatial,ind):
        super(GradualStyleBlock_combine, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))-1
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.encode = nn.Sequential(*modules)
        self.convs=nn.ModuleList()
        self.linears=nn.ModuleList()
        for i in range(ind):
            modules = [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()]
            self.convs.append(nn.Sequential(*modules))
            self.linears.append(EqualLinear(out_c, out_c, lr_mul=1))

    def forward(self, x):
        x = self.encode(x)
        results=[]
        for i in range(len(self.convs)):
            y=self.convs[i](x)
            y=y.view(-1, self.out_c)
            y=self.linears[i](y)
            results.append(y)
        return results
class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x
class AU_Mapping(Module):
    def __init__(self, in_c, out_c, spatial):
        super(AU_Mapping, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        tmp_c=512
        modules += [Conv2d(in_c, tmp_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(tmp_c, tmp_c//2, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
            tmp_c=tmp_c//2
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(tmp_c, out_c, lr_mul=1)
    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
class AU_Encoder(Module):
    def __init__(self):
        super(AU_Encoder, self).__init__()
        self.au_times=AU_Mapping(512+17,17,64)
        self.au_direction = nn.Parameter(torch.rand(17, 18, 512))
        self.styles=nn.ModuleList()
        self.styles.append(GradualStyleBlock_combine(512+17, 512, 16, 3))
        self.styles.append(GradualStyleBlock_combine(512+17, 512, 32, 4))
        self.styles.append(GradualStyleBlock_combine(512+17, 512, 64, 11))
    def forward(self,x,tar_au):
        au = tar_au.unsqueeze(-1).unsqueeze(-1)
        au_bias=[]
        for index,feature in enumerate(x):
            tmp_au = au.expand(au.size(0), au.size(1), feature.size(2), feature.size(3))
            au_bias+=self.styles[index](torch.cat((feature,tmp_au),dim=1))
        au_bias=torch.stack(au_bias,dim=1)
        au_times=(self.au_times(torch.cat((feature,tmp_au),dim=1))*tar_au).unsqueeze(-1).unsqueeze(-1)
        # print(tar_au.squeeze())
        # print(au_times.squeeze())
        au_times = au_times.expand(au_times.size(0), au_times.size(1), self.au_direction.size(1)
                                   , self.au_direction.size(2))
        au_direction=(self.au_direction*au_times).sum(dim=1)
        return au_direction+au_bias
class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None,need_au=True,psp=False):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        self.psp=psp
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        combine_style=opts.combine_style
        self.combine_style=combine_style
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.need_au=need_au
        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        if psp:
            for i in range(self.style_count):
                if i < self.coarse_ind:
                    style = GradualStyleBlock(512, 512, 16)
                elif i < self.middle_ind:
                    style = GradualStyleBlock(512, 512, 32)
                else:
                    style = GradualStyleBlock(512, 512, 64)
                self.styles.append(style)
        else:
            self.styles.append(GradualStyleBlock_combine(512, 512, 16,self.coarse_ind))
            self.styles.append(GradualStyleBlock_combine(512, 512, 32,self.middle_ind-self.coarse_ind))
            self.styles.append(GradualStyleBlock_combine(512, 512, 64,18-self.middle_ind))
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x,return_feature=False):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        if self.psp:
            for j in range(self.coarse_ind):
                latents.append(self.styles[j](c3))
            p2 = self._upsample_add(c3, self.latlayer1(c2))
            for j in range(self.coarse_ind, self.middle_ind):
                latents.append(self.styles[j](p2))
            p1 = self._upsample_add(p2, self.latlayer2(c1))
            for j in range(self.middle_ind, self.style_count):
                latents.append(self.styles[j](p1))
        else:
            latents+=self.styles[0](c3)
            p2 = self._upsample_add(c3, self.latlayer1(c2))
            latents+=self.styles[1](p2)
            p1 = self._upsample_add(p2, self.latlayer2(c1))
            latents+=self.styles[2](p1)
        out = torch.stack(latents, dim=1)
        if return_feature:
            return out,[c3,p2,p1]
        else:
            return out


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x