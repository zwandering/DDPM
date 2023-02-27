from denoising_diffusion_pytorch import Unet
import torch
from model.ddpm import GaussianDiffusion
model_source = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    self_condition=True
)
diffusion_source = GaussianDiffusion(
    model_source,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)
ckpt=torch.load('pretrained_models/481157.pth')
print(ckpt.keys())
#ckpt=torch.load('100.pth')
# keys=['model.init_conv.weight','model.init_conv.bias']
# for i in keys:
#     if i=='model.init_conv.weight':
#         ckpt[i]=torch.cat((ckpt[i],ckpt[i]),dim=1)/2
#     if i=='model.init_conv.bias':
#         ckpt[i]=ckpt[i]/2
diffusion_source.load_state_dict(ckpt)
print('load_success')