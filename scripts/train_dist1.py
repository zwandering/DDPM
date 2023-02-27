import torch
import sys
sys.path.append('..')
# from denoising_diffusion_pytorch import Unet
from unet import MyUnet
from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
from torch.optim import Adam
import numpy as np
import random
import clip
from style_loss import loss
from argparse import ArgumentParser
from model.big_unet import create_model
from util import resize_right
import torchvision

import os
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
# print(local_rank)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")

#在train-clip基础上以风格图像为condition加入训练
parser = ArgumentParser()
#parser.add_argument('--output_dir', type=str,default='results', help='Path to experiment output directory')
parser.add_argument('--resume_iter', type=int,default=-1, help='Path to experiment output directory')
parser.add_argument('--real_data_dir', type=str,default='', help='Path to real data')
opts = parser.parse_args()

save_dir = '../trained'
# save_dir='../results3/%s/clip_dir_loss'%name if opts.clip_mode==0 else '../results3/%s/clip_center_loss'%name
# save_dir=os.path.join(save_dir,'step=%d,beta_f=%s,beta_s=%s'%(opts.noise_step,opts.beta_f,opts.beta_style))
print(save_dir)
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

image_size=256
model = create_model(
    image_size=image_size
)
diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = 1000,   # number of steps
    loss_type = 'l1',    # L1 or L2
    p2_loss_weight_gamma = 1
).cuda()

# diffusion.model.load_state_dict(torch.load('../pretrained_models/ffhq_p2.pt'),strict=True)
if opts.resume_iter!=-1:
    diffusion.model.load_state_dict(torch.load(os.path.join(save_dir,'models','%d.pth'%opts.resume_iter)))
ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
ddp_diffusion = DDP(diffusion, device_ids=[local_rank], output_device=local_rank)

class Train_Data(Dataset):
    def __init__(self, img_path):
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([image_size, image_size])
        ])
        self.data_path = img_path
        self.file_names = os.listdir(self.data_path)
        self.l=max(10000,len(self.file_names))
    def __getitem__(self, idx):
        idx=idx%len(self.file_names)
        image = Image.open(os.path.join(self.data_path, self.file_names[idx])).convert('RGB')
        image = self.loader(image)
        return image

    def __len__(self):
        return self.l
batch_size =8
real_data=Train_Data(opts.real_data_dir)

real_sampler = torch.utils.data.distributed.DistributedSampler(real_data)

real_dataloader = DataLoader(real_data,
                                   batch_size=batch_size,
                                   # shuffle=True,
                                   num_workers=8,
                                   drop_last=True,
                             sampler=real_sampler,
                             )
real_dataloader_iter=iter(real_dataloader)

optimizer = Adam(ddp_diffusion.module.model.parameters(), lr = 1e-4, betas =(0.9, 0.99))
global_step=0 if opts.resume_iter==-1 else opts.resume_iter

os.makedirs(os.path.join(save_dir,'models'),exist_ok=True)
os.makedirs(os.path.join(save_dir,'images'),exist_ok=True)

for epoch in range(100):
    for batch_idx,batch in enumerate(real_dataloader):
        image=batch.cuda()
        loss = diffusion(image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if global_step%1000==0 and global_step!=0:
            sampled_images = ddp_diffusion.module.ddim_sample((batch_size, 3, image_size, image_size), sample_step=50)
            save_image(sampled_images,os.path.join(save_dir,'images/%d.jpg'%global_step),normalize=True)
        if global_step % 1000 == 0 and global_step!=0:
            torch.save(ddp_diffusion.module.model.state_dict(), os.path.join(save_dir,'models/%d.pth'%global_step))
        global_step += 1
# OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=2 train_dist1.py --real_data_dir /home/huteng/dataset/ffhq/images1024x1024/
