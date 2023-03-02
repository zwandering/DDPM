import torch
from denoising_diffusion_pytorch import Unet
from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
from argparse import ArgumentParser
import os
from torch.optim import Adam
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
#DDPM训练
parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, default="/home/huteng/dataset/ffhq-256", help='Path to data dir')
parser.add_argument('--name', type=str, default="", help='dataset name')
parser.add_argument('--batch_size', type=int, default=8, )
# parser.add_argument('--x2', action='store_true', help='Whether to also save inputs + outputs side-by-side')
args = parser.parse_args()
# print(opts.x1,opts.x2)
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()
diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,   # number of steps
    loss_type = 'l1',    # L1 or L2
    p2_loss_weight_gamma = 1,
    beta_schedule= 'cosine'
).cuda()

# diffusion.model.load_state_dict(torch.load('/home/huteng/zhuhaokun/DDPM/pretrained_models/101000.pth', map_location='cuda:0'),strict=True)
# if opts.resume_iter!=-1:
#     diffusion.model.load_state_dict(torch.load(os.path.join(save_dir,'models','%d.pth'%opts.resume_iter)))
ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
ddp_diffusion = DDP(diffusion, device_ids=[local_rank], output_device=local_rank)

class Train_Data(Dataset):
    def __init__(self, img_path):
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256])
        ])
        self.data_path = img_path
        self.file_names = os.listdir(self.data_path)
        self.l=len(self.file_names)
    def __getitem__(self, idx):
        idx=idx
        image = Image.open(os.path.join(self.data_path, self.file_names[idx])).convert('RGB')
        image = self.loader(image)
        return image

    def __len__(self):
        return self.l
#train_data=Train_Data('/home/huteng/dataset/celeba')
# name='sunglasses'
# train_data=Train_Data('style_img/%s'%name)
real_data=Train_Data(args.data_dir)

# train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
real_sampler = torch.utils.data.distributed.DistributedSampler(real_data)

# train_dataloader = DataLoader(train_data,
#                                    batch_size=10,
#                                    # shuffle=True,
#                                    num_workers=8,
#                                    drop_last=True,
#                                     sampler=train_sampler,
#                               )


real_dataloader = DataLoader(real_data,
                                   batch_size=args.batch_size,
                                   # shuffle=True,
                                   num_workers=8,
                                   drop_last=True,
                                sampler=real_sampler,
                             )
real_dataloader_iter=iter(real_dataloader)
# print(torch.load('/home/huteng/DDPM2/pretrained_models/ffhq_p2.pt').items())

# params = torch.load('/home/huteng/zhuhaokun/DDPM/pretrained_models/5500.pth')
# # items = ["model.downs.0.3.weight", "model.downs.0.3.bias", "model.downs.1.3.weight", "model.downs.1.3.bias", "model.downs.2.3.weight", "model.downs.2.3.bias"]
# # items_ = ["model.downs.0.3.weight", "model.downs.0.3.bias", "model.downs.1.3.weight", "model.downs.1.3.bias", "model.downs.2.3.weight", "model.downs.2.3.bias"]
# #
# # for item in items:
# #     data = params[item]  # 备份旧键
# #     del params[item]  # 删除旧键
# #     print(item[0:16]+'1.'+item[16:])
# #     params[item[0:16]+'1.'+item[16:]] = data  # 新建新键值对并存入数据
#
# diffusion.load_state_dict(params,strict=True)
optizer = Adam(ddp_diffusion.module.model.parameters(), lr = 1e-4, betas =(0.9, 0.99))
global_step=0

dir=f'results_128_2_256/fine-tune-model/{args.name}/'

os.makedirs(dir+'models',exist_ok=True)
os.makedirs(dir+'images',exist_ok=True)
for epoch in range(100):
    for batch_idx,batch in enumerate(real_dataloader):
        if batch_idx%100==0:
            print(batch_idx)
        image=batch.cuda()
        loss = diffusion(image)
        optizer.zero_grad()
        loss.backward()
        optizer.step()
        global_step+=1
        if global_step%1==0 and global_step!=0:
            torch.save(ddp_diffusion.module.state_dict(),dir+'models/%d.pth'%global_step)
            sampled_images = ddp_diffusion.module.ddim_sample((36, 3, 256, 256), sample_step=50)
            # sampled_images = diffusion.ddim_sample((36, 3, 256, 256), sample_step=50)
            save_image(sampled_images,dir+'images/%d.jpg'%global_step,nrow=6)

            # noise_step = 800
            # real_image = next(real_dataloader_iter, None)
            # if real_image is None:
            #     real_dataloader_iter = iter(real_dataloader)
            #     real_image = next(real_dataloader_iter, None)
            # real_image = real_image.cuda()
            # t = torch.ones(len(real_image)).long().to('cuda') * noise_step
            # noises = diffusion.p_losses(real_image, t, return_x=True)
            # sampled_images, sampled_middle_images = diffusion.ddim_sample(real_image.shape, sample_step=50,
            #                                                                      return_middle=True, start_img=noises,
            #                                                                      max_step=noise_step,
            #                                                                      min_step=-1,)
            # save_image(torch.cat((real_image, noises, sampled_middle_images, sampled_images), dim=0),
            #            os.path.join(dir, 'images/%d-sample-middle.jpg' % global_step), nrow=16, normalize=False)
            # save_image(sampled_images, os.path.join(dir, 'images/%d--sample.jpg' % global_step),
            #            nrow=4, normalize=False)

# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 train_128_2_256_dist.py --name ffhq --batch_size 8