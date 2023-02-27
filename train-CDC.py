import torch
from denoising_diffusion_pytorch import Unet
from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
from torch.optim import Adam
from argparse import ArgumentParser
import torch.nn as nn
#DDPM+CDC损失
parser = ArgumentParser()
parser.add_argument('--output_dir', type=str,default='results', help='Path to experiment output directory')
parser.add_argument('--beta_kl', type=float,default=100, help='Path to experiment output directory')
opts = parser.parse_args()
model_source = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()
diffusion_source = GaussianDiffusion(
    model_source,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
model_target = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()
diffusion_target = GaussianDiffusion(
    model_target,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

class Train_Data(Dataset):
    def __init__(self, img_path):
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([128, 128])
        ])
        self.data_path = img_path
        self.file_names = os.listdir(self.data_path)
        self.l=1000000
    def __getitem__(self, idx):
        idx=idx%len(self.file_names)
        image = Image.open(os.path.join(self.data_path, self.file_names[idx])).convert('RGB')
        image = self.loader(image)
        return image

    def __len__(self):
        return self.l
#train_data=Train_Data('/home/huteng/dataset/celeba')
train_data=Train_Data('/home/huteng/DDPM2/style_img/Raphael')
batch_size=8
train_dataloader = DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=8,
                                   drop_last=True)
diffusion_source.load_state_dict(torch.load('/home/huteng/kousiqi/result/model/481157.pth'))
diffusion_target.load_state_dict(torch.load('/home/huteng/kousiqi/result/model/481157.pth'))
optimizer = Adam(diffusion_target.parameters(), lr = 1e-4, betas =(0.9, 0.99))
global_step=0
cos_sim = nn.CosineSimilarity()
sfm = nn.Softmax(dim=1)
kl_loss = nn.KLDivLoss()
for epoch in range(100):
    for batch_idx,batch in enumerate(train_dataloader):
        if batch_idx%100==0:
            print(batch_idx)
        image=batch.cuda()
        t,(x,loss_diffusion) = diffusion_target.few_shot_forward(image)

        with torch.no_grad():
            p_source = torch.zeros((batch_size, batch_size - 1)).cuda()
            x_start_source = diffusion_source.batch_p_sample(x, t, None)
            for i in range(batch_size):
                tmp_c=0
                for j in range(batch_size):
                    if i!=j:
                        anchor_img=x_start_source[i].reshape(-1).unsqueeze(0)
                        compare_img = x_start_source[j].reshape(-1).unsqueeze(0)
                        p_source[i][tmp_c]=cos_sim(anchor_img,compare_img)
                        tmp_c+=1
            p_source=sfm(p_source)
        x_start_target = diffusion_target.batch_p_sample(x, t, None)
        p_target = torch.zeros((batch_size, batch_size - 1)).cuda()
        for i in range(batch_size):
            tmp_c=0
            for j in range(batch_size):
                if i!=j:
                    anchor_img=x_start_target[i].reshape(-1).unsqueeze(0)
                    compare_img = x_start_target[j].reshape(-1).unsqueeze(0)
                    p_target[i][tmp_c]=cos_sim(anchor_img,compare_img)
                    tmp_c+=1
        p_target=sfm(p_target)
        loss_rel = kl_loss(torch.log(p_target), p_source)*opts.beta_kl
        optimizer.zero_grad()
        loss=loss_diffusion+loss_rel
        if batch_idx%10==0:
            print(float(loss_diffusion),float(loss_rel))
        loss.backward()
        optimizer.step()
        if global_step%100==0 and global_step!=0:
            # sampled_images = diffusion_target.sample(batch_size = 24)
            save_image(torch.cat((x,x_start_source,x_start_target),dim=0),os.path.join(opts.output_dir,'images/%d.jpg'%global_step),nrow=batch_size)
        if global_step % 1000 == 0 and global_step!=0:
            torch.save(diffusion_target.state_dict(), os.path.join(opts.output_dir,'models/%d.jpg'%global_step))
            sampled_images = diffusion_target.sample(batch_size = 16)
            save_image(sampled_images,
                       os.path.join(opts.output_dir, 'images/%d-sample.jpg' % global_step), nrow=4)
        global_step += 1