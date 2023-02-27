import torch
from denoising_diffusion_pytorch import Unet
from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
from style_loss import loss
from torch.optim import Adam
#DDPM训练
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--save_dir', type=str,default='results', help='Path to experiment output directory')
opts = parser.parse_args()
# print(opts.x1,opts.x2)
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()
diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
# def adjust_images(imgs,add_num=10):
#     l=len(imgs)
#     content_losses = torch.zeros((imgs.size(0), real_imgs.size(0))).cuda()
#     for i in range(len(real_imgs)):
#         tmp_img = real_imgs[i:i + 1]
#         content_losses[:, i] = content_loss(imgs, tmp_img)
#     content_losses = content_losses.min(dim=1).values
#     _,content_idx=torch.sort(content_losses, descending=False)
#     new_imgs=[]
#     for i in range(len(imgs)):
#         if i not in content_idx[:(sample_size-add_num)//2]:
#             new_imgs.append(imgs[i])
#     imgs=torch.stack(new_imgs)
#     style_losses = torch.zeros((imgs.size(0), real_imgs.size(0))).cuda()
#     for i in range(len(real_imgs)):
#         tmp_img = real_imgs[i:i+1]
#         style_losses[:,i]=style_loss(imgs,tmp_img)
#     style_losses=style_losses.mean(dim=1)
#     _, style_idx = torch.sort(style_losses, descending=False)
#     new_imgs=[]
#     for i in style_idx[:l-sample_size+add_num]:
#         new_imgs.append(imgs[i])
#     new_imgs=torch.stack(new_imgs,dim=0)
#     return new_imgs
def adjust_images(imgs,add_num=2):
    l=len(imgs)
    style_losses = torch.zeros((imgs.size(0), real_imgs.size(0))).cuda()
    for i in range(len(real_imgs)):
        tmp_img = real_imgs[i:i + 1]
        style_losses[:, i] = style_loss(imgs, tmp_img)
    style_losses = style_losses.mean(dim=1)
    _, style_idx = torch.sort(style_losses, descending=True)
    new_imgs = []
    for i in range(len(imgs)):
        if i not in style_idx[:(sample_size-add_num)//2]:
            new_imgs.append(imgs[i])
    imgs = torch.stack(new_imgs)
    content_losses = torch.zeros((imgs.size(0), real_imgs.size(0))).cuda()
    for i in range(len(real_imgs)):
        tmp_img = real_imgs[i:i + 1]
        content_losses[:, i] = content_loss(imgs, tmp_img)
    content_losses = content_losses.min(dim=1).values
    _,content_idx=torch.sort(content_losses, descending=True)
    new_imgs=[]
    for i in content_idx[:l-sample_size+add_num]:
        new_imgs.append(imgs[i])
    new_imgs=torch.stack(new_imgs,dim=0)
    return new_imgs



class Train_Data(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs.cpu().detach()
        save_image(self.imgs[20:],os.path.join(opts.save_dir,'images/%d-dataset.jpg'%epoch),nrow=10)
        self.l=160
    def __getitem__(self, idx):
        idx=idx%len(self.imgs)
        return self.imgs[idx]
    def __len__(self):
        return self.l
dir='/home/huteng/DDPM2/style_img/Raphael'
file_names=os.listdir(dir)
real_imgs=[]
for file_name in file_names:
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        img = Image.open(os.path.join(dir, file_name))
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Resize((128,128))(img)
        real_imgs.append(img)
real_imgs=torch.stack(real_imgs,dim=0).cuda()
imgs=real_imgs.clone()
diffusion.load_state_dict(torch.load('/home/huteng/kousiqi/result/model/481157.pth'))
optizer = Adam(diffusion.parameters(), lr = 1e-4, betas =(0.9, 0.99))
global_step=0
style_loss = loss.VGGStyleLoss(transfer_mode=1, resize=True).cuda()
content_loss=loss.VGGPerceptualLoss()
sample_size=50
for epoch in range(100000):
    train_data = Train_Data(torch.cat((real_imgs.repeat(3,1,1,1),imgs),dim=0))
    train_dataloader = DataLoader(train_data,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)
    for batch_idx,batch in enumerate(train_dataloader):
        if batch_idx%10==0:
            print(batch_idx)
        image=batch.cuda()
        loss = diffusion(image)
        optizer.zero_grad()
        loss.backward()
        optizer.step()
        global_step+=1
    with torch.no_grad():
        sampled_images = diffusion.ddim_sample((sample_size, 3, 128, 128), sample_step=10)
        print(float(real_imgs.max()),float(real_imgs.min()),float(real_imgs.mean()),float(sampled_images.max()),float(sampled_images.min()),float(sampled_images.mean()))
        torch.save(diffusion.state_dict(), '%s/models/%d.pth' % (opts.save_dir, epoch))
        save_image(sampled_images, '%s/images/%d-sample.jpg' % (opts.save_dir, epoch), nrow=sample_size//5)
        sampled_images=adjust_images(sampled_images)
        if epoch==0:
            imgs=sampled_images.detach()
        else:
            imgs=torch.cat((imgs,sampled_images.detach()),dim=0)
            imgs=adjust_images(imgs)
