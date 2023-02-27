import torch
from denoising_diffusion_pytorch import Unet

from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
from collections import namedtuple
from style_loss import ada_loss,loss
from utils import read_img

#diffusion直接采样
class Train_Data(Dataset):
    def __init__(self, img_path):
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([128, 128])
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
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    self_condition=True
).cuda()
diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
model0 = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()
diffusion0 = GaussianDiffusion(
    model0,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
batch_size=16
real_data=Train_Data('/home/huteng/dataset/celeba')
real_dataloader = DataLoader(real_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=8,
                                   drop_last=True)
device='cuda'
dir='style_img/sketches'
file_names=os.listdir(dir)
imgs=[]
batch_size=16
Ada_loss = ada_loss.AdaAttNModel()
style_loss = loss.VGGStyleLoss(transfer_mode=1, resize=True).cuda()
style_img=read_img('/home/huteng/DDPM2/style_img/sketches/004_1_1_sz1.jpg').cuda()
#style_loss = loss.VGGStyleLoss(transfer_mode=1, resize=True,style_img=style_img).cuda()
#content_loss=loss.VGGPerceptualLoss()
Ada_loss.set_input(style_img.repeat(batch_size,1,1,1), style_img.repeat(batch_size,1,1,1))
diffusion0.load_state_dict(torch.load('results/fine-tune-model/sketches/models/1000.pth'))
# diffusion.load_state_dict(torch.load('results/cartoon/clip_center_loss/step=500,beta=0.3/models/300.pth'))
#diffusion0.load_state_dict(torch.load('/home/huteng/DDPM2/results/cartoon/clip_dir_loss/step=700,beta=0.5/models/200.pth'))
diffusion.load_state_dict(torch.load('/home/huteng/DDPM2/results/sketches/clip_center_loss/step=300,beta=1/models/500.pth'))
with torch.no_grad():
    for batch_idx,imgs in enumerate(real_dataloader):
        if batch_idx==1:
            break
        imgs=imgs.to(device)
        noise_step=800
        mid_step=100
        t, (x, _) = diffusion0.few_shot_forward(imgs, step=800)
        x_start_target1 = (diffusion0.batch_p_sample(x, t, None) + 1) / 2
        x_start_target2 = (diffusion.batch_p_sample(x, t, None) + 1) / 2
        save_image(torch.cat((imgs,x,x_start_target1),dim=0),'test0.jpg',nrow=16,normalize=False)
        save_image(torch.cat((imgs, x, x_start_target2), dim=0), 'test1.jpg', nrow=16, normalize=False)
        t=torch.ones(len(imgs)).long().to(device)*noise_step
        #noises=torch.randn_like(noises).to(noises.device)
        #img0=diffusion0.p_sample_loop(img=noises)
        noises = diffusion.p_losses(imgs, t,return_x=True)
        sampled_images1, sampled_middle_images1 = diffusion.ddim_sample(imgs.shape, sample_step=25, max_step=noise_step,
                                                                        min_step=mid_step, return_middle=True,start_img=noises,condition=style_loss.get_gram_matrix(style_img.repeat(imgs.size(0),1,1,1)))
        sampled_images, sampled_middle_images2 = diffusion0.ddim_sample(imgs.shape, sample_step=25,
                                                                        max_step=mid_step,
                                                                        min_step=-1, start_img=sampled_images1,
                                                                        return_middle=True)
        noises=(noises-noises.min())/(noises.max()-noises.min())
        print(sampled_middle_images1.shape)
        save_img=torch.cat((imgs,noises,sampled_middle_images1, sampled_middle_images2),dim=0)
        print(save_img.min(),save_img.max())
        save_image(save_img,
                   'test.png', nrow=batch_size, normalize=False)
        save_image(sampled_images,'test2.jpg',nrow=4,normalize=False)