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
#DDPM训练
# parser = ArgumentParser()
# parser.add_argument('--x1', type=str, help='Path to experiment output directory')
# parser.add_argument('--x2', action='store_true', help='Whether to also save inputs + outputs side-by-side')
# opts = parser.parse_args()
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


class Train_Data(Dataset):
    def __init__(self, img_path):
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([128, 128])
        ])
        self.data_path = img_path
        self.file_names = os.listdir(self.data_path)
        self.l=10000
    def __getitem__(self, idx):
        idx=idx%len(self.file_names)
        image = Image.open(os.path.join(self.data_path, self.file_names[idx])).convert('RGB')
        image = self.loader(image)
        return image

    def __len__(self):
        return self.l
#train_data=Train_Data('/home/huteng/dataset/celeba')
name='sunglasses'
train_data=Train_Data('style_img/%s'%name)
train_dataloader = DataLoader(train_data,
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=8,
                                   drop_last=True)

real_data=Train_Data('/home/huteng/dataset/ffhq/images1024x1024')
real_dataloader = DataLoader(real_data,
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=8,
                                   drop_last=True)
real_dataloader_iter=iter(real_dataloader)
diffusion.load_state_dict(torch.load('/home/huteng/DDPM2/pretrained_models/ffhq/400000.pth'))
optizer = Adam(diffusion.parameters(), lr = 1e-4, betas =(0.9, 0.99))
global_step=0
dir='results/fine-tune-model/%s/'%name
os.makedirs(dir+'models',exist_ok=True)
os.makedirs(dir+'images',exist_ok=True)
for epoch in range(100):
    for batch_idx,batch in enumerate(train_dataloader):
        if batch_idx%100==0:
            print(batch_idx)
        image=batch.cuda()
        loss = diffusion(image)
        optizer.zero_grad()
        loss.backward()
        optizer.step()
        global_step+=1
        if global_step%50==0 and global_step!=0:
            torch.save(diffusion.state_dict(),dir+'models/%d.pth'%global_step)
            sampled_images = diffusion.ddim_sample((36, 3, 128, 128), sample_step=50)
            save_image(sampled_images,dir+'images/%d.jpg'%global_step,nrow=6)
            noise_step = 800
            real_image = next(real_dataloader_iter, None)
            if real_image is None:
                real_dataloader_iter = iter(real_dataloader)
                real_image = next(real_dataloader_iter, None)
            real_image = real_image.cuda()
            t = torch.ones(len(real_image)).long().to('cuda') * noise_step
            noises = diffusion.p_losses(real_image, t, return_x=True)
            sampled_images, sampled_middle_images = diffusion.ddim_sample(real_image.shape, sample_step=50,
                                                                                 return_middle=True, start_img=noises,
                                                                                 max_step=noise_step,
                                                                                 min_step=-1,)
            save_image(torch.cat((real_image, noises, sampled_middle_images, sampled_images), dim=0),
                       os.path.join(dir, 'images/%d-sample-middle.jpg' % global_step), nrow=16, normalize=False)
            save_image(sampled_images, os.path.join(dir, 'images/%d--sample.jpg' % global_step),
                       nrow=4, normalize=False)
