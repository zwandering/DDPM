import torch
from denoising_diffusion_pytorch import Unet
from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import os
from torch.optim import Adam
#DDPM训练
# parser = ArgumentParser()
# parser.add_argument('--x1', type=str, help='Path to experiment output directory')
# parser.add_argument('--x2', action='store_true', help='Whether to also save inputs + outputs side-by-side')
# opts = parser.parse_args()
# print(opts.x1,opts.x2)
diffusions=[]
for i in range(10):
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
    diffusions.append(diffusion)


dir='/home/huteng/DDPM2/style_img/sketches'
loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([128, 128])
        ])
file_names=os.listdir(dir)
imgs=[]
for file_name in file_names:
    file_path=os.path.join(dir,file_name)
    img=Image.open(file_path).convert('RGB')
    imgs.append(loader(img))

diffusion.load_state_dict(torch.load('/home/huteng/kousiqi/result/model/481157.pth'))
optizer = Adam(diffusion.parameters(), lr = 1e-4, betas =(0.9, 0.99))
global_step=0
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
        if global_step%200==0 and global_step!=0:
            torch.save(diffusion.state_dict(),'/home/huteng/DDPM2/results/models/%d.pth'%global_step)
            sampled_images = diffusion.sample(batch_size = 36)
            save_image(sampled_images,'/home/huteng/DDPM2/results/images/%d.jpg'%global_step,nrow=4)