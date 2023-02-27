import torch
from denoising_diffusion_pytorch import Unet

from model.ddpm import GaussianDiffusion
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
from collections import namedtuple
from tqdm.auto import tqdm
from style_loss import ada_loss,loss
from utils import read_img
from torchvision import transforms
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
diffusion.load_state_dict(torch.load('/home/huteng/DDPM2/results/cartoon/clip_center_loss/step=300,beta=1/models/500.pth'))
filter_N=8
filter=transforms.Compose([
            transforms.Resize((128//filter_N,128//filter_N)),
            transforms.Resize((128,128))])
with torch.no_grad():
    for batch_idx,imgs in enumerate(real_dataloader):
        if batch_idx==1:
            break
        imgs=imgs.to(device)
        noise_step=800
        mid_step=100
        t, (x, _) = diffusion0.few_shot_forward(imgs, step=800)
        # x_start_target1 = (diffusion0.batch_p_sample(x, t, None) + 1) / 2
        # x_start_target2 = (diffusion.batch_p_sample(x, t, None) + 1) / 2
        # save_image(torch.cat((imgs,x,x_start_target1),dim=0),'test0.jpg',nrow=16,normalize=False)
        # save_image(torch.cat((imgs, x, x_start_target2), dim=0), 'test1.jpg', nrow=16, normalize=False)
        t=torch.ones(len(imgs)).long().to(device)*noise_step
        noises = diffusion.p_losses(imgs, t,return_x=True)
        sampled_images1, sampled_middle_images = diffusion.ddim_sample(imgs.shape, sample_step=25, max_step=noise_step,
                                                                       return_middle=True,start_img=noises,condition=style_loss.get_gram_matrix(style_img.repeat(imgs.size(0),1,1,1)))
        #img=diffusion.p_sample_loop(x.shape,img=noises,start_t=noise_step,condition=style_loss.get_gram_matrix(style_img.repeat(imgs.size(0),1,1,1)))
        save_image(imgs,'results/00.jpg',normalize=True)




        #ddim process
        times = torch.linspace(0, noise_step - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        if start_img is not None:
            # print(start_img.min(), start_img.max())
            if start_img.min() >= 0 and start_img.max() <= 1:
                img = start_img * 2 - 1
            else:
                img = start_img
            batch = img.size(0)
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        middle_img = None
        self_cond = condition
        for iter, (time, time_next) in enumerate(tqdm(time_pairs, desc='sampling loop time step')):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=clip_denoised)
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            # sigma=0
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            if return_middle and (iter % 5 == 0):
                if middle_img is None:
                    middle_img = my_unnormalize_to_zero_to_one(x_start.clone())
                else:
                    middle_img = torch.cat((middle_img, my_unnormalize_to_zero_to_one(x_start.clone())), dim=0)
            # save_image(img,'results/%d.jpg'%time,normalize=True)
        if min_step == -1:
            img = unnormalize_to_zero_to_one(img)
        # print('img range', img.min(), img.max())
        if return_middle:
            return img, middle_img
        return img

        noises = (noises - noises.min()) / (noises.max() - noises.min())
        save_img = torch.cat((imgs, noises, sampled_middle_images, sampled_images1), dim=0)

        print(save_img.min(),save_img.max())
        save_image(save_img,
                   'test.png', nrow=batch_size, normalize=False)