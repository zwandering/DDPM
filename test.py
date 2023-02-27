import torch
from model.unet import MyUnet
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
from resizer import Resizer
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
model_target = MyUnet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    self_condition=True,
).cuda()
model_target.prepare(two_stage_step=200)
diffusion = GaussianDiffusion(
    model_target,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
def load_condition_model(path):
    ckpt=torch.load(path)
    keys = ['model.init_conv.weight', 'model.init_conv.bias']
    for i in keys:
        if i == 'model.init_conv.weight':
            ckpt[i] = torch.cat(( torch.zeros_like(ckpt[i]),ckpt[i]), dim=1)
        # if i == 'model.init_conv.bias':
        #     ckpt[i] = ckpt[i] / 2
    for i in list(ckpt.keys()):
        if 'ups' in i:
            i2=i.replace('ups','ups2')
            ckpt[i2]=ckpt[i]
        if 'final' in i:
            i2=i.replace('final','final2')
            ckpt[i2]=ckpt[i]
    return ckpt
batch_size=10
# real_data=Train_Data('/home/huteng/dataset/celeba')
real_data=Train_Data('images/source')
real_dataloader = DataLoader(real_data,
                                   batch_size=batch_size,
                                   shuffle=False,
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
# diffusion0.load_state_dict(torch.load('results/fine-tune-model/sketches/models/1000.pth'))
#diffusion.load_state_dict(torch.load('/home/huteng/DDPM2/results/cartoon/clip_center_loss/step=300,beta=1/models/500.pth'))
diffusion.load_state_dict(torch.load('/home/huteng/DDPM2/results/sketches/clip_center_loss/step=300,beta=1/models/700.pth'))
filter_N=4
filter=transforms.Compose([
            transforms.Resize((128//filter_N,128//filter_N)),
            transforms.Resize((128,128))])
shape = (batch_size, 3, 128,128)
shape_d = (batch_size, 3,128//filter_N, 128//filter_N)
down = Resizer(shape, 1 / filter_N).cuda()
up = Resizer(shape_d, filter_N).cuda()
with torch.no_grad():
    for batch_idx,imgs in enumerate(real_dataloader):
        if batch_idx==1:
            break
        imgs=imgs.to(device)
        noise_step=800
        mid_step=100
        #t, (x, _) = diffusion.few_shot_forward(imgs, step=800)
        # x_start_target1 = (diffusion0.batch_p_sample(x, t, None) + 1) / 2
        # x_start_target2 = (diffusion.batch_p_sample(x, t, None) + 1) / 2
        # save_image(torch.cat((imgs,x,x_start_target1),dim=0),'test0.jpg',nrow=16,normalize=False)
        # save_image(torch.cat((imgs, x, x_start_target2), dim=0), 'test1.jpg', nrow=16, normalize=False)
        t=torch.ones(len(imgs)).long().to(device)*noise_step
        noises = diffusion.p_losses(imgs, t,return_x=True)
        sampled_images1, sampled_middle_images = diffusion.ddim_sample(imgs.shape, sample_step=25, max_step=noise_step,
                                                                       return_middle=True,start_img=noises,condition=style_loss.get_gram_matrix(style_img.repeat(imgs.size(0),1,1,1)))
        #img=diffusion.p_sample_loop(x.shape,img=noises,start_t=noise_step,condition=style_loss.get_gram_matrix(style_img.repeat(imgs.size(0),1,1,1)))
        save_image(sampled_images1,'results/00.jpg',normalize=True)
        self_cond=style_loss.get_gram_matrix(style_img.repeat(imgs.size(0),1,1,1))
        img=noises.detach()
        for step in tqdm(reversed(range(0, noise_step)), desc = 'sampling loop time step', total =noise_step):
            t = torch.ones(len(imgs)).long().to(device) * step
            img, x_start = diffusion.p_sample(img, step, self_cond,loss_fn=None)
            if step >=400:
                tmp_noise=diffusion.p_losses(imgs, t,return_x=True)
                tmp_img,tmp_x_start = diffusion.p_sample(tmp_noise, step, self_cond, loss_fn=None)
                #tmp_img=tmp_noise
                img=img-up(down(img))+up(down(tmp_img))
                if step%10==0:
                    save_image(torch.cat((up(down(img)),up(down(tmp_img))),dim=0),'results/%d-filter.jpg'%step,normalize=True,nrow=16)
                    save_image(x_start, 'results/%d-x0.jpg' % step,nrow=5, normalize=True)
                    # save_image(torch.cat((x_start,tmp_x_start),dim=0), 'results/%d-x0.jpg' % step, normalize=True)
                    save_image(torch.cat((img,tmp_img),dim=0), 'results/%d.jpg'%step,normalize=True)
            elif step%10==0:
                save_image(x_start, 'results/%d-x0.jpg' % step,nrow=5, normalize=True)
        img = (img+1)*0.5

        noises = (noises - noises.min()) / (noises.max() - noises.min())
        save_img = torch.cat((imgs, noises, sampled_middle_images, sampled_images1,img), dim=0)

        print(save_img.min(),save_img.max())
        save_image(save_img,
                   'test.png', nrow=batch_size, normalize=False)