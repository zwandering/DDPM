import torch
from model.big_unet import create_model
from model.ddpm import GaussianDiffusion
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image,make_grid
import os
from collections import namedtuple
from style_loss import ada_loss,loss
from utils import read_img
from torchvision import transforms
from resizer import Resizer
#diffusion直接采样
image_size=256
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
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

model_target = create_model(
    image_size=image_size
).cuda()
diffusion = GaussianDiffusion(
    model_target,
    image_size = image_size,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
batch_size=16
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
Ada_loss.set_input(style_img.repeat(batch_size,1,1,1), style_img.repeat(batch_size,1,1,1))
diffusion.model.load_state_dict(torch.load('pretrained_models/ffhq_p2.pt'))
cnt=0
resize=transforms.Resize((256,256))
style_name='cartoon'
os.makedirs('results/generated/%s/source'%style_name,exist_ok=True)
os.makedirs('results/generated/%s/out'%style_name,exist_ok=True)
with torch.no_grad():
    for batch_idx,imgs in enumerate(real_dataloader):
        if batch_idx==60:
            break
        for noise_step in [200,400,600,800,1000]:
            imgs=imgs.to(device)
            #noise_step=800
            mid_step=100
            #t, (x, _) = diffusion.few_shot_forward(imgs, step=800)
            # x_start_target1 = (diffusion0.batch_p_sample(x, t, None) + 1) / 2
            # x_start_target2 = (diffusion.batch_p_sample(x, t, None) + 1) / 2
            # save_image(torch.cat((imgs,x,x_start_target1),dim=0),'test0.jpg',nrow=16,normalize=False)
            # save_image(torch.cat((imgs, x, x_start_target2), dim=0), 'test1.jpg', nrow=16, normalize=False)
            t=torch.ones(len(imgs)).long().to(device)*noise_step
            noises = diffusion.p_losses(imgs, t,return_x=True)
            sampled_images1, sampled_middle_images = diffusion.ddim_sample(imgs.shape, sample_step=50, max_step=noise_step,
                                                                           return_middle=True,start_img=noises,condition=None)
            print(noise_step)
            save_image(torch.cat((imgs,noises,sampled_images1)),'results/step=%d.jpg'%noise_step,normalize=False)
        #img=diffusion.p_sample_loop(x.shape,img=noises,start_t=noise_step,condition=style_loss.get_gram_matrix(style_img.repeat(imgs.size(0),1,1,1)))
        # for i in range(len(sampled_images1)):
        #     grid = make_grid(imgs[i], nrow=1, normalize=True, padding=0)
        #     save_image(grid, 'results/generated/%s/source/%d.jpg'%(style_name,cnt))
        #     grid=make_grid(sampled_images1[i],nrow=1,normalize=True,padding=0)
        #     save_image(grid,'results/generated/%s/out/%d.jpg'%(style_name,cnt))
        #     cnt+=1
        #save_image(sampled_images1,'results/00.jpg',normalize=True)