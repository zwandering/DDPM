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
import numpy as np
import clip
from style_loss import loss
from criteria import id_loss
#DDPM直接训练
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--output_dir', type=str,default='results', help='Path to experiment output directory')
parser.add_argument('--resume_iter', type=int,default=-1, help='Path to experiment output directory')
parser.add_argument('--beta_f', type=float,default=100, help='Path to experiment output directory')
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

class Clip:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        #print(self.preprocess)
        self.transfroms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    def encode_text(self,text_input):
        return self.model.encode_text(clip.tokenize(text_input).to(self.device))
    def encode_img(self,img):
        return self.model.encode_image(self.transfroms(img))
    def forward(self,img,text):
        image = self.transfroms(img)
        text = clip.tokenize([text]).to(self.device)
        logits_per_image, logits_per_text = self.model(image, text)
        #probs = logits_per_image.softmax(dim=-1)
        return -logits_per_image
class Data(Dataset):
    def __init__(self, img_path):
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224])
        ])
        self.data_path = img_path
        self.file_names = os.listdir(self.data_path)
        self.l=len(self.file_names)
    def __getitem__(self, idx):
        idx=idx%len(self.file_names)
        image = Image.open(os.path.join(self.data_path, self.file_names[idx])).convert('RGB')
        image = self.loader(image)
        return image

    def __len__(self):
        return self.l
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
batch_size =16
clip_model=Clip()
content_loss=loss.VGGPerceptualLoss()
train_data=Train_Data('style_img/sketches')
real_data=Train_Data('/home/huteng/dataset/celeba')
features_source=torch.from_numpy(np.load('draw/features-arcface1.npy')).cuda().mean(0)
features_target=torch.from_numpy(np.load('draw/features-arcface2.npy')).cuda().mean(0)
feature_dir=(features_target-features_source).cuda()
real_dataloader = DataLoader(real_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=8,
                                   drop_last=True)
real_dataloader_iter=iter(real_dataloader)
train_dataloader = DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=8,
                                   drop_last=True)
diffusion_source.load_state_dict(torch.load('pretrained_models/481157.pth'))
diffusion_target.load_state_dict(torch.load('pretrained_models/481157.pth'))
if opts.resume_iter!=-1:
    diffusion_target.load_state_dict(torch.load(os.path.join(opts.output_dir,'models','%d.pth'%opts.resume_iter)))
optimizer = Adam(diffusion_target.parameters(), lr = 1e-4, betas =(0.9, 0.99))
global_step=0 if opts.resume_iter==-1 else opts.resume_iter
id_model = id_loss.IDLoss().cuda()
mse_loss=torch.nn.MSELoss()
print(mse_loss(torch.zeros_like(feature_dir).cuda(),feature_dir))
cos_loss=torch.nn.CosineSimilarity(dim=1)
for epoch in range(100):
    for batch_idx,batch in enumerate(train_dataloader):
        if batch_idx%100==0:
            print(batch_idx)
        image=batch.cuda()
        t,(x,loss_diffusion) = diffusion_target.few_shot_forward(image)
        loss=loss_diffusion
        if global_step>10:
            real_image = next(real_dataloader_iter, None)
            if real_image is None:
                real_dataloader_iter = iter(real_dataloader)
                real_image = next(real_dataloader_iter, None)
            real_image = real_image.cuda()
            with torch.no_grad():
                #x_start_source=diffusion_target.ddim_sample(x.shape, sample_step=25,start_img=x,start_step=t[0])
                t, (x, _) = diffusion_source.few_shot_forward(real_image,step=800)
                #x_start_source = (diffusion_source.batch_p_sample(x, t, None)+1)/2
                feature_source=id_model.encode_img(real_image)
            feature_source_to_target=feature_source+feature_dir.repeat(batch_size,1)
            x_start_target = (diffusion_target.batch_p_sample(x, t, None)+1)/2
            # #loss_feature = content_loss(x_start_target,x_start_source).mean()
            feature_target=id_model.encode_img(x_start_target)
            loss_feature=mse_loss(feature_target,feature_source_to_target)*opts.beta_f
            #loss_feature = (1-cos_loss(feature_target-feature_source, feature_dir.repeat(batch_size,1))).mean()
            loss_feature*= opts.beta_f
        else:
            loss_feature=0
        optimizer.zero_grad()
        loss=loss_diffusion+loss_feature
        if batch_idx%10==0:
            print(global_step,float(loss_diffusion),float(loss_feature))
        loss.backward()
        optimizer.step()
        if global_step%100==0 and global_step!=0:
            # sampled_images = diffusion_target.sample(batch_size = 24)
            sampled_images = diffusion_target.ddim_sample((16, 3, 128, 128), sample_step=25)
            save_image(sampled_images,os.path.join(opts.output_dir, 'images/%d-sample-random-sample.jpg' % global_step), nrow=16, normalize=False)
            save_image(torch.cat((real_image,x_start_target),dim=0),os.path.join(opts.output_dir,'images/%d.jpg'%global_step),nrow=batch_size,normalize=False)
            sampled_images,sampled_middle_images = diffusion_target.ddim_sample(x.shape, sample_step=50,return_middle=True)
            save_image(torch.cat((sampled_middle_images,sampled_images),dim=0),
                       os.path.join(opts.output_dir, 'images/%d-sample.jpg' % global_step), nrow=16,normalize=False)
        if global_step % 500 == 0 and global_step!=0:
            torch.save(diffusion_target.state_dict(), os.path.join(opts.output_dir,'models/%d.pth'%global_step))
        global_step += 1