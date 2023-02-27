import torch

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet,default
import torch.nn as nn
#from unet import Unet
from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
from torch.optim import Adam
import numpy as np
from torch import einsum
import clip
from einops import rearrange
from style_loss import loss
import math
#DDPM直接训练
from argparse import ArgumentParser
parser = ArgumentParser()
#parser.add_argument('--output_dir', type=str,default='results', help='Path to experiment output directory')
parser.add_argument('--resume_iter', type=int,default=-1, help='Path to experiment output directory')
parser.add_argument('--beta_f', type=str,default='100', help='Path to experiment output directory')
parser.add_argument('--beta_style', type=str,default='1', help='Path to experiment output directory')
parser.add_argument('--style', type=str,default='sketches', help='Path to experiment output directory')
parser.add_argument('--noise_step', type=int,default=800, help='Path to experiment output directory')
parser.add_argument('--clip_mode', type=int,default=0, help='0:clip direction loss;1:clip center loss')
opts = parser.parse_args()
class Attention(nn.Module):
    def __init__(self, dim,dim_out, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim_out, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)
class mean_var_mapping(nn.Module):
    def __init__(self, size=16,dim_in=512):
        super().__init__()
        n=int(math.log2(size))
        self.n=n
        model=[]
        for i in range(n-1):
            model.append(nn.Conv2d(dim_in,dim_in,3,2,1))
            model.append(nn.BatchNorm2d(dim_in))
            model.append(nn.LeakyReLU())
        self.net=nn.Sequential(*model)
        self.linear1=nn.Linear(dim_in*2*2,dim_in)
        self.linear2=nn.Linear(dim_in*2*2,dim_in)
    def forward(self, x):
        x=self.net(x)
        x=torch.flatten(x,1)
        mean=self.linear1(x)
        var=self.linear2(x)
        return mean,var
class my_unet(Unet):
    def prepare(self):
        in_out=[(64,64),(64,64),(128,128),(256,256)]
        self.attentions = nn.ModuleList([])
        # for ind, (dim_in, dim_out) in enumerate(in_out):
        #     self.attentions.append(Attention(dim_in, dim_out))
        mid_dim=512
        self.attentions.append(Attention(mid_dim, mid_dim))
        #self.attentions.append(mean_var_mapping())    #虚假的attention，只是卷积
        # for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
        #     self.attentions.append(Attention(dim_out, dim_in))
    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        cnt=0
        for block1, block2, attn, downsample in self.downs:
            # print(x.shape)
            # tmp=self.attentions[cnt](x)
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)
            # print('hh')
            # print(x.shape)
            # print(tmp.shape)
            # x = x + tmp
            #cnt += 1
            x = downsample(x)
        print(x.shape)
        tmp=self.attentions[0](x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        print(x.shape,tmp.shape)
        x=x+tmp
        # mean,var=self.attentions[0](x)
        # mean=mean.unsqueeze(-1).unsqueeze(-1)
        # var=var.unsqueeze(-1).unsqueeze(-1)
        # x_mean=x.flatten(2).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
        # x_var=x.flatten(2).var(dim=-1).unsqueeze(-1).unsqueeze(-1)
        # x=(x-x_mean)/x_var
        # x=x*var+mean
        #cnt += 1
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)
            # x = x + self.attentions[cnt](x)
            # cnt += 1
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
model_source = my_unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()
model_source.prepare()
diffusion_source = GaussianDiffusion(
    model_source,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
model_target = my_unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()
model_target.prepare()
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
def load_model(path):
    # keys0=["model.downs.0.3.weight", "model.downs.0.3.bias",
    #     "model.downs.1.3.weight", "model.downs.1.3.bias", "model.downs.2.3.weight", "model.downs.2.3.bias"]
    # keys1 = ["model.downs.0.3.1.weight", "model.downs.0.3.1.bias",
    #         "model.downs.1.3.1.weight", "model.downs.1.3.1.bias", "model.downs.2.3.1.weight", "model.downs.2.3.1.bias"]
    ckpt=torch.load(path)
    # for i in range(len(keys0)):
    #     print(keys0[i],ckpt[keys0[i]].shape)
    #     ckpt[keys1[i]]=ckpt[keys0[i]]
    #     ckpt.pop(keys0[i])
    for k,v in diffusion_target.named_parameters():
        if 'attention' in k:
            ckpt[k]=torch.randn_like(v)*0.01
    return ckpt
batch_size =16
clip_model=Clip()
style_loss = loss.VGGStyleLoss(transfer_mode=1, resize=True).cuda()
#content_loss=loss.VGGPerceptualLoss()
name=opts.style
train_data=Train_Data('style_img/%s'%name)
real_data=Train_Data('/home/huteng/dataset/celeba')
param=list(diffusion_target.parameters())
for k,v in diffusion_target.named_parameters():
    v.requires_grad = False
for k,v in diffusion_target.named_parameters():
    if 'attention' in k:
        v.requires_grad = True
params = filter(lambda p: p.requires_grad, diffusion_target.parameters())
features_source0=torch.from_numpy(np.load('draw/features-real_img.npy')).cuda().mean(0)
features_target0=torch.from_numpy(np.load('draw/features-%s.npy'%name)).cuda().mean(0)
#features_target0=torch.from_numpy(np.load('draw/features-%s.npy'%name)).cuda()[0:1]
feature_dir=(features_target0-features_source0).type(torch.HalfTensor).cuda().unsqueeze(0)
print(feature_dir.shape,feature_dir.dtype)
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
diffusion_source.load_state_dict(load_model('pretrained_models/481157.pth'),strict=False)
diffusion_target.load_state_dict(load_model('pretrained_models/481157.pth'),strict=False)
save_dir='results/%s/clip_dir_loss'%name if opts.clip_mode==0 else 'results/%s/clip_center_loss'%name
save_dir=os.path.join(save_dir,'step=%d,beta=%s'%(opts.noise_step,opts.beta_f))
if opts.resume_iter!=-1:
    diffusion_target.load_state_dict(torch.load(os.path.join(save_dir,'models','%d.pth'%opts.resume_iter)))
optimizer = Adam(params, lr = 1e-4, betas =(0.9, 0.99))
global_step=0 if opts.resume_iter==-1 else opts.resume_iter
mse_loss=torch.nn.MSELoss()
print(mse_loss(torch.zeros_like(feature_dir).cuda(),feature_dir))
cos_loss=torch.nn.CosineSimilarity(dim=1)
os.makedirs(os.path.join(save_dir,'models'),exist_ok=True)
os.makedirs(os.path.join(save_dir,'images'),exist_ok=True)
opts.beta_f=float(opts.beta_f)
opts.beta_style=float(opts.beta_style)
indices=[batch_size-i-1 for i in range(batch_size)]
for epoch in range(100):
    for batch_idx,batch in enumerate(train_dataloader):
        if batch_idx%100==0:
            print(batch_idx)
        image=batch.cuda()
        t,(x,loss_diffusion) = diffusion_target.few_shot_forward(image)
        loss=loss_diffusion
        if opts.beta_f!=0 and (global_step<=100 or (global_step>100 and global_step%1==0)):
            real_image = next(real_dataloader_iter, None)
            if real_image is None:
                real_dataloader_iter = iter(real_dataloader)
                real_image = next(real_dataloader_iter, None)
            real_image = real_image.cuda()
            with torch.no_grad():
                #x_start_source=diffusion_target.ddim_sample(x.shape, sample_step=25,start_img=x,start_step=t[0])
                t, (x, _) = diffusion_source.few_shot_forward(real_image,step=opts.noise_step)
                #x_start_source = (diffusion_source.batch_p_sample(x, t, None)+1)/2
                feature_source=clip_model.encode_img(real_image)
            x_start_target = (diffusion_target.batch_p_sample(x, t, None)+1)/2
            # #loss_feature = content_loss(x_start_target,x_start_source).mean()
            feature_target=clip_model.encode_img(x_start_target)
            if opts.clip_mode==0:
                loss_feature = (1-cos_loss(feature_target-feature_source, feature_dir.repeat(batch_size,1))).mean()
            else:
                feature_source_to_target = feature_source + feature_dir.repeat(batch_size, 1)
                loss_feature = mse_loss(feature_target, feature_source_to_target)
                #loss_feature = mse_loss(feature_target, features_target0.repeat(feature_target.size(0),1))
            if opts.beta_style!=0:
                loss_style=style_loss(x_start_target,image).mean()*opts.beta_style
            else:
                loss_style=0
            loss_feature*= opts.beta_f
        else:
            loss_feature=0
            loss_style=0
        optimizer.zero_grad()
        loss=loss_diffusion+loss_feature+loss_style
        if global_step%10==0:
            print(global_step,float(loss_diffusion),float(loss_feature),float(loss_style))
            #print(global_step,float(mse_loss(feature_target,feature_target[indices])),float(mse_loss(feature_target, features_target0)), float(mse_loss(feature_target, feature_source)))
        loss.backward()
        optimizer.step()
        if global_step%50==0 and global_step!=0:
            # for k, v in diffusion_target.named_parameters():
            #     if 'attention' in k:
            #         print(k, float(v.min()), float(v.mean()), float(v.max()))
            if opts.beta_f==0:
                sampled_images = diffusion_target.ddim_sample((16, 3, 128, 128), sample_step=50)
                save_image(sampled_images, os.path.join(save_dir, 'images/%d-sample-random-sample.jpg' % global_step),
                           nrow=4, normalize=False)
            else:
                if opts.clip_mode==0:
                # sampled_images = diffusion_target.sample(batch_size = 24)
                    sampled_images = diffusion_target.ddim_sample((16, 3, 128, 128), sample_step=50)
                    save_image(sampled_images,os.path.join(save_dir, 'images/%d-sample-random-sample.jpg' % global_step), nrow=4, normalize=False)
                save_image(torch.cat((real_image,(x+1)/2,x_start_target),dim=0),os.path.join(save_dir,'images/%d.jpg'%global_step),nrow=batch_size,normalize=False)
                noise_step=800
                t = torch.ones(len(real_image)).long().to('cuda') * noise_step
                noises = diffusion_target.p_losses(real_image, t, return_x=True)
                sampled_images,sampled_middle_images = diffusion_target.ddim_sample(x.shape, sample_step=50,return_middle=True,start_img=noises, max_step=noise_step,
                                                                            min_step=-1)
                save_image(torch.cat((real_image,noises,sampled_middle_images,sampled_images),dim=0),
                           os.path.join(save_dir, 'images/%d-sample.jpg' % global_step), nrow=16,normalize=False)
                if opts.clip_mode==1:
                    save_image(sampled_images, os.path.join(save_dir, 'images/%d-sample-random-sample.jpg' % global_step),
                               nrow=4, normalize=False)
        if global_step % 100 == 0 and global_step!=0:
            torch.save(diffusion_target.state_dict(), os.path.join(save_dir,'models/%d.pth'%global_step))
        global_step += 1
#CUDA_VISIBLE_DEVICES=1 python3 train-clip.py --beta_f=1 --noise_step=0 --output_dir=results/clip_dir_loss/step=0,beta=1
