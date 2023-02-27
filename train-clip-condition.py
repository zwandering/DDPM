import torch
# from denoising_diffusion_pytorch import Unet
from model.unet import MyUnet
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
from argparse import ArgumentParser
#在train-clip基础上以风格图像为condition加入训练
parser = ArgumentParser()
#parser.add_argument('--output_dir', type=str,default='results', help='Path to experiment output directory')
parser.add_argument('--resume_iter', type=int,default=-1, help='Path to experiment output directory')
parser.add_argument('--beta_f', type=str,default='1', help='Path to experiment output directory')
parser.add_argument('--beta_style', type=str,default='1', help='Path to experiment output directory')
parser.add_argument('--style', type=str,default='sketches', help='Path to experiment output directory')
parser.add_argument('--noise_step', type=int,default=300, help='Path to experiment output directory')
parser.add_argument('--clip_mode', type=int,default=1, help='0:clip direction loss;1:clip center loss')
opts = parser.parse_args()
model_source = MyUnet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    self_condition=True,
).cuda()
model_source.prepare(two_stage_step=opts.noise_step)
diffusion_source = GaussianDiffusion(
    model_source,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
model_target = MyUnet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    self_condition=True,
).cuda()
model_target.prepare(two_stage_step=opts.noise_step)
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
batch_size =16
clip_model=Clip()
style_loss = loss.VGGStyleLoss(transfer_mode=1, resize=True).cuda()
x1=torch.zeros(2,3,128,128).cuda()
x2=torch.zeros(2,3,128,128).cuda()
#content_loss=loss.VGGPerceptualLoss()
name=opts.style
train_data=Train_Data('style_img/%s'%name)
real_data=Train_Data('/home/huteng/dataset/ffhq/images1024x1024')
features_source=torch.from_numpy(np.load('draw/features-ffhq.npy')).cuda().mean(0)
features_target=torch.from_numpy(np.load('draw/features-%s.npy'%name)).cuda().mean(0)
features_target0=torch.from_numpy(np.load('draw/features-%s.npy'%name)).cuda()[0:1]
feature_dir=(features_target-features_source).type(torch.HalfTensor).cuda().unsqueeze(0)
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
# diffusion_source.load_state_dict(load_condition_model('pretrained_models/481157.pth'))
# diffusion_target.load_state_dict(load_condition_model('pretrained_models/481157.pth'))
diffusion_source.load_state_dict(load_condition_model('/home/huteng/DDPM2/pretrained_models/ffhq/400000.pth'))
diffusion_target.load_state_dict(load_condition_model('/home/huteng/DDPM2/pretrained_models/ffhq/400000.pth'))
save_dir='results/%s/clip_dir_loss'%name if opts.clip_mode==0 else 'results/%s/clip_center_loss'%name
save_dir=os.path.join(save_dir,'step=%d,beta_f=%s,beta_s=%s'%(opts.noise_step,opts.beta_f,opts.beta_style))
if opts.resume_iter!=-1:
    diffusion_target.load_state_dict(torch.load(os.path.join(save_dir,'models','%d.pth'%opts.resume_iter)))
optimizer = Adam(diffusion_target.parameters(), lr = 1e-4, betas =(0.9, 0.99))
global_step=0 if opts.resume_iter==-1 else opts.resume_iter
mse_loss=torch.nn.MSELoss(reduction='none')
print(mse_loss(torch.zeros_like(feature_dir).cuda(),feature_dir).mean())
cos_loss=torch.nn.CosineSimilarity(dim=1)
os.makedirs(os.path.join(save_dir,'models'),exist_ok=True)
os.makedirs(os.path.join(save_dir,'images'),exist_ok=True)
opts.beta_f=float(opts.beta_f)
opts.beta_style=float(opts.beta_style)
for epoch in range(100):
    for batch_idx,batch in enumerate(train_dataloader):
        # tmp = diffusion_target.model.init_conv.weight[:, 3:, :, :]
        # print(tmp.min(), tmp.max(), tmp.mean())
        if batch_idx%100==0:
            print(batch_idx)
        image=batch.cuda()
        # t,(x,loss_diffusion) = diffusion_target.few_shot_forward(image,x_self_cond=style_loss.get_gram_matrix(image.detach()))
        # loss=loss_diffusion
        if global_step%2==0:
        # if global_step<=100 or (global_step>100 and global_step%3==0) :
            real_image = next(real_dataloader_iter, None)
            if real_image is None:
                real_dataloader_iter = iter(real_dataloader)
                real_image = next(real_dataloader_iter, None)
            real_image = real_image.cuda()
            with torch.no_grad():
                #x_start_source=diffusion_target.ddim_sample(x.shape, sample_step=25,start_img=x,start_step=t[0])
                t, (x, _) = diffusion_source.few_shot_forward(real_image,step=opts.noise_step,x_self_cond=style_loss.get_gram_matrix(image).detach())
                #x_start_source = (diffusion_source.batch_p_sample(x, t, None)+1)/2
                feature_source=clip_model.encode_img(real_image)
            x_start_target = (diffusion_target.batch_p_sample(x, t, x_self_cond=style_loss.get_gram_matrix(image).detach())+1)/2
            # #loss_feature = content_loss(x_start_target,x_start_source).mean()
            feature_target=clip_model.encode_img(x_start_target)
            if opts.clip_mode==0:
                loss_feature = (1-cos_loss(feature_target-feature_source, feature_dir.repeat(batch_size,1))).mean()
            else:
                feature_source_to_target = feature_source + feature_dir.repeat(batch_size, 1)
                loss_feature = mse_loss(feature_target, feature_source_to_target).mean(-1)
                #loss_feature = mse_loss(feature_target, features_target0.repeat(feature_target.size(0),1))
            if opts.beta_style!=0:
                loss_style=style_loss(x_start_target,image)*opts.beta_style
            else:
                loss_style=0
            loss_feature*= opts.beta_f
            t2, (x2, loss_diffusion) = diffusion_target.few_shot_forward(image, t=t,x_self_cond=style_loss.get_gram_matrix(
                image).detach())
            dishu = 20
            alpha = dishu ** (t / 1000)
            loss_diffusion = ((dishu ** 0.9 - alpha) * loss_diffusion).mean() * 3
            loss_style = (alpha * loss_style).mean()
            loss_feature = (alpha * loss_feature).mean()
            loss = loss_diffusion + loss_feature + loss_style
            loss.backward()
        else:
            t = torch.randint(0, opts.noise_step, (batch_size,)).long().cuda()
            t2, (x2, loss_diffusion2) = diffusion_target.few_shot_forward(image, t=t,
                                                                         x_self_cond=style_loss.get_gram_matrix(
                                                                             image).detach())
            dishu = 20
            alpha = dishu ** (t / 1000)
            loss_diffusion2 = ((dishu ** 0.9 - alpha) * loss_diffusion2).mean()/3
            loss=loss_diffusion2
            loss.backward()
        if global_step%10==0 and global_step!=0:
            print(global_step,float(loss_diffusion2),float(loss_diffusion),float(loss_feature),float(loss_style))

        if global_step%50==0 and global_step!=0:
            #if opts.clip_mode==0:
            # sampled_images = diffusion_target.sample(batch_size = 24)
            sampled_images = diffusion_target.ddim_sample((batch_size, 3, 128, 128), sample_step=50,condition=style_loss.get_gram_matrix(image).detach())
            save_image(sampled_images,os.path.join(save_dir, 'images/%d-sample-random-sample.jpg' % global_step), nrow=4, normalize=False)
            save_image(torch.cat((real_image,(x+1)/2,x_start_target),dim=0),os.path.join(save_dir,'images/%d.jpg'%global_step),nrow=batch_size,normalize=False)
            noise_step=800
            t = torch.ones(len(real_image)).long().to('cuda') * noise_step
            noises = diffusion_target.p_losses(real_image, t, return_x=True)
            sampled_images,sampled_middle_images = diffusion_target.ddim_sample(x.shape, sample_step=50,return_middle=True,start_img=noises, max_step=noise_step,
                                                                        min_step=-1,condition=style_loss.get_gram_matrix(image).detach())
            save_image(torch.cat((real_image,noises,sampled_middle_images,sampled_images),dim=0),
                       os.path.join(save_dir, 'images/%d-sample.jpg' % global_step), nrow=16,normalize=False)
            # img=diffusion_target.p_sample_loop(x.shape,img=noises,start_t=noise_step)
            # save_image(img,'results/0.jpg',nrow=4,normalize=True)
            if opts.clip_mode==1:
                save_image(sampled_images, os.path.join(save_dir, 'images/%d-sample-sample.jpg' % global_step),
                           nrow=4, normalize=False)
        if global_step%2==1:
            optimizer.step()
            optimizer.zero_grad()
        if global_step % 100 == 0 and global_step!=0:
            torch.save(diffusion_target.state_dict(), os.path.join(save_dir,'models/%d.pth'%global_step))
        global_step += 1
#CUDA_VISIBLE_DEVICES=1 python3 train-clip-condition.py --beta_f=1 --beta_style=1 --noise_step=300 --style=sketches
