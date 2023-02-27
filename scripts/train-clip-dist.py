import sys
sys.path.append('..')
import torch
# from denoising_diffusion_pytorch import Unet
from unet import MyUnet
from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
from torch.optim import Adam,SGD
import numpy as np
import random
import clip
from style_loss import loss
from argparse import ArgumentParser
from model.big_unet import create_model
from util import resize_right
import torchvision
import os
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
# print(local_rank)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")

#在train-clip基础上以风格图像为condition加入训练
parser = ArgumentParser()
# parser.add_argument('--output_dir', type=str,default='results', help='Path to experiment output directory')
parser.add_argument('--resume_iter', type=int,default=-1, help='Path to experiment output directory')
parser.add_argument('--beta_f', type=str,default='1', help='Path to experiment output directory')
parser.add_argument('--beta_style', type=str,default='1', help='Path to experiment output directory')
parser.add_argument('--beta_content', type=str,default='1', help='Path to experiment output directory')
parser.add_argument('--style', type=str,default='sketches', help='Path to experiment output directory')
parser.add_argument('--noise_step', type=int,default=300, help='Path to experiment output directory')
parser.add_argument('--clip_mode', type=int,default=1, help='0:clip direction loss;1:clip center loss')
opts = parser.parse_args()
image_size=256
model = create_model(
    image_size=image_size,
    use_fp16=False,
    condition=True
)
diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
diffusion.load_state_dict(torch.load('/home/huteng/DDPM2/results2/recon/clip_center_loss/step=300,beta_f=1,beta_s=1/models/60000.pth'),strict=True)
# print('-1',torch.cuda.memory_allocated()/1000000000)
# ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
# print('-0',torch.cuda.memory_allocated()/1000000000)
ddp_diffusion = DDP(diffusion, device_ids=[local_rank], output_device=local_rank)

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
batch_size =3
clip_model=Clip()
style_loss = loss.VGGStyleLoss(transfer_mode=1, resize=True).cuda()
vgg = torchvision.models.vgg16(pretrained=True).cuda()
name=opts.style
dir='../style_img/%s'%name
loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([image_size, image_size])
        ])
train_data=Train_Data('../style_img/%s'%name)
style_imgs=[]
for i in os.listdir(dir):
    image = Image.open(os.path.join(dir, i)).convert('RGB')
    style_imgs.append(loader(image))
style_imgs=torch.stack(style_imgs,dim=0).cuda()
style_features=clip_model.encode_img(style_imgs).detach()
real_data=Train_Data('/home/huteng/dataset/ffhq/images1024x1024')
real_image2=[]
real_image_dir2='/home/huteng/DDPM2/images/source'
for i in os.listdir(real_image_dir2):
    image = Image.open(os.path.join(real_image_dir2, i)).convert('RGB')
    real_image2.append(loader(image))
real_image2=torch.stack(real_image2,dim=0)[:8].cuda()
real_image2_features=vgg.features[:16](
    real_image2.mean(1).unsqueeze(1).repeat(1, 3, 1, 1)).detach()
condition2={}
condition2['content']=real_image2_features
condition2['style']=style_features
features_source=torch.from_numpy(np.load('../draw/features-ffhq.npy')).cuda().mean(0)
features_target=torch.from_numpy(np.load('../draw/features-%s.npy'%name)).cuda().mean(0)
feature_dir=(features_target-features_source).type(torch.HalfTensor).cuda().unsqueeze(0)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
real_sampler = torch.utils.data.distributed.DistributedSampler(real_data)

real_dataloader = DataLoader(real_data,
                                   batch_size=batch_size,
                                   # shuffle=True,
                                   num_workers=8,
                                   drop_last=True,
                             sampler=real_sampler,
                             )
real_dataloader_iter=iter(real_dataloader)
train_dataloader = DataLoader(train_data,
                                   batch_size=batch_size,
                                   # shuffle=True,
                                   num_workers=8,
                                   drop_last=True,
                              sampler=train_sampler,
                              )


save_dir='../results2/%s/clip_dir_loss'%name if opts.clip_mode==0 else '../results2/%s/clip_center_loss'%name
save_dir=os.path.join(save_dir,'step=%d,beta_f=%s,beta_s=%s'%(opts.noise_step,opts.beta_f,opts.beta_style))
# optimizer = Adam(ddp_diffusion.module.model.parameters(), lr = 1e-4, betas =(0.9, 0.99))
optimizer = Adam(ddp_diffusion.module.model.parameters(), lr = 1e-4,betas =(0.9, 0.99))
global_step=0 if opts.resume_iter==-1 else opts.resume_iter
mse_loss=torch.nn.MSELoss(reduction='none')
mse_loss_reduce=torch.nn.MSELoss()
cos_loss=torch.nn.CosineSimilarity(dim=1)
os.makedirs(os.path.join(save_dir,'models'),exist_ok=True)
os.makedirs(os.path.join(save_dir,'images'),exist_ok=True)
opts.beta_f=float(opts.beta_f)
opts.beta_style=float(opts.beta_style)
opts.beta_content=float(opts.beta_content)
loss_diffusion=0
loss_diffusion2=0
loss_feature=0
loss_content=0
loss_style=0
filter_N=4
condition={}
for epoch in range(100):
    for batch_idx,batch in enumerate(train_dataloader):
        if batch_idx%100==0:
            print(batch_idx)
        image=batch.cuda()
        if global_step%1==0:
            real_image = next(real_dataloader_iter, None)
            if real_image is None:
                real_dataloader_iter = iter(real_dataloader)
                real_image = next(real_dataloader_iter, None)
            real_image = real_image.cuda()
            real_image_vgg_features = vgg.features[:16](
                real_image.mean(1).unsqueeze(1).repeat(1, 3, 1, 1)).detach()  # 256,32,32
            condition["content"] = real_image_vgg_features
            condition["style"] = style_features
            with torch.no_grad():
                #x_start_source=diffusion.ddim_sample(x.shape, sample_step=25,start_img=x,start_step=t[0])
                t, x = ddp_diffusion.module.few_shot_forward(real_image,step=opts.noise_step,x_self_cond=condition,return_x=True)
                #x_start_source = (diffusion.batch_p_sample(x, t, None)+1)/2
                feature_source=clip_model.encode_img(real_image)
            x_start_target = (ddp_diffusion.module.batch_p_sample(x, t, x_self_cond=condition)+1)/2
            # #loss_feature = content_loss(x_start_target,x_start_source).mean()
            feature_target=clip_model.encode_img(x_start_target)
            if opts.beta_f != 0:
                if opts.clip_mode==0:
                    loss_feature = (1-cos_loss(feature_target-feature_source, feature_dir.repeat(batch_size,1))).mean()
                else:
                    feature_source_to_target = feature_source + feature_dir.repeat(batch_size, 1)
                    loss_feature = mse_loss(feature_target, feature_source_to_target).mean(-1)
                    #loss_feature = mse_loss(feature_target, features_target0.repeat(feature_target.size(0),1))
            if opts.beta_style!=0:
                # loss_style=torch.zeros(x_start_target.size(0)).cuda()
                # for i in range(x_start_target.size(0)):
                #     loss_style[i]=style_loss(x_start_target[i:i+1].repeat(style_imgs.size(0),1,1,1),style_imgs).mean()
                # loss_style=loss_style*opts.beta_style/x_start_target.size(0)
                loss_style=style_loss(x_start_target,image)*opts.beta_style
            if opts.beta_content!=0:
                x_start_target_features = vgg.features[:16](x_start_target.mean(1).unsqueeze(1).repeat(1, 3, 1, 1))
                loss_content = mse_loss(real_image_vgg_features, x_start_target_features).mean(-1).mean(-1).mean(-1) * opts.beta_content
            loss_feature*= opts.beta_f
            t2, (x2, loss_diffusion) = ddp_diffusion.module.few_shot_forward(image, t=t,x_self_cond=condition)
            dishu = 20
            alpha = dishu ** (t / 1000)
            loss_diffusion = ((dishu ** 0.9 - alpha) * loss_diffusion).mean() * 3
            loss_style = (alpha * loss_style).mean()
            loss_content= (alpha*loss_content).mean()
            loss_feature = (alpha * loss_feature).mean()
            loss = loss_diffusion + loss_feature + loss_style
        # else:
        #     t = torch.randint(0, opts.noise_step, (batch_size,)).long().cuda()
        #     t2, (x2, loss_diffusion2) = ddp_diffusion.module.few_shot_forward(image, t=t,
        #                                                                  x_self_cond=condition)
        #     dishu = 20
        #     alpha = dishu ** (t / 1000)
        #     loss_diffusion2 = ((dishu ** 0.9 - alpha) * loss_diffusion2).mean()/3
        #     loss=loss_diffusion2
        loss.backward()
        if global_step%10==0 and global_step!=0 and local_rank==0:
            print('step=%d,dif1=%.4f, dif2=%.4f, fea=%.4f, sty=%.4f, cont=%.4f' % (
            global_step, float(loss_diffusion2), float(loss_diffusion), float(loss_feature), float(loss_style),
            float(loss_content)))
            if global_step%100==0:
                save_image(torch.cat((real_image,(x+1)/2,x_start_target),dim=0),os.path.join(save_dir,'images/%d.jpg'%global_step),nrow=batch_size,normalize=False)
                noise_step=800
                t = torch.ones(len(real_image2)).long().to('cuda') * noise_step
                noises = ddp_diffusion.module.p_losses(real_image2, t, return_x=True)
                sampled_images,sampled_middle_images = ddp_diffusion.module.ddim_sample(x.shape, sample_step=50,return_middle=True,start_img=noises, max_step=noise_step,
                                                                            min_step=-1,condition=condition2)
                save_image(torch.cat((real_image2,noises,sampled_middle_images,sampled_images),dim=0),
                           os.path.join(save_dir, 'images/%d-sample.jpg' % global_step), nrow=sampled_images.size(0),normalize=False)
                if opts.clip_mode==1:
                    save_image(sampled_images, os.path.join(save_dir, 'images/%d-sample-sample.jpg' % global_step),
                               nrow=4, normalize=False)
            if global_step % 100 == 0:
                torch.save(ddp_diffusion.module.state_dict(), os.path.join(save_dir, 'models/%d.pth' % global_step))
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
        if global_step%2==1:
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1
#CUDA_VISIBLE_DEVICES=1 python3 train-clip.py --beta_f=1 --beta_style=1 --noise_step=300 --style=sketches
