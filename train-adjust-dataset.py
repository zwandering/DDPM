import torch
from denoising_diffusion_pytorch import Unet
from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
from style_loss import loss
from torch.optim import Adam
#DDPM扩充训练集训练训练 先去除风格损失较远的图像，然后去除与小样本内容损失小的图像
from argparse import ArgumentParser
import utils
parser = ArgumentParser()
parser.add_argument('--save_dir', type=str,default='results', help='Path to experiment output directory')
opts = parser.parse_args()
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
ignore_color=True
def adjust_images2_conbime(imgs,delete_num=10):
    #算content loss时与real_imgs与add_imgs一起算
    l = len(imgs)
    style_losses = torch.zeros((imgs.size(0), real_imgs.size(0))).cuda()
    for i in range(len(real_imgs)):
        tmp_img = real_imgs[i:i + 1]
        style_losses[:, i] = style_loss(imgs, tmp_img)
    style_losses = style_losses.mean(dim=1)
    _, style_idx = torch.sort(style_losses, descending=True)  # True为降序
    save_image(imgs[style_idx], '%s/images/%d-sorted_style_images.jpg' % (opts.save_dir, epoch), nrow=sample_size // 5,
               normalize=False)
    print(style_losses.min(),style_losses.mean(),style_losses.max())
    new_imgs = []
    for i in range(len(imgs)):
        if i not in style_idx[:delete_num // 2]:
            new_imgs.append(imgs[i])
    imgs = torch.stack(new_imgs, dim=0)
    combine_imgs=torch.cat([real_imgs,add_imgs],dim=0)
    content_losses = torch.zeros((imgs.size(0), combine_imgs.size(0))).cuda()
    for i in range(len(combine_imgs)):
        tmp_img = combine_imgs[i:i + 1]
        content_losses[:, i] = content_loss(imgs, tmp_img,ignore_color=ignore_color)
    content_losses = content_losses.min(dim=1).values
    _, content_idx = torch.sort(content_losses, descending=True)
    save_image(imgs[content_idx], '%s/images/%d-sorted_content_imgs.jpg' % (opts.save_dir, epoch),
               nrow=sample_size // 5,
               normalize=False)
    new_imgs = []
    for i in content_idx[:l - delete_num]:
        new_imgs.append(imgs[i])
    new_imgs = torch.stack(new_imgs, dim=0)
    return new_imgs
def adjust_images2(imgs,delete_num=10):
    l = len(imgs)
    style_losses = torch.zeros((imgs.size(0), real_imgs.size(0))).cuda()
    for i in range(len(real_imgs)):
        tmp_img = real_imgs[i:i + 1]
        style_losses[:, i] = style_loss(imgs, tmp_img)
    style_losses = style_losses.mean(dim=1)
    _, style_idx = torch.sort(style_losses, descending=True)  # True为降序
    #save_image(imgs[style_idx], '%s/images/%d-sorted_style_images.jpg' % (opts.save_dir, epoch), nrow=sample_size // 5,normalize=False)
    new_imgs = []
    for i in range(len(imgs)):
        if i not in style_idx[:delete_num // 2]:
            new_imgs.append(imgs[i])
    imgs = torch.stack(new_imgs, dim=0)
    content_losses = torch.zeros((imgs.size(0), real_imgs.size(0))).cuda()
    for i in range(len(real_imgs)):
        tmp_img = real_imgs[i:i + 1]
        content_losses[:, i] = content_loss(imgs, tmp_img,ignore_color=ignore_color)
    content_losses = content_losses.min(dim=1).values
    _, content_idx = torch.sort(content_losses, descending=True)
    #save_image(imgs[content_idx], '%s/images/%d-sorted_content_imgs.jpg' % (opts.save_dir, epoch),nrow=sample_size // 5,normalize=False)
    new_imgs = []
    for i in content_idx[:l - delete_num]:
        new_imgs.append(imgs[i])
    new_imgs = torch.stack(new_imgs, dim=0)
    return new_imgs
def adjust_images1(imgs,ori_imgs,delete_num=10):
    l=len(imgs)
    style_losses = torch.zeros((imgs.size(0), real_imgs.size(0))).cuda()
    for i in range(len(real_imgs)):
        tmp_img = real_imgs[i:i + 1]
        style_losses[:, i] = style_loss(imgs, tmp_img)
    style_losses = style_losses.mean(dim=1)
    _,style_idx = torch.sort(style_losses, descending=True) #True为降序
    save_image(imgs[style_idx],'%s/images/%d-sorted_style_images.jpg' % (opts.save_dir, epoch), nrow=sample_size // 5,normalize=False)
    new_imgs = []
    new_ori_imgs=[]
    for i in range(len(imgs)):
        if i not in style_idx[:delete_num//2]:
            new_imgs.append(imgs[i])
            new_ori_imgs.append(ori_imgs[i])
    ori_imgs=torch.stack(new_ori_imgs,dim=0)
    imgs = torch.stack(new_imgs,dim=0)
    content_losses=content_loss(ori_imgs,imgs)
    _,content_idx=torch.sort(content_losses, descending=False)
    save_image(ori_imgs[content_idx], '%s/images/%d-sorted_content_imgs-ori.jpg' % (opts.save_dir, epoch),nrow=sample_size // 5,normalize=False)
    save_image(imgs[content_idx], '%s/images/%d-sorted_content_imgs.jpg' % (opts.save_dir, epoch), nrow=sample_size // 5,
               normalize=False)
    new_imgs=[]
    new_ori_imgs = []
    for i in content_idx[:l-delete_num]:
        new_imgs.append(imgs[i])
        new_ori_imgs.append(ori_imgs[i])
    new_imgs=torch.stack(new_imgs,dim=0)
    new_ori_imgs=torch.stack(new_ori_imgs,dim=0)
    return new_imgs,new_ori_imgs


class Test_Data(Dataset):
    def __init__(self, img_path):
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([128, 128])
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
    def __init__(self, real_imgs,add_imgs,epoch=0):
        repeat_num=1
        imgs=torch.cat([real_imgs.repeat(repeat_num,1,1,1),add_imgs],dim=0)
        self.imgs = imgs.cpu().detach()
        save_image(self.imgs,os.path.join(opts.save_dir,'images/%d-dataset.jpg'%epoch),nrow=10)
        self.l = 1500
        self.l-=min(10,epoch)*100
    def __getitem__(self, idx):
        idx=idx%len(self.imgs)
        return self.imgs[idx]
    def __len__(self):
        return self.l
clip_model=utils.clip_content_loss()
dir='style_img/sketches'
file_names=os.listdir(dir)
real_imgs=[]
for file_name in file_names:
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        img = Image.open(os.path.join(dir, file_name))
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Resize((128,128))(img)
        real_imgs.append(img)
real_imgs=torch.stack(real_imgs,dim=0).cuda()
add_imgs=real_imgs.clone()
diffusion.load_state_dict(torch.load('pretrained-models/481157.pth'))
optizer = Adam(diffusion.parameters(), lr = 1e-4, betas =(0.9, 0.99))
global_step=0
style_loss = loss.VGGStyleLoss(transfer_mode=1, resize=True).cuda()
content_loss=loss.VGGPerceptualLoss()
sample_size=50
test_data=Test_Data('/home/huteng/dataset/celeba')
test_dataloader = DataLoader(test_data,
                              batch_size=sample_size,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True)
test_iter = iter(test_dataloader)
for epoch in range(100000):
    # repeat_num=add_imgs.size(0)//real_imgs.size(0)
    # if repeat_num==0:
    #     repeat_num=1
    train_data = Train_Data(real_imgs,add_imgs,epoch=epoch)
    train_dataloader = DataLoader(train_data,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)
    for batch_idx,batch in enumerate(train_dataloader):
        if batch_idx%10==0:
            print(batch_idx)
        image=batch.cuda()
        loss = diffusion(image)
        optizer.zero_grad()
        loss.backward()
        optizer.step()
        global_step+=1
    with torch.no_grad():
        sampled_images = diffusion.ddim_sample((sample_size, 3, 128, 128),sample_step=25)
        #save_image(sampled_images, '%s/images/%d-random-sample.jpg' % (opts.save_dir, epoch), nrow=sample_size//5,normalize=False)
        # test_imgs = next(test_iter, None).cuda()
        # step = 800
        # t = torch.ones(len(test_imgs)).long().cuda() * step
        # noises = diffusion.p_losses(test_imgs, t,return_x=True)
        # sampled_images = diffusion.ddim_sample((sample_size, 3, 128, 128), sample_step=10,start_step=step,start_img=noises)
        # print(float(real_imgs.max()),float(real_imgs.min()),float(real_imgs.mean()),'||',float(sampled_images.max()),float(sampled_images.min()),float(sampled_images.mean()))
        # torch.save(diffusion.state_dict(), '%s/models/%d.pth' % (opts.save_dir, epoch))
        # save_image(torch.cat((test_imgs[:25],noises[:25]),dim=0), '%s/images/%d-sample-ori.jpg' % (opts.save_dir, epoch), nrow=sample_size // 5,normalize=False)
        # save_image(sampled_images, '%s/images/%d-sample.jpg' % (opts.save_dir, epoch), nrow=sample_size//5,normalize=False)
        #sampled_images,sampled_ori_images=adjust_images1(sampled_images,test_imgs,delete_num=sample_size-4)
        sampled_images=adjust_images2_conbime(sampled_images,delete_num=sample_size-8)
        if epoch==0:
            add_imgs=sampled_images.detach()
            #ori_imgs=sampled_ori_images.detach()
        else:
            add_imgs=torch.cat((add_imgs,sampled_images.detach()),dim=0)
            #ori_imgs=torch.cat((ori_imgs,sampled_ori_images.detach()),dim=0)
            #imgs,ori_imgs=adjust_images1(imgs,ori_imgs,delete_num=2)
            add_imgs=adjust_images2(add_imgs,delete_num=4)
