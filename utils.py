from torchvision import  transforms
from PIL import Image
from matplotlib import pyplot as plt
import clip
import numpy as np
import os
import torch
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
class clip_content_loss():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        #print(self.preprocess)
        self.transfroms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        features_source = torch.from_numpy(np.load('draw/features1.npy')).cuda().mean(0).cpu()
        features_target = torch.from_numpy(np.load('draw/features2.npy')).cuda().mean(0).cpu()
        self.feature_dir = (features_target - features_source).cuda().unsqueeze(0)
        self.mse_loss=torch.nn.MSELoss()
    def encode_text(self,text_input):
        return self.model.encode_text(clip.tokenize(text_input).to(self.device))
    def encode_img(self,img):
        return self.model.encode_image(self.transfroms(img))
    def content_loss(self,source_img,target_img):
        feature_source=self.encode_img(source_img)
        feature_target=self.encode_img(target_img)
        feature_source_to_target=feature_source+self.feature_dir.repeat(feature_source.size(0),0)
        loss_mse=self.mse_loss(feature_target,feature_source_to_target)
        return loss_mse
def read_img(path,size=128):
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([size, size])
    ])
    img=Image.open(path).convert('RGB')
    return loader(img).unsqueeze(0)
def show_tensor_img(img):
    if img.ndim ==4:
        img=img[0]
    plt.imshow(img.cpu().detach().numpy().transpose((1, 2, 0)))
    plt.pause(0.1)