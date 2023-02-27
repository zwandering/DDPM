import clip
import torch
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torch.optim import Adam
import utils
from torchvision.utils import save_image
from criteria import id_loss
class Clip:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        #self.model.eval()
        self.transfroms=transforms.Compose([
            transforms.Resize([224,224]),
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
model=Clip()
model_id = id_loss.IDLoss().cuda()
mse_loss=torch.nn.MSELoss()
name='cari'
dataset1=Data('/home/huteng/dataset/ffhq/images1024x1024')
dataset2=Data('style_img/%s'%name)
test_dataloader1 = DataLoader(dataset1,
                              batch_size=50,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True)
test_dataloader2 = DataLoader(dataset2,
                              batch_size=10,
                              shuffle=False,
                              num_workers=8,
                              drop_last=True)
features1=[]
features2=[]
with torch.no_grad():
    for index,batch in enumerate(test_dataloader1):
        print(index)
        if index==200:
            break
        batch=batch.cuda()
        feature=model.encode_img(batch)
        features1.append(feature.cpu())
    # for index,batch in enumerate(test_dataloader2):
    #     batch=batch.cuda()
    #     feature=model.encode_img(batch)
    #     features2.append(feature.cpu())
    features1=torch.cat(features1,dim=0)
    #features2=torch.cat(features2,dim=0)
    print(features1.shape)
    features1=features1.numpy()
    #features2=features2.numpy()
    np.save('draw/features-ffhq',features1)
    #np.save('draw/features-%s'%name,features2)

# x=torch.ones(10000,512)
# print(x.mean())
# features1=torch.from_numpy(np.load('draw/features1.npy')).float()
# # for i in range(features1.size(0)):
# #     for j in range(features1.size(1)):
# #         if torch.isnan(features1[i][j]):
# #             print(i,j,features1[i][j])
# print(features1.shape)
# print(features1.mean())
# features2=torch.from_numpy(np.load('draw/features2.npy')).cuda()
# features1_mean=features1.mean(0).unsqueeze(0)
# features2_mean=features2.mean(0).unsqueeze(0)
# print(features1_mean.mean())
# print(torch.abs(features1-features1_mean).mean())
# print(torch.abs(features2-features2_mean).mean())
# print(torch.abs(features1_mean-features2_mean).mean())
# print(features1.shape,features2.shape)
# features1=torch.from_numpy(np.load('draw/features2.npy')).cuda()[:1]
# img=torch.rand(1,3,224,224).cuda()
# resize_224=transforms.Resize((224,224))
# ori_img=resize_224(utils.read_img('style_img/sketches/004_1_1_sz1.jpg')).cuda()
# dir='style_img/sketches'
# file_names=os.listdir(dir)
# model=Clip()
# img=resize_224(utils.read_img('style_img/sketches/005_1_1_sz1.jpg')).cuda()
# img.requires_grad=True
# optimzer=Adam([img], lr = 1e-3, betas =(0.9, 0.99))
# features1=model.encode_img(ori_img).detach()
# for i in range(10000):
#     #img = utils.read_img(os.path.join(dir,file_names[i])).cuda()
#     optimzer.zero_grad()
#     feature=model.encode_img(img)
#     loss=torch.abs(feature-features1).mean()
#     loss.backward()
#     optimzer.step()
#     #img.data = torch.clamp(img.data, min=0, max=1)
#     if i%100==0:
#         print(i, loss)
#         #print(img.min(),img.max())
#         if i%500==0:
#             save_image(img,'results/%d.jpg'%i,normalize=True)
#         #utils.show_tensor_img((img))