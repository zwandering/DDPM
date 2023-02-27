import torch
from denoising_diffusion_pytorch import Unet

from model.ddpm import GaussianDiffusion
from model import ddpm
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import os
from tqdm.auto import tqdm
from functools import partial


def read_img(path):
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([128, 128])
    ])
    img=Image.open(path).convert('RGB')
    return loader(img).unsqueeze(0)
device='cuda'
# model0 = Unet(
#     dim = 64,
#     dim_mults = (1, 2, 4, 8)
# ).to(device)
# diffusion0 = GaussianDiffusion(
#     model0,
#     image_size = 128,
#     timesteps = 1000,   # number of steps
#     loss_type = 'l1'    # L1 or L2
# ).to(device)
class my_diffusion(GaussianDiffusion):
    def prepare_set(self):
        self.predictions=[]
        self.noises=[]
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, x_self_cond)
        self.predictions.append(model_output)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else ddpm.identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ddpm.ModelPrediction(pred_noise, x_start)

    def p_sample(self, x, t: int, x_self_cond=None, clip_denoised=True, loss_fn=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, model_variance, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times,
                                                                                       x_self_cond=x_self_cond,
                                                                                       clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        self.noises.append(noise)
        if loss_fn is not None and t < 200:
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                loss = loss_fn(x_in)
                grad = torch.autograd.grad(loss, x_in)[0]
                print(loss, grad.mean())
            model_mean = model_mean - grad * model_variance * 0.5
            pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        else:
            pred_img = model_mean
        return pred_img, x_start
    def model_predictions_merge(self, x, t, x_self_cond=None, clip_x_start=False):
        model_output0 = self.predictions[-t[0]-1]
        b = x.size(0)
        if t[0] > 0:
            model_output = torch.zeros_like(model_output0).to(x.device).float()
            for i in range(b):
                z = torch.rand(b)
                z = z / z.sum()
                for j in range(b):
                    model_output[i] += z[j] * model_output0[j]
        else:
            model_output = model_output0
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else ddpm.identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ddpm.ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def p_mean_variance_merge(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions_merge(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample_merge(self, x, t: int, x_self_cond=None, clip_denoised=True, loss_fn=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, model_variance, model_log_variance, x_start = self.p_mean_variance_merge(x=x, t=batched_times,
                                                                                       x_self_cond=x_self_cond,
                                                                                       clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        #noise=self.noises[-t-1]
        if loss_fn is not None and t < 200:
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                loss = loss_fn(x_in)
                grad = torch.autograd.grad(loss, x_in)[0]
                print(loss, grad.mean())
            model_mean = model_mean - grad * model_variance * 0.5
            pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        else:
            pred_img = model_mean
        return pred_img, x_start

    def p_sample_loop_merge(self, shape=None, loss_fn=None, img=None, start_t=None):
        # batch, device = shape[0], self.betas.device
        # img = torch.randn(shape, device=device)
        if start_t is None:
            start_t = self.num_timesteps
        if img is None:
            batch, device = shape[0], self.betas.device
            img = torch.randn(shape, device=device)
        x_start = None
        for t in tqdm(reversed(range(0, start_t)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample_merge(img, t, self_cond, loss_fn=loss_fn)
            if t % 100 == 0:
                save_image(x_start, 'results/%d-x0.jpg' % t, normalize=True)
                save_image(img, 'results/%d.jpg' % t, normalize=True)
        img = ddpm.unnormalize_to_zero_to_one(img)
        return img
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).to(device)
diffusion = my_diffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).to(device)
model0 = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).to(device)
diffusion0 = my_diffusion(
    model0,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).to(device)
dir='style_img/sketches'
file_names=os.listdir(dir)
imgs=[]
diffusion0.load_state_dict(torch.load('results/fine-tune-model/cari/models/600.pth'))
diffusion.load_state_dict(torch.load('results/cari/clip_center_loss/step=600,beta=0.1/models/600.pth'))
batch_size=16
with torch.no_grad():
    for file_name in file_names[:batch_size]:
        file_path=os.path.join(dir,file_name)
        imgs.append(read_img(file_path))
    imgs=torch.cat(imgs,dim=0).to(device)
    step=600
    t=torch.ones(len(imgs)).long().to(device)*step
    #noises=torch.randn_like(noises).to(noises.device)
    #img0=diffusion0.p_sample_loop(img=noises)
    noises = diffusion.p_losses(imgs, t,return_x=True)
    sampled_images1, sampled_middle_images1 = diffusion.ddim_sample(imgs.shape, sample_step=25, max_step=1000,
                                                                    min_step=600, return_middle=True)
    sampled_images, sampled_middle_images2 = diffusion0.ddim_sample(imgs.shape, sample_step=25,
                                                                    max_step=600,
                                                                    min_step=-1, start_img=sampled_images1,
                                                                    return_middle=True)
    save_image(torch.cat((sampled_middle_images1, sampled_middle_images2), dim=0),
               'test.png', nrow=16, normalize=False)