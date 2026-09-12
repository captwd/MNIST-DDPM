import os

import torch
import yaml
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from models.unet_base import Unet
from compare.data import torch_load


def load_ddpm(config_path, ckpt_path, device, num_timesteps_override=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    diffusion_config = dict(config['diffusion_params'])
    model_config = config['model_params']
    if num_timesteps_override is not None:
        diffusion_config['num_timesteps'] = num_timesteps_override
    model = Unet(model_config).to(device)
    model.load_state_dict(torch_load(ckpt_path, map_location=device))
    model.eval()
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'])
    return model, scheduler, diffusion_config, model_config


@torch.no_grad()
def generate_samples(model, scheduler, diffusion_config, model_config,
                     num_samples, device, gen_batch=250, seed=42):
    torch.manual_seed(seed)
    im_c = model_config['im_channels']
    im_s = model_config['im_size']
    num_timesteps = diffusion_config['num_timesteps']
    images = []
    outer = tqdm(range(0, num_samples, gen_batch), desc='DDPM sampling', unit='batch')
    for start_idx in outer:
        n = min(gen_batch, num_samples - start_idx)
        xt = torch.randn((n, im_c, im_s, im_s), device=device)
        for i in reversed(range(num_timesteps)):
            t_batch = torch.full((n,), i, device=device, dtype=torch.long)
            noise_pred = model(xt, t_batch)
            xt, _ = scheduler.sample_prev_timestep(xt, noise_pred,
                                                   torch.as_tensor(i, device=device))
        images.append(torch.clamp(xt, -1., 1.).cpu())
        outer.set_postfix({'done': start_idx + n})
    return torch.cat(images, dim=0)


def save_preview_grid(images, out_path, nrow=8):
    grid = make_grid((images[:nrow * nrow] + 1) / 2, nrow=nrow)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)


@torch.no_grad()
def pseudo_label(images, classifier, device, batch_size=512, threshold=0.9):
    classifier.eval()
    labels, confs = [], []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].to(device)
        prob = torch.softmax(classifier(x), dim=1)
        conf, lab = prob.max(dim=1)
        labels.append(lab.cpu())
        confs.append(conf.cpu())
    labels = torch.cat(labels)
    confs = torch.cat(confs)
    keep = confs >= threshold
    return labels, confs, keep
