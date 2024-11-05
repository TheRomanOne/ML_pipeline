from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from utils.loss_functions import vae_loss
import torch
from global_settings import device
import torch.nn as nn
from custom_models.Diffusion import forward_diffusion
import torch.nn.functional as F

def train_diffusion(dataloader, model, n_epochs, lr, test_image=None) -> tuple:
  print("\n\nTraining type: Diffusion")
  opt = Adam(model.parameters(), lr=lr)
  # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.01)

  losses = []
  frames = []
  pbar = tqdm(range(n_epochs))
  
  model.train()
  loss_f = nn.L1Loss()
  
  
  for epoch in pbar:
    for X, y in dataloader:
      x = X.to(device)
      y = y.to(device)
      
      # Forward diffusion
      t = torch.randint(0, 10, (x.shape[0],), device=device).long()
      noisy_x, noise, _, _ = forward_diffusion(x, t, device)
      noise_pred = model(noisy_x, t)
      loss = loss_f(noise, noise_pred)


      opt.zero_grad()
      loss.backward()
      # scheduler.step()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      opt.step()

      loss = loss.cpu()
      l = loss.item()
      losses.append(l)

      # det = recon_batch.detach()
      if test_image is not None:
        with torch.no_grad():
          noisy_test, _, _, _ = forward_diffusion(x, t, device)
          test_noise_pred = model(noisy_x, t)
          det = (test_noise_pred - noisy_test)
        det = det.detach()
        det = det.cpu()
        det = det.squeeze().numpy()
        frames.append(det)

      pbar.set_description(f"Loss: {l:.3f}")
    torch.cuda.empty_cache()
  
  model.eval()
  frames = np.vstack(frames)
  return np.array(losses), frames


def train_vae(dataloader, model, n_epochs, betha, lr, test_image=None) -> tuple:
  print("\n\nTraining type: Variational Auto-Encoder")
  opt = Adam(model.parameters(), lr=lr)
  # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.01)

  losses = []
  frames = []
  pbar = tqdm(range(n_epochs))
  
  model.train()

  for epoch in pbar:
    for X, y in dataloader:
      x = X.to(device)
      y = y.to(device)
      recon_batch, mu, log_var = model(x)
      loss = vae_loss(recon_batch, y, mu, log_var, betha)

      opt.zero_grad()
      loss.backward()
      # scheduler.step()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      opt.step()

      loss = loss.cpu()
      l = loss.item()
      losses.append(l)

      det = recon_batch.detach()
      if test_image is not None:
        with torch.no_grad():
            det, _, _ = model(test_image.unsqueeze(0))
        det = det.detach()
        det = det.cpu()
        det = det.squeeze().numpy()
        frames.append(det)

      pbar.set_description(f"Loss: {l:.3f}")
    torch.cuda.empty_cache()
  
  model.eval()

  return np.array(losses), frames
