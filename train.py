from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from loss_functions import vae_loss
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(dl, model, n_epochs, device, test_image=None, lr=1e-3):
  print("\n\nTraining in progress")
  opt = Adam(model.parameters(), lr=lr)
  # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.01)

  losses = []
  frames = []
  pbar = tqdm(range(n_epochs))
  for epoch in pbar:
    for i, (X, y) in enumerate(dl):
      x = X.to(device)
      y = y.to(device)
      recon_batch, mu, log_var = model(x)
      loss = vae_loss(recon_batch, y, mu, log_var)

      opt.zero_grad()
      loss.backward()
      # scheduler.step()
      opt.step()

      # if device.type == "cuda":
      loss = loss.cpu()
      l = loss.item()
      losses.append(l)

      det = recon_batch.detach()
      if test_image is not None:
        with torch.no_grad():
            det, _, _ = model(test_image.unsqueeze(0))
        det = det.detach()
        # if device.type == "cuda":
        det = det.cpu()
        det = det.squeeze().numpy()
        frames.append(det)

      pbar.set_description(f"Loss: {l:.3f}")
    torch.cuda.empty_cache()

  return np.array(losses), frames
