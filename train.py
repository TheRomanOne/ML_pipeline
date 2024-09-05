from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from loss_functions import vae_loss
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(dataloader, vae, test_image, params):
  # train Variational Auto Encode
  if params['train_as'] == 'vae':
    result = train_vae(
      dataloader=dataloader,
      model=vae,
      n_epochs=params['n_epochs'],
      test_image=test_image,
      lr=params['learning_rate']
    )
  return result

def train_vae(dataloader, model, n_epochs, test_image=None, lr=1e-3) -> tuple:
  print("\n\nTraining type: Variational Auto-Encoder")
  opt = Adam(model.parameters(), lr=lr)
  # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.01)

  losses = []
  frames = []
  pbar = tqdm(range(n_epochs))
  for epoch in pbar:
    for i, (X, y) in enumerate(dataloader):
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
