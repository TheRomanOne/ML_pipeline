from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from torch.nn import CrossEntropyLoss
from loss_functions import vae_loss
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(dataloader, model, test_sample, params):
  # train Variational Auto Encode
  if params['train_as'] == 'vae':
    result = train_vae(
      dataloader=dataloader,
      model=model,
      n_epochs=params['n_epochs'],
      betha=params['kl_betha'],
      test_image=test_sample,
      lr=params['learning_rate']
    )
  elif params['train_as'] == 'lstm':
    result = train_lstm(
      dataloader=dataloader,
      model=model,
      n_epochs=params['n_epochs'],
      test_sequence=test_sample,
      lr=params['learning_rate']
    )
  return result

def train_lstm(dataloader, model, n_epochs, test_sequence=None, lr=1e-3):

  print("\n\nTraining type: LSTM")
  opt = Adam(model.parameters(), lr=lr)
  # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.01)

  losses = []
  sequences = []
  pbar = tqdm(range(n_epochs))
  loss_f = CrossEntropyLoss()
  model.train()

  for _ in pbar:
    for X, y in dataloader:
      x = X.to(device)
      y = y.to(device)
      recon_batch = model(x)
      loss = loss_f(recon_batch, y)

      opt.zero_grad()
      loss.backward()
      # scheduler.step()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      opt.step()

      loss = loss.cpu()
      l = loss.item()
      losses.append(l)

      det = recon_batch.detach()
      if test_sequence is not None:
        with torch.no_grad():
            det = model(test_sequence)
        det = det.detach()
        det = det.cpu()
        det = det.squeeze().numpy()
        sequences.append(det)

      pbar.set_description(f"Loss: {l:.3f}")
    torch.cuda.empty_cache()
  
  model.eval()

  return np.array(losses), sequences

def train_vae(dataloader, model, n_epochs, betha, test_image=None, lr=1e-3) -> tuple:
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
