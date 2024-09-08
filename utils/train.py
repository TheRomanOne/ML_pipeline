from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from torch.nn import CrossEntropyLoss
from utils.loss_functions import vae_loss
import torch
import global_settings as gs

device = gs.device


def train_model(dataloader, model, test_sample, config):

  if config['method'] == 'image_reconstruction':
    result = train_image_reconstruction(
      dataloader=dataloader,
      model=model,
      n_epochs=config['n_epochs'],
      betha=config['kl_betha'],
      lr=config['learning_rate'],
      test_image=test_sample,
    )
  elif config['method'] == 'sequence_prediction':
    result = train_sequence_prediction(
      dataloader=dataloader,
      model=model,
      n_epochs=config['n_epochs'],
      lr=config['learning_rate'],
      test_sequence=test_sample,
    )
  return result

def train_sequence_prediction(dataloader, model, n_epochs, lr, test_sequence=None):

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
            det = model(test_sequence.to(device))
        det = det.detach()
        det = det.cpu()
        det = det.squeeze().numpy()
        sequences.append(det)

      pbar.set_description(f"Loss: {l:.3f}")
    torch.cuda.empty_cache()
  
  model.eval()

  return np.array(losses), sequences

def train_image_reconstruction(dataloader, model, n_epochs, betha, lr, test_image=None) -> tuple:
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
