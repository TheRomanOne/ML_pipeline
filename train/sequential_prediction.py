from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from torch.nn import CrossEntropyLoss
import torch
from global_settings import device



def train_sequence(dataloader, model, n_epochs, lr, test_sequence=None):

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

