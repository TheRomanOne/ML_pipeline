import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rotate_2d_tensor(tensor, row_indices, angle):

    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    rotation_matrix = torch.tensor([[cos_theta, -sin_theta],
                                    [sin_theta,  cos_theta]])


    rows_to_rotate = tensor[row_indices]
    rotated_rows = torch.matmul(rotation_matrix, rows_to_rotate.T).T
    tensor[row_indices] = rotated_rows

    return tensor

def get_random_direction(logvar):
  std = torch.exp(0.5 * logvar)
  eps = torch.randn_like(std)
  return std * eps

def evaluate_latent(model, images):
  with torch.no_grad():
    _, latents, _ = model(images.to(device))
    latents_np = latents.cpu().detach().numpy()
  return latents_np

def evaluate_latent_batches(model, images, batches=16):
  result = []
  while len(images) > 0:
    group = images[:batches].to(device)
    e = evaluate_latent(model, group)
    result.append(e)
    images = images[batches:]
  
  return np.vstack(result)

def evaluate_model(model, images):
  with torch.no_grad():
    x_recon, mu, log_var = model(images.to(device))
    x_recon = x_recon.detach().cpu()
  return x_recon

def evaluate_model_batches(model, images, batches=16):
  result = []
  while len(images) > 0:
    group = images[:batches].to(device)
    e = evaluate_model(model, group)
    result.append(e)
    images = images[batches:]
  
  return np.vstack(result)

def conv2d_output_shape(input_size, kernel_size, stride, padding):
    # Calculate output height and width
    return (np.array(input_size) - kernel_size + 2 * padding) // stride + 1

def deconv2d_output_shape(input_size, kernel_size, stride, padding, output_padding):
    # Calculate output height/width for ConvTranspose2d (Deconv)
    return (np.array(input_size) - 1) * stride - 2 * padding + kernel_size + output_padding

print('')