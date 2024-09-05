import torch, cv2
from torch.utils.data import DataLoader
from IPython.display import HTML
from matplotlib import animation
import matplotlib.pyplot as plt
from VideoDataset import VideoDataset
from ImageDataset import ImageDataset
import numpy as np
from utils import rotate_2d_tensor, get_random_direction, evaluate_model_batches
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sharpen_image(img):
  return img
    # return cv2.filter2D(im, -1,
    #   np.array([[0, -1, 0],
    #             [-1, 5, -1],
    #             [0, -1, 0]])
    # )

def render_image(img, save_path, title='', to_horizontal=False):
  # img = ((img / 2 + .5))
  size = img.shape[1:]
  img = np.transpose(img,  (1, 2, 0))
  if to_horizontal:
    img = torch.rot90(img.unsqueeze(0), k=3, dims=[1, 2]).squeeze()

  plt.figure(figsize=(3, 3), tight_layout=True)
  plt.imshow(img)
  plt.title(title, fontsize=18, weight='bold')
  plt.axis('off')
  # plt.show()
  plt.savefig(f'{save_path}/{title}.png', bbox_inches='tight', pad_inches=0)
  plt.close()

def random_walk_image(img, net, angle, steps, change_prob, to_horizontal, save_path):
  _, mu, lv = net(img.to(device).unsqueeze(0))
  mu = mu.cpu()
  dir = get_random_direction(lv).cpu().detach().squeeze()
  new_frames = []
  print('Direction', dir)
  row_indices = np.array([0, 1])
  for i in range(steps):
    if np.random.rand() > change_prob:
      # row_indices = np.random.choice(dir.size(0), 2, replace=False)
      row_indices[[1, 0]] = row_indices
      row_indices[1] = torch.randint(0, dir.size(0), (1,)).item()
    n_dir = rotate_2d_tensor(dir, row_indices, torch.deg2rad(torch.tensor(angle)))
    z = mu + n_dir*5
    pic = net.from_latent(z.to(device))
    pic = pic.squeeze().cpu().detach().numpy()
    new_frames.append(pic)
  return create_video('random_walk', new_frames, save_path=save_path, to_horizontal=to_horizontal)

def plot_interpolation(net, latents_np, save_path):
  idx1, idx2 = np.random.choice(len(latents_np), 2, replace=False)
  z1, z2 = latents_np[idx1], latents_np[idx2]

  steps = 10
  inter = []
  for i in range(steps):
      alpha = i / (steps - 1)
      z = (1 - alpha) * z1 + alpha * z2
      
      with torch.no_grad():
        net_img = net.from_latent(torch.tensor(z).to(device).unsqueeze(0))

      net_img = net_img.cpu()
      net_img = net_img.detach().squeeze().numpy()
      inter.append(net_img)
  render_images(inter, 'interpolation', save_path)

def render_images(images, title, save_path):
  steps = len(images)
  plt.figure()
  for i, img in enumerate(images):
    plt.subplot(1, steps, i + 1)
    plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
    plt.axis('off')
  plt.savefig(f'{save_path}/{title}.png', bbox_inches='tight', pad_inches=0)
  # plt.show()
  plt.close()






def resize_image(image, size):
    # original_size = image.shape
    # scale_factor = (size[0] / original_size[1], size[1] / original_size[0])
    # # Resize the image using nearest-neighbor interpolation
    # image = image.resize(size, resample=Image.NEAREST)
    # return transforms.ToTensor()(image)


    resize = Resize(size, interpolation=Image.NEAREST)
    resize = Resize(size)
    image = transforms.ToPILImage()(image)  # Convert tensor to PIL Image
    image = resize(image)  # Resize image
    return transforms.ToTensor()(image)  # Convert back to tensor

def sample_images(n, net, images, labels, title, save_path, to_horizontal=False):
    print(f'Sampling {n} images')
    x_recon = evaluate_model_batches(net, images, batches=16)
    
    indices = np.random.choice(len(images), size=n, replace=False)
    chosen_images = images[indices]
    chosen_labels = labels[indices]
    chosen_recons = torch.tensor(x_recon[indices])
    
    target_size = (chosen_labels.shape[2], chosen_labels.shape[3])  # (height, width)
    
    resized_chosen_images = torch.stack([resize_image(img, target_size) for img in chosen_images])
    resized_chosen_recons = torch.stack([resize_image(img, target_size) for img in chosen_recons])
    
    images_t = torch.cat([resized_chosen_images, chosen_labels, resized_chosen_recons], dim=0)
    images_t = torch.clamp(images_t, 0, 1)
    
    grid = make_grid(images_t, nrow=n, pad_value=1)  # Adjust pad_value if needed
    
    plt.figure(figsize=(15, 15))  # Adjust figure size as needed
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.savefig(f'{save_path}/samples.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def interpolate_images(net, images, steps, save_path, to_horizontal=False, sharpen=False):
  y_prev, mu_prev, log_var_prev = net(images[0].to(device).unsqueeze(0))
  prev_z = net.reparameterize(mu_prev, log_var_prev).cpu().detach()

  inter = []
  images = torch.vstack([images, images[0].unsqueeze(0)])
  for img in images[1:]:
    y_2, mu_2, log_var_2 = net(img.to(device).unsqueeze(0))
    img_z = net.reparameterize(mu_2, log_var_2).cpu().detach()

    for i in range(steps):
      alpha = i / (steps - 1)
      z = img_z * alpha + prev_z * (1 - alpha)
      z = torch.tensor(z).to(device)
      net_img = net.from_latent(z, enhance=1).cpu().detach().squeeze().numpy()
      if sharpen:
        net_img = sharpen_image(net_img)
      inter.append(net_img)
    prev_z = img_z
  return create_video('interpolation', inter, save_path, to_horizontal=to_horizontal)

def create_video(name, frames, save_path, transform=True, to_horizontal=False, limit_frames=-1):
  frames = np.array(frames)
  if limit_frames > 0:
    while frames.shape[0] > 500:
      frames = frames[::2]
      
  if transform:
    frames = np.transpose(frames, (0, 2, 3, 1))
  if to_horizontal:
    frames = np.rot90(frames, k=3, axes=(1, 2))
  # if sharpen:
  #   frames = sharpen_image(frames)
  # frames = ((frames / 2 + .5))
  print('Video shape:', np.array(frames).shape)
  fig = plt.figure()
  plt.axis('off')
  im = plt.imshow(frames[0])
  plt.close() # this is required to not display the generated image
  def init():
      im.set_data(frames[0])

  def animate(i):
      im.set_data(frames[i])
      return im

  anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames.shape[0], interval=100)
  # return None#HTML(anim.to_html5_video())
  anim.save(f'{save_path}/{name}.mp4', writer='ffmpeg')

def crop_to_shape(image, target_size):
    """
    Crop an image to a specific shape.

    Args:
        image (PIL.Image): Input image.
        target_size (tuple): Target size (width, height) for cropping.

    Returns:
        PIL.Image: Cropped image.
    """
    img_width, img_height = image.size
    target_width, target_height = target_size

    # Calculate the cropping box
    left = (img_width - target_width) / 2
    top = (img_height - target_height) / 2
    right = (img_width + target_width) / 2
    bottom = (img_height + target_height) / 2

    # Crop and return the image
    return image.crop((left, top, right, bottom))

def load_video_ds(video_path, id_from, id_to, max_size, ratio=4, zoom_pixels=0, to_horizontal=False):
  video_dataset = VideoDataset(video_path, id_from, id_to, max_size, ratio, zoom_pixels, to_horizontal)
  print('Video dataset length', video_dataset.X_gt.shape)
  video_loader = DataLoader(video_dataset, batch_size=16, shuffle=True)

  return video_dataset, video_loader

def load_image_ds(video_path, max_size, ratio=4, to_horizontal=False):
  image_dataset = ImageDataset(video_path, max_size, ratio, to_horizontal)
  print('Image dataset length', len(image_dataset))
  image_loader = DataLoader(image_dataset, batch_size=16, shuffle=True)

  return image_dataset, image_loader

def plot_losses(losses, save_path):
  plt.figure()
  plt.plot(losses, marker='o', linestyle='-', color='b', label='Mean Loss')
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Mean Loss')
  plt.savefig(f'{save_path}/loss_plot.png', bbox_inches='tight', dpi=300)
  plt.close()