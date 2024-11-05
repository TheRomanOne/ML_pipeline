import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import utils.vision_utils as i_utils
import numpy as np
import seaborn as sns
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.manifold import TSNE
from global_settings import device
from custom_models.Diffusion import forward_diffusion, get_index_from_list, get_gaussian_noise

def run_pca(latents_np, save_path):
  print('\t --- pca')
  pca = PCA(n_components=2)
  latent_2d = pca.fit_transform(latents_np)

  plt.figure(figsize=(8, 6))
  plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=range(latents_np.shape[0]), cmap='tab10', s=5)
  plt.colorbar()
  plt.title('2D PCA of VAE Latent Space')
  plt.xlabel('PC 1')
  plt.ylabel('PC 2')
  plt.savefig(f'{save_path}/PCA_2D.png', bbox_inches='tight', pad_inches=0)
  # plt.show()

  pca = PCA(n_components=3)
  latent_3d = pca.fit_transform(latents_np)
  fig = plt.figure(figsize=(12, 8))
  ax = fig.add_subplot(111, projection='3d')

  sc = ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2], c=range(latents_np.shape[0]), cmap='tab10', s=5)

  cbar = plt.colorbar(sc, ax=ax)
  cbar.set_label('Image number')

  ax.set_title('3D PCA of VAE Latent Space')
  ax.set_xlabel('PC 1')
  ax.set_ylabel('PC 2')
  ax.set_zlabel('PC 3')
  plt.savefig(f'{save_path}/PCA_3D.png', bbox_inches='tight', pad_inches=0)
  # plt.show()
  plt.close()


def run_t_sne(latents_np, save_path):
  print('\t --- t_sne')
  tsne = TSNE(n_components=2, random_state=42)
  latent_2d_tsne = tsne.fit_transform(latents_np)

  plt.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], c=range(latents_np.shape[0]), cmap='tab10', s=5)
  plt.colorbar()
  plt.title('t-SNE of VAE Latent Space')
  plt.xlabel('t-SNE 1')
  plt.ylabel('t-SNE 2')
  plt.savefig(f'{save_path}/t_SNE.png', bbox_inches='tight', pad_inches=0)
  # plt.show()
  plt.close()

def run_umap(latents_np, save_path):
  print('\t --- umap')
  lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
  latent_2d_lle = lle.fit_transform(latents_np)

  plt.scatter(latent_2d_lle[:, 0], latent_2d_lle[:, 1], c=range(latents_np.shape[0]), cmap='tab10', s=5)
  plt.colorbar()
  plt.title('UMAP of VAE Latent Space')
  plt.xlabel('UMAP 1')
  plt.ylabel('UMAP 2')
  plt.savefig(f'{save_path}/umap.png', bbox_inches='tight', pad_inches=0)
  # plt.show()
  plt.close()

def run_isomap(latents_np, save_path):
  print('\t --- isomap')
  isomap = Isomap(n_components=2, n_neighbors=10)
  latent_2d_isomap = isomap.fit_transform(latents_np)

  plt.scatter(latent_2d_isomap[:, 0], latent_2d_isomap[:, 1], c=range(latents_np.shape[0]), cmap='tab10', s=5)
  plt.colorbar()
  plt.title('Isomap of VAE Latent Space')
  plt.xlabel('Isomap 1')
  plt.ylabel('Isomap 2')
  plt.savefig(f'{save_path}/isomap.png', bbox_inches='tight', pad_inches=0)
  # plt.show()
  plt.close()

def run_scatter(latents_np, save_path):
  print('\t --- scatter')
  df = pd.DataFrame(latents_np)
  scatter_matrix(df, c=range(latents_np.shape[0]), cmap='tab10', alpha=0.2)
  plt.savefig(f'{save_path}/scatter.png', bbox_inches='tight', pad_inches=0)
  # plt.show()
  plt.close()

def run_heatmap(latents_np, save_path):
  print('\t --- heatmap')
  df = pd.DataFrame(latents_np)
  sns.heatmap(df.corr(), cmap='coolwarm')
  plt.title('Heatmap of Latent Space')
  plt.savefig(f'{save_path}/heatmap.png', bbox_inches='tight', pad_inches=0)
  # plt.show()
  plt.close()


def run_full_analysis(latents_np, save_path):
  print('Running full latent analysis:')
  run_pca(latents_np, save_path)
  run_t_sne(latents_np, save_path)
  run_umap(latents_np, save_path)
  run_isomap(latents_np, save_path)
  run_scatter(latents_np, save_path)
  run_heatmap(latents_np, save_path)

def eval_and_interp(model, X_gt, y_gt, to_horizontal, session_path):
  print('evaluating 5 images')
  indices = i_utils.evaluate_images(
      n=5,
      model=model,
      images=X_gt,
      labels=y_gt,
      title='samples',
      to_horizontal=to_horizontal,
      save_path=f'{session_path}/images'
  )


  print('interpolating 5 images')
  s = X_gt.shape[0]
  # sampled_indices = np.random.choice(s, 20, replace=False)
  sampled_indices = indices
  sampled_indices = np.sort(sampled_indices)
  _imgs = X_gt[-s:][sampled_indices]
  i_utils.interpolate_images(model, _imgs, steps=5, save_path=f'{session_path}/videos', to_horizontal=to_horizontal)


  index = torch.randint(0, X_gt.size(0), (1,)).item()
  i_utils.random_walk_image(
      img=X_gt[index],
      model=model,
      angle=35,
      steps=200,
      change_prob=.9,
      to_horizontal=to_horizontal,
      save_path=f'{session_path}/videos'
  )


@torch.no_grad()
def sample_timestep(x, model, betas, sqrt_one_minus_alphas_cumprod, posterior_variance, sqrt_recip_alphas):
    
    t = torch.full((1,), 10, device=device, dtype=torch.long)
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x.squeeze(), t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
# def sample_plot_image(img_size, t, model, betas, sqrt_one_minus_alphas_cumprod, posterior_variance, sqrt_recip_alphas):
def sample_plot_image(img_size, save_path):
    # Sample noise
    img = torch.randn((1, img_size[0], *img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(30/num_images)

    for i in range(0,30)[::-1]:
        
        # betas, sqrt_recip_alphas, _, sqrt_one_minus_alphas_cumprod, posterior_variance = get_gaussian_noise()
        # img = sample_timestep(img, model, betas, sqrt_one_minus_alphas_cumprod, posterior_variance, sqrt_recip_alphas)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            i_utils.render_image(img.detach().cpu(), save_path)


def test_diffusion(model, images, save_path):
  
  model.eval()
  with torch.no_grad():
    images = images.to(device)

    # Add noise
    t = torch.rand(images.size(0), 1, 1, 1).to(device) * np.deg2rad(180)
    noisy_images, _, sqrt_recip_alphas, posterior_variance = forward_diffusion(images, t)

    denoised = []

    # denoise real image
    reconstructed_images = [noisy_images[0].unsqueeze(0)]
    while len(reconstructed_images) < 10:
      reconstructed_images.append(model(reconstructed_images[-1]))
    reconstructed_images = torch.tensor(np.array([i.squeeze().detach().cpu() for i in reconstructed_images]))
    denoised.append(reconstructed_images)
    
    # denoise nooise
    for i in range(2):
      # reconstructed_images = [torch.rand_like(images[0]).unsqueeze(0)]
      reconstructed_images = [torch.rand_like(noisy_images[0].unsqueeze(0))]
      while len(reconstructed_images) < 10:
        reconstructed_images.append(model(reconstructed_images[-1]))
      reconstructed_images = torch.tensor(np.array([i.squeeze().detach().cpu() for i in reconstructed_images]))
      denoised.append(reconstructed_images)

    
    i_utils.render_image_grid(denoised[0], denoised[1], denoised[2], save_path)

