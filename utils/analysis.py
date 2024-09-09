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
from custom_models.UNet import forward_diffusion

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


# Testing the model on noisy images
def test_diffusion(model, images, save_path):
  model.eval()
  with torch.no_grad():
    images = images.to(device)

    # Add noise
    t = torch.rand(images.size(0), 1, 1, 1).to(device) * np.deg2rad(180)
    noisy_images = forward_diffusion(images, t, model.num_base_filters)

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

