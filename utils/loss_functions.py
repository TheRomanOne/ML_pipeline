import torch
# from custom_models.Diffusion import forward_diffusion_sample
# from global_settings import device



def vae_loss(recon_x, x, mu, logvar, betha):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + betha * kl_loss
