from train.sequential_prediction import train_sequence
from train.image_reconstruction import train_vae, train_diffusion

def train_model(dataloader, model, config, test_sample=None):

  if config['method'] == 'vae_reconstruction':
    result = train_vae(
      dataloader=dataloader,
      model=model,
      n_epochs=config['n_epochs'],
      betha=config['kl_betha'],
      lr=config['learning_rate'],
      test_image=test_sample,
    )
  
  elif config['method'] == 'diffusion':
    result = train_diffusion(
      dataloader=dataloader,
      model=model,
      n_epochs=config['n_epochs'],
      lr=config['learning_rate'],
      test_image=test_sample,
    )

    
  elif config['method'] == 'sequence_prediction':
    result = train_sequence(
      dataloader=dataloader,
      model=model,
      n_epochs=config['n_epochs'],
      lr=config['learning_rate'],
      test_sequence=test_sample,
    )
      
  return result

