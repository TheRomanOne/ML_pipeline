import os, shutil, torch
import yaml, argparse
from models.VAE import VAE
from models.VAE_SR import VAE_SR
from models.VAE_SR_landscape import VAE_SR_landscape


with open('dataset_config.yaml', 'r') as yaml_file:
    dataset_config = yaml.safe_load(yaml_file)

def start_session(config_path):
    print('Loading config:', config_path)

    with open(config_path, 'r') as yaml_file:
        run_config = yaml.safe_load(yaml_file)
    session_name= run_config['session_name']
    dataset= run_config['dataset']
    scene_name = dataset['name']
    scene_type = dataset['type']
    print(f'Starting session for {scene_type}:', session_name)
    if not os.path.isdir('../sessions'):
        os.mkdir('../sessions')

    session_path = f'../sessions/{session_name}'


    if os.path.isdir(session_path):
        shutil.rmtree(session_path)

    os.mkdir(session_path)
    os.mkdir(f'{session_path}/videos')
    os.mkdir(f'{session_path}/images')

    w_path = '../weights'
    if not os.path.isdir(w_path):
        os.mkdir(w_path)
    
    weights_path = f"{w_path}/{session_name}.pth"
    
    run_config['nn_params']['learning_rate'] = float(run_config['nn_params']['learning_rate'])
    run_config.update({
        'path': session_path,
        'weights_path': weights_path,
        'dataset_path': dataset_config[scene_type][scene_name]
    })

    return run_config
    
def load_weights(model, weights_path):
    if os.path.isfile(weights_path):
        print('Loading checkout')
        model.load_state_dict(torch.load(weights_path))
    else:
        print('No model checkout found')

def parse_args(default):
    parser = argparse.ArgumentParser(description="Inset name and type")
    parser.add_argument('--config', default=default, type=str, help='Type argument')
    return parser.parse_args()

def load_model_from_params(session):
    params = session['nn_params']
    model_type = params['model_type'].lower()

    if model_type == 'vae':
        model = VAE(params, session['input_shape'])

    elif model_type == 'vae_sr':
        latent_dim = params['latent_dim']
        model = VAE_SR(latent_dim)

    elif model_type == 'vae_sr_landscape':
        latent_dim = params['latent_dim']
        model = VAE_SR_landscape(latent_dim)

    else:
        print(f"Model {model_type} is not supported yet")
        exit()
    

    if params['load_weights']:
        print('Loading weight for:', model_type)
        load_weights(model, session['weights_path'])
    else:
        print('Creating a new model:', model_type)
    
    return model
