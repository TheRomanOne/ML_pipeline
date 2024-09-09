import os, shutil, torch
import yaml, argparse
from custom_models.VAE import VAE
from custom_models.VAE_SR import VAE_SR
from custom_models.VAE_SR_landscape import VAE_SR_landscape
from custom_models.LSTM import LSTMText, LSTMTimeSeq
from custom_models.UNet import UNet
import time
import global_settings as gs

device = gs.device

with open('dataset_config.yaml', 'r') as yaml_file:
    dataset_config = yaml.safe_load(yaml_file)

def start_session(config_path):
    print('Loading config:', config_path)

    with open(config_path, 'r') as yaml_file:
        session = yaml.safe_load(yaml_file)
    if len(session['session_name']) == 0:
        session['session_name'] = 'session_' + str(time.time()).split('.')[0]
    session_name = session['session_name']

    scene_name = session['dataset']['name']
    scene_type = session['dataset']['type']

    is_vision = scene_type == 'image' or scene_type == 'video'
    is_sequential = scene_type == 'text' or scene_type == 'timeseries'

    print(f'Starting session for {scene_type}:', session_name)
    if not os.path.isdir('../sessions'):
        os.mkdir('../sessions')

    session_path = f'../sessions/{session_name}'


    if os.path.isdir(session_path):
        shutil.rmtree(session_path)

    os.mkdir(session_path)
    os.mkdir(f'{session_path}/images')
    if is_vision:
        os.mkdir(f'{session_path}/videos')
    elif is_sequential:
        os.mkdir(f'{session_path}/{scene_type}')


    w_path = '../weights'
    if not os.path.isdir(w_path):
        os.mkdir(w_path)
    
    weights_path = f"{w_path}/{session_name}.pth"
    
    # evaluate expressions if exist in config (such as: param = '.5 * 1e-3')
    for k in session['nn']['params'].keys():
        if isinstance(session['nn']['params'][k], str):
            session['nn']['params'][k] = eval(session['nn']['params'][k])
    
    if isinstance(session['training']['learning_rate'], str):
        session['training']['learning_rate'] = eval(session['training']['learning_rate'])

    if is_vision:
        session['dataset']['max_size'] = session['nn']['params']['max_size']
        
    session.update({ 'path': session_path })
    session['nn'].update({ 'weights_path': weights_path })
    session['dataset'].update({ 'dataset_path': dataset_config[scene_type][scene_name] })

    return session, is_vision, is_sequential
    
def load_weights(model, weights_path):
    if os.path.isfile(weights_path):
        print('Loading checkout')
        model.load_state_dict(torch.load(weights_path))
    else:
        print('No model checkout found')

def parse_args():
    parser = argparse.ArgumentParser(description="Inset name and type")
    parser.add_argument('--config', type=str, help='Type argument')
    parser.add_argument('--name', type=str, help='Type argument')

    parsed = parser.parse_args()

    if parsed.name and parsed.config:
        print("Both namd and config path were given. choose one")
        exit()
    
    return parsed


def load_model_from_params(nn_config):
    params = nn_config['params']
    use_model = nn_config['use_model'].lower()

    if use_model == 'vae':
        model = VAE(
            params=params, 
            input_shape=params['input_shape']
        )

    elif use_model == 'vae_sr':
        model = VAE_SR(
            params=params, 
            input_shape=params['input_shape']
        )

    elif use_model == 'vae_sr_landscape':
        model = VAE_SR_landscape(
            latent_dim=params['latent_dim']
        )

    elif use_model == 'lstm_text':
        model = LSTMText(
            vocab_size=params['input_dim'], 
            embedding_dim=params['embedding_dim'], 
            hidden_dim=params['hidden_dim'], 
            num_layers=params['num_layers']
        )

    elif use_model == 'lstm_time_seq':
        model = LSTMTimeSeq(
            input_dim=params['input_dim'], 
            hidden_dim=params['hidden_dim'], 
            n_stacked_layers=params['n_layers']
        )
    elif use_model == 'unet':
        model = UNet(
            base_filters=params['base_filters']
        )
    else:
        print(f"Model {use_model} is not supported yet")
        exit()
    

    if nn_config['load_weights']:
        print('Loading weight for:', use_model)
        try:
            load_weights(model, nn_config['weights_path'])
        except Exception:
            print("Failed to load weights. Probably due to a change in nn in the config")
            exit()
    else:
        print('Creating a new model:', use_model)
    
    model.to(device)
    return model

