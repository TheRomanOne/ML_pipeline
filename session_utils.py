import os, shutil, torch
import yaml, argparse

with open('dataset_config.yaml', 'r') as yaml_file:
    dataset_config = yaml.safe_load(yaml_file)

def start_session(config_path):
    with open(config_path, 'r') as yaml_file:
        run_config = yaml.safe_load(yaml_file)
    scene_name = run_config['name']
    scene_type = run_config['type']
    print('Starting session', scene_type, scene_name)
    if not os.path.isdir('sessions'):
        os.mkdir('sessions')

    session_path = f'sessions/{scene_name}'


    if os.path.isdir(session_path):
        shutil.rmtree(session_path)

    os.mkdir(session_path)
    os.mkdir(f'{session_path}/videos')
    os.mkdir(f'{session_path}/images')

    w_path = 'weights'
    if not os.path.isdir(w_path):
        os.mkdir(w_path)
        os.mkdir(f'{w_path}/images')
        os.mkdir(f'{w_path}/videos')
    
    weights_path = f"{w_path}/{scene_type}/{scene_name}.pth"

    return {
        'name': scene_name,
        'type': scene_type,
        'path': session_path,
        'weights_path': weights_path,
        'video_path': dataset_config[scene_type][scene_name],
        'params': run_config['params']
    }

def load_weights(model, weights_path):
    if os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path))
    else:
        print('No model checkout found')

def parse_args(default):
    parser = argparse.ArgumentParser(description="Inset name and type")
    parser.add_argument('--config', default=default, type=str, help='Type argument')
    return parser.parse_args()
