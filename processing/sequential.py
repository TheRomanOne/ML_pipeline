import torch
import utils.vision_utils as i_utils
from train.train import train_model
from utils.utils import print_model_params
from utils.text_utils import generate_text
from utils.session_utils import load_model_from_params
from datasets.Loader import load_dataset
import global_settings as gs

device = gs.device




def get_text_intro(word_to_index):
  test_sequence_text = list(word_to_index.keys())[-10]
  # test_sequence = parse_text(test_sequence_text)[0]
  test_sequence = [word_to_index[test_sequence_text]]
  text_tensor = torch.tensor(test_sequence, dtype=torch.long).unsqueeze(0)
  return test_sequence_text, text_tensor

def get_numeric_intro(dataset):
  X = dataset.X_gt[-10]
  return X[:int(X.shape[0]/3)]

def process_sequential_session(session):

    session_type = session['dataset']['type']
    training_config = session['training']
    session_path = session['path']
    nn_config = session['nn']

# ________________________________ Prepare dataset ________________________________ 

    dataset, dataloader = load_dataset(session)

    # # Create test sequence
    # if session_type == 'text':
    
    # elif session_type == 'timeseries':
    #   test_sequence = get_numeric_intro(dataset)



# ________________________________ Initialize model ________________________________ 

    seq_model = load_model_from_params(nn_config)
    print_model_params(seq_model)



# __________________________________ Train model __________________________________ 

    losses, _ = train_model(
      dataloader=dataloader,
      model=seq_model,
      config=training_config,
      # test_sample=test_sequence,
    )
    i_utils.plot_losses(losses, training_config['n_epochs'], save_path=f'{session_path}/images')

    print("Saving weights")
    torch.save(seq_model.state_dict(), nn_config['weights_path'])


# __________________________________ Post process __________________________________ 


    if len(session['post_process']) == 0:
      print('No post processing was requested')
      exit()

    # Sample model
    if 'generate_text' in session['post_process']:
      print("Generating text")
      # learning_text = []
      # for s in sequences:
      #   _, predicted = torch.max(torch.tensor(s).unsqueeze(0), dim=1)
      #   next_word = dataset.index_to_word[predicted.item()]
      #   learning_text.append(next_word)
      
      test_sequence_text, test_sequence = get_text_intro(word_to_index=dataset.word_to_index)
      test_sequence.to(device)

      generated_text = generate_text(seq_model, test_sequence_text, dataset.word_to_index, dataset.index_to_word, max_length=1000)
      with open(f"{session_path}/text/generated_text.txt", "w") as file:
        file.write(generated_text)

    elif session_type == 'timeseries':
      pass


        
    
    
    
    
    
    
    
    
    
    






