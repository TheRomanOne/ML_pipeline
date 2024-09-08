import torch
import utils.vision_utils as i_utils
from utils.train import train_model
from utils.analysis import run_full_analysis, eval_and_interp
from utils.utils import evaluate_latent_batches, print_model_params
from utils.text_utils import parse_text, generate_text
from utils.utils import get_sequential_data
from utils.session_utils import load_model_from_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_text_intro(word_to_index):
  test_sequence_text = list(word_to_index.keys())[-10]
  # test_sequence = parse_text(test_sequence_text)[0]
  test_sequence = [word_to_index[test_sequence_text]]
  text_tensor = torch.tensor(test_sequence, dtype=torch.long).unsqueeze(0).to(device)
  return test_sequence_text, text_tensor

def get_numeric_intro(dataset):
  X = dataset.X_gt[-10]
  return X[:int(X.shape[0]/3)]

def process_sequential_session(session, dataset, dataloader):
    
    session_type = session['dataset']['type']

    nn_config = session['nn']
    nn_params = nn_config['params']
    training_config = session['training']

    # Create test sequence
    if session_type == 'text':
      test_sequence_text, test_sequence = get_text_intro(word_to_index=dataset.word_to_index)
    
    elif session_type == 'timeseries':
      test_sequence = get_numeric_intro(dataset)



# _____________________ Main Sequence _____________________

    session_path = session['path']

    seq_model = load_model_from_params(session['nn'])
    print_model_params(seq_model)

    losses, sequences = train_model(
      dataloader=dataloader,
      model=seq_model,
      test_sample=test_sequence,
      config=training_config
    )
    i_utils.plot_losses(losses, training_config['n_epochs'], save_path=f'{session_path}/images')

    # print("Saving weights")
    # torch.save(seq_model.state_dict(), session['weights_path'])
# _________________________________________________________


    # Sample model
    if session_type == 'text':
      learning_text = []
      for s in sequences:
        _, predicted = torch.max(torch.tensor(s).unsqueeze(0), dim=1)
        next_word = dataset.index_to_word[predicted.item()]
        learning_text.append(next_word)
      
      print('Creating learning sequence')
      with open(f"{session_path}/text/learning_sequence.txt", "w") as file:
        file.write(' '.join(learning_text))

      generated_text = generate_text(seq_model, test_sequence_text, dataset.word_to_index, dataset.index_to_word, max_length=300)
      with open(f"{session_path}/text/generated_text.txt", "w") as file:
        file.write(generated_text)

    elif session_type == 'timeseries':
      pass

    # # --------------------------- Run analysis --------------------------- 

    if len(session['analysis']) == 0:
      print('No analysis was specified')
      exit()    
    
    
    
    
    
    
    
    
    
    






