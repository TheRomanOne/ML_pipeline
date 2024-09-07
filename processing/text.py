import torch
import utils.vision_utils as i_utils
from train import train_model
from analysis import run_full_analysis, eval_and_interp
from utils.utils import evaluate_latent_batches, count_parameters
from utils.text_utils import parse_text, generate_text, get_sequential_data
from utils.session_utils import load_model_from_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_text_session(session, dataset, dataloader):
    
    word_to_index = dataset.word_to_index
    test_sequence_text = "citizen:"
    test_sequence = parse_text(test_sequence_text)[0]
    test_sequence = [word_to_index[word] for word in test_sequence]
    test_sequence = torch.tensor(test_sequence, dtype=torch.long).unsqueeze(0).to(device)
    params = session['nn_params']
    params['vocab_size'] = len(word_to_index)
    params['output_dim'] = len(word_to_index)



    session_path = session['path']

    seq_model = load_model_from_params(session)
    count_parameters(seq_model)

    losses, sequences = train_model(
      dataloader=dataloader,
      model=seq_model,
      test_sample=test_sequence,
      params=params
    )
    i_utils.plot_losses(losses, params['n_epochs'], save_path=f'{session_path}/images')

    print("Saving weights")
    torch.save(seq_model.state_dict(), session['weights_path'])

    learning_text = []
    for s in sequences:
      _, predicted = torch.max(torch.tensor(s).unsqueeze(0), dim=1)
      next_word = dataset.index_to_word[predicted.item()]
      learning_text.append(next_word)
    
    print('Creating learning sequence')
    with open(f"{session_path}/text/learning_sequence.txt", "w") as file:
      file.write(' '.join(learning_text))

    generated_text = generate_text(seq_model, test_sequence_text, word_to_index, dataset.index_to_word, max_length=300)
    with open(f"{session_path}/text/generated_text.txt", "w") as file:
      file.write(generated_text)

    # i_utils.create_video('learning', frames, save_path=f'{session_path}/videos', to_horizontal=to_horizontal, limit_frames=500)

    # # --------------------------- Run analysis --------------------------- 

    if len(session['analysis']) == 0:
      print('No analysis was specified')
      exit()    
    
    
    
    
    
    
    
    
    
    






