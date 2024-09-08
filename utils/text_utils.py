import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_text(text):

    # text = text.lower().replace('\n\n', '\n')
    # for sign in ["\n", "'", '"', '!', '.', ',', 'ing ', 'es ', 'ed ', ':', '/', "\\", '?']:
    #     text = text.replace(sign, f' {sign} ').replace('  ', ' ')

    # raw_data = text.split(' ')

    raw_data = list(text.lower())
    dictionary = sorted(list(set(raw_data)))

    word_to_index = {word: idx for idx, word in enumerate(dictionary)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    
    return raw_data, word_to_index, index_to_word, dictionary


def generate_text(model, start_words, word_to_index, index_to_word, max_length=10):
    start_words = parse_text(start_words)[0]
    model.eval()
    start_seq = torch.tensor([word_to_index[word] for word in start_words], dtype=torch.long).unsqueeze(0).to(device)
    generated_words = list(start_words)
    
    for _ in tqdm(range(max_length)):
        with torch.no_grad():
            output = model(start_seq)
            _, predicted = torch.max(output, dim=1)
            next_word = index_to_word[predicted.item()]
            generated_words.append(next_word)
            start_seq = torch.cat([start_seq[:, 1:], predicted.unsqueeze(0)], dim=1)
    
    return ' '.join(generated_words)

