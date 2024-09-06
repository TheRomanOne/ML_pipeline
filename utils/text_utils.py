import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_text(text):

    text = text.lower().replace('\n\n', '\n')
    for sign in ["'s", '.', ',', 'ing ', 'es ', 'ed ']:
        text = text.replace(sign, f' {sign} ').replace('  ', ' ')

    raw_data = text.split(' ')
    words = sorted(list(set(raw_data)))

    word_to_index = {word: idx for idx, word in enumerate(words)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    
    return raw_data, word_to_index, index_to_word, words


def get_sequential_data(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append(seq)
        labels.append(label)
    return sequences, labels

def generate_text(model, start_words, word_to_index, index_to_word, max_length=10):
    start_words = parse_text(start_words)[0]
    model.eval()
    start_seq = torch.tensor([word_to_index[word] for word in start_words], dtype=torch.long).unsqueeze(0).to(device)
    generated_words = list(start_words)
    
    for _ in range(max_length):
        with torch.no_grad():
            output = model(start_seq)
            _, predicted = torch.max(output, dim=1)
            next_word = index_to_word[predicted.item()]
            generated_words.append(next_word)
            start_seq = torch.cat([start_seq[:, 1:], predicted.unsqueeze(0)], dim=1)
    
    return ' '.join(generated_words)

