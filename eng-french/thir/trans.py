import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import random

# Tokenization with spacy
spacy_en = spacy.load('en_core_web_sm')
spacy_fr = spacy.load('fr_core_news_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

# Dataset
data = [
    ("I am happy.", "Je suis heureux."),
    ("She is reading a book.", "Elle lit un livre."),
    ("The cat is on the table.", "Le chat est sur la table."),
    ("He runs fast.", "Il court vite."),
    ("We are learning.", "Nous apprenons.")
]

# Build vocabularies
def build_vocab(data, tokenizer, min_freq=1):
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    idx = 4
    word_freq = {}
    for sentence in data:
        for word in tokenizer(sentence):
            word_freq[word] = word_freq.get(word, 0) + 1
            if word_freq[word] >= min_freq and word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

src_data = [pair[0] for pair in data]
tgt_data = [pair[1] for pair in data]
src_vocab = build_vocab(src_data, tokenize_en)
tgt_vocab = build_vocab(tgt_data, tokenize_fr)

# Convert text to indices
def text_to_indices(text, vocab, tokenizer):
    indices = [vocab['<sos>']] + [vocab.get(word, vocab['<unk>']) for word in tokenizer(text)] + [vocab['<eos>']]
    return indices

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)  # outputs: [src_len, batch_size, hid_dim * 2], hidden: [2, batch_size, hid_dim]
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))  # [batch_size, hid_dim]
        return outputs, hidden

# Attention
class Attention(nn.Module):
    def __init__(self, hid_dim, enc_bidirect=True):
        super().__init__()
        enc_dim = hid_dim * 2 if enc_bidirect else hid_dim
        self.attn = nn.Linear(hid_dim + enc_dim, hid_dim)  # hid_dim (decoder) + hid_dim * 2 (encoder outputs)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hid_dim]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [batch_size, src_len, hid_dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        return torch.softmax(attention, dim=1)

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, enc_bidirect=True):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        enc_dim = hid_dim * 2 if enc_bidirect else hid_dim
        self.rnn = nn.GRU(enc_dim + emb_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim + enc_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hid_dim, enc_bidirect)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]
        a = self.attention(hidden, encoder_outputs)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        weighted = torch.bmm(a, encoder_outputs.transpose(0, 1))  # [batch_size, 1, hid_dim * 2]
        weighted = weighted.transpose(0, 1)  # [1, batch_size, hid_dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [1, batch_size, emb_dim + hid_dim * 2]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))  # output: [1, batch_size, hid_dim], hidden: [1, batch_size, hid_dim]
        embedded = embedded.squeeze(0)  # [batch_size, emb_dim]
        output = output.squeeze(0)  # [batch_size, hid_dim]
        weighted = weighted.squeeze(0)  # [batch_size, hid_dim * 2]
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))  # [batch_size, output_dim]
        return prediction, hidden.squeeze(0)

# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# Training Setup
INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(tgt_vocab)
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT, enc_bidirect=True)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=src_vocab['<pad>'])

# Prepare Data for Training
def prepare_batch(src_sentences, tgt_sentences, src_vocab, tgt_vocab):
    src_indices = [text_to_indices(sent, src_vocab, tokenize_en) for sent in src_sentences]
    tgt_indices = [text_to_indices(sent, tgt_vocab, tokenize_fr) for sent in tgt_sentences]
    max_src_len = max(len(s) for s in src_indices)
    max_tgt_len = max(len(t) for t in tgt_indices)
    src_padded = torch.tensor([s + [src_vocab['<pad>']] * (max_src_len - len(s)) for s in src_indices]).t().to(device)
    tgt_padded = torch.tensor([t + [tgt_vocab['<pad>']] * (max_tgt_len - len(t)) for t in tgt_indices]).t().to(device)
    return src_padded, tgt_padded

# Training Loop (Fixed)
def train(model, optimizer, criterion, src, tgt, clip=1):
    model.train()
    epoch_loss = 0
    src, tgt = prepare_batch(src, tgt, src_vocab, tgt_vocab)
    optimizer.zero_grad()
    output = model(src, tgt)
    output_dim = output.shape[-1]
    output = output[1:].reshape(-1, output_dim)  # Also use reshape for output
    tgt = tgt[1:].reshape(-1)  # Fix: Use reshape instead of view
    loss = criterion(output, tgt)
    if not torch.isnan(loss):  # Check for valid loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss

# Train for a few epochs
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    loss = train(model, optimizer, criterion, src_data, tgt_data)
    print(f'Epoch: {epoch+1:02} | Loss: {loss:.3f}')

# Inference (Translation)
def translate_sentence(sentence, src_vocab, tgt_vocab, model, device, max_len=50):
    model.eval()
    tokens = tokenize_en(sentence)
    src_indices = [src_vocab['<sos>']] + [src_vocab.get(token, src_vocab['<unk>']) for token in tokens] + [src_vocab['<eos>']]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    trg_indices = [tgt_vocab['<sos>']]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)
        if pred_token == tgt_vocab['<eos>']:
            break
    trg_tokens = [list(tgt_vocab.keys())[list(tgt_vocab.values()).index(i)] for i in trg_indices]
    return ' '.join(trg_tokens[1:-1])

# Test the model
test_sentence = "I am happy."
translation = translate_sentence(test_sentence, src_vocab, tgt_vocab, model, device)
print(f"Input: {test_sentence}")
print(f"Translation: {translation}")