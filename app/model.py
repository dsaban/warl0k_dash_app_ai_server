# --- model.py ---
# RNN with Attention for regenerating master secret from noisy obfuscated input

import torch
import torch.nn as nn
import random

def add_noise_to_tensor(tensor, vocab_size, noise_level=0.4):
    noisy = tensor.clone()
    for i in range(len(noisy)):
        if random.random() > noise_level:
            noisy[i] = random.randint(0, vocab_size - 1)
    return noisy

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        hidden = hidden.unsqueeze(0).repeat(seq_len, 1, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(1, 2, 0)
        v = self.v.unsqueeze(0).repeat(encoder_outputs.size(1), 1).unsqueeze(1)
        weights = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(weights, dim=1)

class SecretRegenerator(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.GRU(emb_dim, hidden_dim)
        self.attention = Attention(hidden_dim)
        self.decoder = nn.GRU(emb_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, seq, target_len):
        emb = self.embedding(seq)
        enc_out, hidden = self.encoder(emb)
        logits = []
        input_tok = seq[0]
        for _ in range(target_len):
            emb_tok = self.embedding(input_tok).unsqueeze(0)
            attn_weights = self.attention(hidden[-1], enc_out)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_out.permute(1, 0, 2)).permute(1, 0, 2)
            dec_input = torch.cat((emb_tok, context), dim=2)
            out, hidden = self.decoder(dec_input, hidden)
            combined = torch.cat((out.squeeze(0), context.squeeze(0)), dim=1)
            logits.append(self.out(combined))
            input_tok = logits[-1].argmax(1)
        return torch.stack(logits)

# Training and evaluation utilities

def train_secret_regenerator(secret_str, vocab, epochs=100, input_override=None):
    def text_to_tensor(text):
        return torch.tensor([vocab.index(c) for c in text], dtype=torch.long).unsqueeze(1)

    output_seq = text_to_tensor(secret_str)
    input_seq = text_to_tensor(input_override) if input_override else output_seq
    vocab_size = len(vocab)
    model = SecretRegenerator(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        noisy_input = add_noise_to_tensor(input_seq.squeeze(1), vocab_size, noise_level=0.2).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(noisy_input, output_seq.size(0))
        loss = criterion(logits.view(-1, vocab_size), output_seq.view(-1))
        loss.backward()
        optimizer.step()

    return model


def evaluate_secret_regenerator(model, noisy_seq, vocab):
    model.eval()
    with torch.no_grad():
        logits = model(noisy_seq, noisy_seq.size(0))
        preds = logits.argmax(dim=2).squeeze(1)
        return ''.join([vocab[i] for i in preds.tolist()])
