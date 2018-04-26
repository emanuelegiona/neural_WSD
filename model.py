'''
Contains the language model to predict a held-out word
given the surrounding context of a sentence.
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

torch.manual_seed(1)

class WordGuesser(nn.Module):
    def __init__(self, hidden_dim, context_dim, embedding_dim, vocabulary_dim, batch_dim):
        super(WordGuesser, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_dim = batch_dim
        self.word_embeddings = nn.Embedding(vocabulary_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.extract_context = nn.Linear(hidden_dim, context_dim)
        self.predict = nn.Linear(context_dim, vocabulary_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, self.batch_dim, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_dim, self.hidden_dim).cuda()))

    def forward(self, sentence):
        embeddings = self.word_embeddings(sentence)
        packed = embeddings.permute(1, 0, 2)
        out, self.hidden = self.lstm(packed, self.hidden)
        lstm_out = out[-1]
        context = self.extract_context(lstm_out)
        prediction = self.predict(context)
        return prediction, context
