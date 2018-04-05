'''
Contains the language model to predict a held-out word
given the surrounding context of a sentence.
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class WordGuesser(nn.Module):
    def __init__(self, hidden_dim, context_dim, embedding_dim, vocabulary_dim, batch_dim, window_dim):
        super(WordGuesser, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_dim = batch_dim
        self.window_dim = window_dim
        self.word_embeddings = nn.Embedding(vocabulary_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.extract_context = nn.Linear((2 * window_dim + 1) * hidden_dim, context_dim)
        self.predict = nn.Linear(context_dim, vocabulary_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, self.batch_dim, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_dim, self.hidden_dim).cuda()))

    def forward(self, sentence):
        #0 rimpiazza parola w con $ --> nel training
        #1 consuma tutte le parole della frase
        embeddings = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeddings.permute(1, 0, 2), self.hidden)
        lstm_out = lstm_out.view(-1, (2 * self.window_dim + 1) * self.hidden_dim)

        #2 extract_context --> contesto c
        context = self.extract_context(lstm_out)

        #3 softmax per predire la parola w dal contesto c
        prediction = self.predict(context)
        #out = F.softmax(prediction, dim=1)
        return prediction, context
