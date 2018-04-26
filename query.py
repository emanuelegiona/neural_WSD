'''
Executes queries on a trained model to gather its accuracy.
'''

import sys
import random
from model import *
import data
import time
import numpy

# Hyper-parameters
hidden_units = 2048
context_dim = 512
embedding_dim = 512
dictionary_dim = 100000
window_dim = 10
batch_dim = 32
start = time.time()
modelfile = 'word_guesser.pt'
# ---

if len(sys.argv) < 2:
    print("Error | Argument missing: testing corpus needed.")
    sys.exit(1)

if sys.argv[1].find(".txt") != len(sys.argv[1])-4:
    print("Error | Bad argument: textual (.txt) corpora only.")
    sys.exit(1)

print("Initializing...")
word_to_ix, ix_to_word = data.init_dictionary(dictionary_dim)

if len(sys.argv) == 3:
    modelfile = sys.argv[2]

model = WordGuesser(hidden_units, context_dim, embedding_dim, len(word_to_ix), 1)
model.load_state_dict(torch.load(modelfile))
model.train(False)
model.hidden = model.init_hidden()
model = model.cuda()
test_sentences = []

with open(sys.argv[1]) as f_in:
    for line in f_in:
        ss = line.strip().split("\t")
        if len(ss[0]) <= 0:
            continue
        test_sentences.append(ss)
print("Done.")

print("Querying...")
warm_up = 3

#warming up internal gradients before model.eval()
while warm_up != 0:
    warm_up -= 1
    if warm_up == 0:
        model.eval()

    for t_sent in test_sentences:
        sent = t_sent[0]
        target = t_sent[1]
        sent_tensor, _ = data.prepare_sequence(sent.split(), target, word_to_ix)

        input_tensor = torch.LongTensor([sent_tensor])
        input_tensor = input_tensor.cuda()
        input_tensor = autograd.Variable(input_tensor)

        model.zero_grad()
        model.hidden = model.init_hidden()
        predictions, context = model(input_tensor)

        if warm_up == 0:
            word_ids = predictions.data.topk(5)[1][0].cpu().numpy()
            word_predictions = []
            for i in word_ids:
                word_predictions.append(ix_to_word[i])

            print('Sentence: %s\nWord: %s' % (sent, target))
            print("Predictions:\n", word_predictions)
            print("")

print("Done.")
