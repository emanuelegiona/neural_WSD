#!/home/emanuele/anaconda3/bin/python3.6
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
dictionary_dim = 400000
window_dim = 20
batch_dim = 32
print_every = 10000
start = time.time()
modelfile = 'word_guesser.pt'
train2 = False
# ---

if len(sys.argv) < 2:
    print("Error | Argument missing: testing corpus needed.")
    sys.exit(1)

if sys.argv[1].find(".txt") != len(sys.argv[1])-4:
    print("Error | Bad argument: textual (.txt) corpora only.")
    sys.exit(1)

print("Initializing...")
#word_to_ix, ix_to_word = data.init_dictionary(dictionary_dim)
#'''
if train2:
    lines = open('training/overfit.txt').readlines()
    sentences = data.split_to_sentences(lines)
    word_to_ix=dict()
    ix_to_word =dict()
    i = 0
    for s in sentences:
        for t in s.strip().split(' '):
            if t in word_to_ix:
                continue
            word_to_ix[t] = i
            ix_to_word[i] = t
            i+= 1
    word_to_ix['$'] = i
    ix_to_word[i] = '$'
else:
    word_to_ix = {"$":0, "PAD":1, "UNK":2}
    ix_to_word = ["$", "PAD", "UNK"]

    with open("training/overfit.txt") as f_in:
        for l in f_in.readlines():
            for w in l.strip().split(' '):
                if w not in word_to_ix:
                    word_to_ix[w] = len(word_to_ix)
                    ix_to_word.append(w)
#'''

if len(sys.argv) == 3:
    modelfile = sys.argv[2]
'''
model = torch.load(modelfile)
model.train(False)
'''

hidden_units = 512
context_dim = 256
embedding_dim = 256
if train2:
    model = WordGuesser(hidden_units, context_dim, embedding_dim, len(word_to_ix), 1, 13)
else:
    model = WordGuesser(hidden_units, context_dim, embedding_dim, len(word_to_ix), 1, window_dim)
model.load_state_dict(torch.load(modelfile))
model.train(False)
model.hidden = model.init_hidden()
#model = torch.load('word_guesser.pt')
model = model.cuda()
#sent_count = 0
test_sentences = []

with open(sys.argv[1]) as f_in:
    for line in f_in:
        ss = line.strip().split("\t")
        if len(ss[0]) <= 0:
            continue
        test_sentences.append(ss)
print("Done.")

'''
print("Starting querying...")
for sent in test_sentences:
    sent_list = sent[0].split()
    if len(sent_list) <= 0:
        continue

    word = sent[1]
    sent_tensor = data.prepare_tensor(sent_list, word_to_ix)
    if sent_tensor is None:
        print('Sentence: %s\nFound word out of dictionary\n' % (sent[0]))
        continue

    sent_tensor = torch.LongTensor(sent_tensor)
    sent_tensor = sent_tensor.cuda()
    sent_tensor = autograd.Variable(sent_tensor)

    prediction, c = model(sent_tensor)
    prediction = F.softmax(prediction)

    word_ids = prediction.data.topk(10)[1][0].cpu().numpy()
    word_predictions = []
    for i in word_ids:
        word_predictions.append(ix_to_word[i])

    print('Sentence: %s | Word: %s' % (sent[0], word))
    print("Predictions:\n", word_predictions)
    print("")
print("Done.")
'''

print("Querying...")
warm_up = 3
while warm_up != 0:
    warm_up -= 1
    if warm_up == 0:
        model.eval()

    for t_sent in test_sentences:
        sent = t_sent[0]
        target = t_sent[1]
        #sent_tensor, _ = data.prepare_sequence(sent.split(), target, word_to_ix, window_dim, False)
        #sent_tensor = numpy.array([sent_tensor])

        sent_tensor = [word_to_ix[w] for w in sent.split()]

        input_tensor = torch.LongTensor([sent_tensor])
        input_tensor = input_tensor.cuda()
        input_tensor = autograd.Variable(input_tensor)

        hidden = model.init_hidden()
        predictions, context = model(input_tensor, hidden)
        #predictions = F.softmax(predictions, dim=1)

        #'''
        if warm_up == 0:
            #'''
            print('Sentence: %s\nTarget: %s' % (sent, target))
            for i, prediction in enumerate(predictions):
                sorted_val = sorted(enumerate(numpy.array(prediction.data)), key=lambda x : x[1], reverse=True)
                print([(ix_to_word[x[0]], x[1]) for x in sorted_val[:5]]) #, ix_to_word[Y[i]]
            print("")
            #'''

            '''
            word_ids = predictions.data.topk(10)[1][0].cpu().numpy()
            word_predictions = []
            for i in word_ids:
                word_predictions.append(ix_to_word[i])

            print('Sentence: %s | Word: %s' % (sent, target))
            print("Predictions:\n", word_predictions)
            print("")
            #'''
        #'''

print("Done.")
