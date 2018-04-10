#!/home/emanuele/anaconda3/bin/python3.6
'''
Trains the model.
'''

import sys
import time
import random
import numpy
import queue
import threading
from model import *
import data

# Hyper-parameters
#dictionary_dim = 1000000
dictionary_dim = 400000
hidden_units = 512
context_dim = 256
embedding_dim = 256
learning_rate = 0.1
chunk_dim = 1024*1000
window_dim = 10
batch_dim = 32
num_epoch = 1
save_every = 5000
modelfile = 'word_guesser.pt'
logfile = 'word_guesser.log'
# ---

if len(sys.argv) < 2:
    print("Error | Argument missing: training corpus needed.")
    sys.exit(1)

if sys.argv[1].find(".txt") != len(sys.argv[1])-4:
    print("Error | Bad argument: textual (.txt) corpora only.")
    sys.exit(1)

chunk_q = queue.Queue(10)
sent_q = queue.Queue()
batch_q = queue.Queue()

chunk_q_lock = threading.RLock()
sent_q_lock = threading.RLock()
batch_q_lock = threading.RLock()
log_lock = threading.RLock()

chunk_q_cv = threading.Condition(chunk_q_lock)
sent_q_cv = threading.Condition(sent_q_lock)
batch_q_cv = threading.Condition(batch_q_lock)

end_training_event = threading.Event()

def read_routine(filename, chunk_dim):
    print('Reading %s' % (filename))
    #read file in chunks
    with open(filename) as f_in:
        while True:
            chunk = f_in.readlines(chunk_dim)
            if not chunk:
                chunk = "EOF"

            with chunk_q_cv:
                while chunk_q.full():
                    chunk_q_cv.wait()

                chunk_q.put(chunk)
                chunk_q_cv.notify_all()

            if chunk == "EOF":
                break

    print("Finished reading file.")
    with log_lock:
        with open(logfile, 'a') as f_log:
            f_log.write('Finished reading file on %s\n' % (data.format_date(time.time())))
            f_log.flush()

def split_routine():
    while True:
        chunk = ""
        with chunk_q_cv:
            while chunk_q.empty():
                chunk_q_cv.wait()

            #get chunk from chunk_q
            chunk = chunk_q.get()
            chunk_q_cv.notify_all()

        #split chunk
        sents = ""
        if chunk == "EOF":
            sents = "EOF"
        else:
            sents = data.split_to_sentences(chunk)

        with sent_q_cv:
            sent_q.put(sents)
            sent_q_cv.notify_all()

        if sents == "EOF":
            break

    print("Finished splitting file.")
    with log_lock:
        with open(logfile, 'a') as f_log:
            f_log.write('Finished splitting file on %s\n' % (data.format_date(time.time())))
            f_log.flush()

def create_batches(sentences):
    batches = ""
    batched_sents = list()
    batched_targets = list()
    preproc_sents = list()
    preproc_targets = list()

    for sent in sentences:
        masked_sents, words = data.mask_words(sent)
        if masked_sents is None:
            continue

        for i in range(len(masked_sents)):
            sent_list = masked_sents[i]
            word = words[i]
            sent_tensor, target = data.prepare_sequence(sent_list, word, word_to_ix, window_dim)
            if sent_tensor is None:
                continue

            preproc_sents.append(sent_tensor)
            preproc_targets.append(target)
            if len(preproc_sents) == batch_dim:
                batched_sents.append(preproc_sents)
                batched_targets.append(preproc_targets)
                preproc_sents = list()
                preproc_targets = list()

    if len(preproc_sents) > 0:
        '''
        #pad the remaining batch dim
        sent_list = ['PAD'] * (2 * window_dim + 1)
        sent_tensor = data.prepare_tensor(sent_list, word_to_ix)
        target = word_to_ix['PAD']
        while len(preproc_sents) < batch_dim:
            preproc_sents.append(sent_tensor)
            preproc_targets.append(target)
        '''
        batched_sents.append(preproc_sents)
        batched_targets.append(preproc_targets)

    batches = (numpy.array(batched_sents), numpy.array(batched_targets))
    return batches

def batch_routine(word_to_ix):
    while True:
        sents = ""
        with sent_q_cv:
            while sent_q.empty():
                sent_q_cv.wait()

            #get sent chunk from sent_q
            sents = sent_q.get()

        if sents == "EOF":
            break

        sents_by_len = {}
        for s in sents:
            n = len(s.split())
            if n in sents_by_len:
                sents_by_len[n].append(s)
            else:
                sents_by_len[n] = [s]

        for k,s in sents_by_len.items():
            batches = create_batches(s)

            with batch_q_cv:
                batch_q.put(batches)
                batch_q_cv.notify_all()

        if sents == "EOF":
            break

    #batching is over, once training thread gets no batch from batch_q, end training
    end_training_event.set()
    print("Finished batching.")
    with log_lock:
        with open(logfile, 'a') as f_log:
            f_log.write('Finished batching on %s\n' % (data.format_date(time.time())))
            f_log.flush()

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == autograd.Variable:
        return autograd.Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(batches, batch_count, loss_acc, epoch, save_every):
    #exit(1)
    batch_sents, batch_targets = batches
    for batch_num in range(len(batch_sents)):
        sent_tensors = batch_sents[batch_num]

        #dynamic batch dim to avoid filling the last batch with dummy sentences
        curr_batch_dim = len(sent_tensors)
        if curr_batch_dim <= batch_dim:
            model.batch_dim = curr_batch_dim
        #1

        model.zero_grad()
        hidden = model.init_hidden()
        #model.hidden = [repackage_hidden(hidden) for hidden in model.hidden]

        #2
        batch_count += 1
        #sent_tensors = batch_sents[batch_num]
        targets = batch_targets[batch_num]

        '''
        if epoch == 0:
            with open("model_input.txt", 'a') as f_out:
                msg = ('%s\t%s\n' % (sent_tensors, targets))
                f_out.write(msg)
                f_out.flush()
        '''

        #3
        input_tensor = torch.LongTensor(sent_tensors)
        input_tensor = input_tensor.cuda()
        input_tensor = autograd.Variable(input_tensor)

        prediction, context = model(input_tensor, hidden)

        '''
        if batch_count % 32 == 0:
            print('Epoch: %d' % (epoch))
            for i, p in enumerate(prediction):
                sorted_val = sorted(enumerate(numpy.array(p.data)), key=lambda x : x[1], reverse=True)
                print([(ix_to_word[x[0]], x[1]) for x in sorted_val[:5]])
            print("")
        #'''

        target_tensor = torch.LongTensor(targets)
        target_tensor = target_tensor.cuda()
        target_tensor = autograd.Variable(target_tensor)

        #4
        loss = loss_fn(prediction, target_tensor)
        loss.backward()
        optimizer.step()

        loss_acc += loss.data[0]
        if batch_count % save_every == 0:
            msg = ('%s - Epoch: %d Batch: %d Loss: %f' % (data.elapsed(start), epoch, batch_count, loss_acc / batch_count))
            print('- %s' % (msg))
            with log_lock:
                with open(logfile, 'a') as f_log:
                    f_log.write('%s\n' % (msg))
                    f_log.flush()
            #torch.save(model, modelfile)
            torch.save(model.state_dict(), modelfile)
    return batch_count, loss_acc

def train_routine(save_every):
    first = False
    stored_batches = list()
    batch_count = 0
    for epoch in range(num_epoch):
        loss_acc = 0
        #first epoch has to wait for batches to be produced
        if epoch == 0:
            while True:
                #if not set, eventually wait for batch to be produced
                if not end_training_event.is_set():
                    with batch_q_cv:
                        while batch_q.empty():
                            batch_q_cv.wait()

                #get batches from batch_q
                try:
                    batches = batch_q.get_nowait()
                    if first == False:
                        print("Starting training.")
                        with log_lock:
                            with open(logfile, 'a') as f_log:
                                f_log.write('Starting training on %s\n' % (data.format_date(time.time())))
                                f_log.flush()
                    first = True
                    stored_batches.append(batches)
                    batch_count, loss_acc = train(batches, batch_count, loss_acc, epoch, save_every)
                except Exception:
                    break
        #batches already produced for next epochs
        else:
            for batches in stored_batches:
                batch_count, loss_acc = train(batches, batch_count, loss_acc, epoch, save_every)

    #training over
    msg = ('%s - Epochs: %d Batches: %d Loss: %f' % (data.elapsed(start), num_epoch, batch_count, loss_acc / (batch_count + 1)))
    msg = ('%s\nTraining ended on %s.\n' % (msg, data.format_date(time.time())))
    print(msg)
    with log_lock:
        with open(logfile, 'a') as f_log:
            f_log.write('%s\n' % (msg))
            f_log.flush()
    #torch.save(model, modelfile)
    torch.save(model.state_dict(), modelfile)

print("Initializing...")
torch.backends.cudnn.benchmark = True

start = time.time()
with open(logfile, 'w') as f_log:
    f_log.write('Inizialization on %s\n' % (data.format_date(start)))
    f_log.flush()

word_to_ix, ix_to_word = data.init_dictionary(dictionary_dim)
'''
word_to_ix = {"$":0, "PAD":1, "UNK":2}
ix_to_word = ["$", "PAD", "UNK"]

with open(sys.argv[1]) as f_in:
    for l in f_in.readlines():
        for w in l.strip().split(' '):
            if w not in word_to_ix:
                word_to_ix[w] = len(word_to_ix)
                ix_to_word.append(w)
#'''

model = WordGuesser(hidden_units, context_dim, embedding_dim, len(word_to_ix), batch_dim, window_dim)
if len(sys.argv) == 3:
    modelfile = sys.argv[2]
    #model = torch.load(modelfile)
    model.load_state_dict(torch.load(modelfile))
#else:
    #model = WordGuesser(hidden_units, context_dim, embedding_dim, len(word_to_ix), batch_dim, window_dim)
model.train()
model = model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
model.hidden = model.init_hidden()

read_thread = threading.Thread(target=read_routine, args=[sys.argv[1], chunk_dim])
split_thread = threading.Thread(target=split_routine)
batch_thread = threading.Thread(target=batch_routine, args=[word_to_ix])
train_thread = threading.Thread(target=train_routine, args=[save_every])

end_training_event.clear()

print("Initialization done.")

read_thread.start()
split_thread.start()
batch_thread.start()
train_thread.start()

read_thread.join()
split_thread.join()
batch_thread.join()
train_thread.join()

print("EXIT")
