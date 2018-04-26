'''
Trains the model.
'''

import sys
import time
import random
import numpy as np
import queue
import threading
from shutil import copyfile
import pickle as pkl
import os
from model import *
import data

# Hyper-parameters
dictionary_dim = 100000
hidden_units = 2048
context_dim = 512
embedding_dim = 512
learning_rate = 0.1
chunk_dim = 1024*2000
window_dim = 10
batch_dim = 32
num_epoch = 10
print_every = 5000
save_every = 20000
q_size = 20
max_sb_len = 1000
modelfile = 'word_guesser.pt'
logfile = 'word_guesser.log'
# ---

if len(sys.argv) < 2:
    print("Error | Argument missing: training corpus needed.")
    sys.exit(1)

if sys.argv[1].find(".txt") != len(sys.argv[1])-4:
    print("Error | Bad argument: textual (.txt) corpora only.")
    sys.exit(1)

chunk_q = queue.Queue(q_size)
sent_q = queue.Queue(q_size*2)
batch_q = queue.Queue(q_size*3)

chunk_q_lock = threading.RLock()
sent_q_lock = threading.RLock()
batch_q_lock = threading.RLock()
log_lock = threading.RLock()

chunk_q_cv = threading.Condition(chunk_q_lock)
sent_q_cv = threading.Condition(sent_q_lock)
batch_q_cv = threading.Condition(batch_q_lock)

end_training_event = threading.Event()

def read_routine(filename, chunk_dim):
    base = os.path.basename(filename)
    sb_filename = 'batches/%s.b0' % (base)
    if os.path.exists(sb_filename):
        chunk = "EOF"
        with chunk_q_cv:
            while chunk_q.full():
                chunk_q_cv.wait()

            chunk_q.put(chunk)
            chunk_q_cv.notify_all()
    else:
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

        del chunk
        with sent_q_cv:
            while sent_q.full():
                sent_q_cv.wait()

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
            sent_tensor, target = data.prepare_sequence(sent_list, word, word_to_ix)
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
        batched_sents.append(preproc_sents)
        batched_targets.append(preproc_targets)

    batches = (np.array(batched_sents), np.array(batched_targets))
    return batches

def batch_routine(word_to_ix):
    while True:
        sents = ""
        with sent_q_cv:
            while sent_q.empty():
                sent_q_cv.wait()

            #get sent chunk from sent_q
            sents = sent_q.get()
            sent_q_cv.notify_all()

        if sents == "EOF":
            break

        sents_by_len = {}
        for s in sents:
            n = len(s.split())
            if n in sents_by_len:
                sents_by_len[n].append(s)
            else:
                sents_by_len[n] = [s]

        del sents

        for k,s in sents_by_len.items():
            batches = create_batches(s)

            with batch_q_cv:
                while batch_q.full():
                    batch_q_cv.wait()

                batch_q.put(batches)
                batch_q_cv.notify_all()

    #batching is over, once training thread gets no batch from batch_q, end training
    end_training_event.set()
    print("Finished batching.")
    with log_lock:
        with open(logfile, 'a') as f_log:
            f_log.write('Finished batching on %s\n' % (data.format_date(time.time())))
            f_log.flush()

def train(batches, batch_count, loss_acc, epoch, print_every, save_every):
    batch_sents, batch_targets = batches
    for batch_num in range(len(batch_sents)):
        sent_tensors = batch_sents[batch_num]
        targets = batch_targets[batch_num]

        #dynamic batch dim to avoid filling the last batch with dummy sentences
        model.batch_dim = len(sent_tensors)
        #1

        model.zero_grad()
        model.hidden = model.init_hidden()

        #2
        batch_count += 1

        #3
        input_tensor = torch.LongTensor(sent_tensors)
        input_tensor = input_tensor.cuda()
        input_tensor = autograd.Variable(input_tensor)
        
        prediction, context = model(input_tensor)

        target_tensor = torch.LongTensor(targets)
        target_tensor = target_tensor.cuda()
        target_tensor = autograd.Variable(target_tensor)

        #4
        loss = loss_fn(prediction, target_tensor)
        loss.backward()
        optimizer.step()

        loss_acc += loss.data[0]
        if batch_count % print_every == 0:
            msg = ('%s - Epoch: %d Batch: %d Loss: %f' % (data.elapsed(start), epoch, batch_count, loss_acc / batch_count))
            print('- %s' % (msg))
            with log_lock:
                with open(logfile, 'a') as f_log:
                    f_log.write('%s\n' % (msg))
                    f_log.flush()

        if batch_count % save_every == 0:
            bakfile = modelfile + ".bak"
            copyfile(modelfile, bakfile)
            torch.save(model.state_dict(), modelfile)

    return batch_count, loss_acc

def train_routine(filename, print_every, save_every, num_epoch, max_sb_len):
    base = os.path.basename(filename)
    first = True
    stored_batches = list()
    sb_count = 0
    batch_count = 0
    epoch = 0
    while epoch < num_epoch:
        epoch += 1
        loss_acc = 0
        #first epoch has to wait for batches to be produced
        if first:
            while True:
                #if not set, eventually wait for batch to be produced
                if not end_training_event.is_set():
                    with batch_q_cv:
                        while batch_q.empty():
                            batch_q_cv.wait()

                #get batches from batch_q
                try:
                    batches = batch_q.get_nowait()
                    if first:
                        print("Starting training.")
                        with log_lock:
                            with open(logfile, 'a') as f_log:
                                f_log.write('Starting training on %s\n' % (data.format_date(time.time())))
                                f_log.flush()

                    batch_count, loss_acc = train(batches, batch_count, loss_acc, epoch, print_every, save_every)
                    stored_batches.append(batches)

                    if len(stored_batches) % max_sb_len == 0:
                        sb_filename = 'batches/%s.b%d' % (base, sb_count)
                        pkl.dump(stored_batches, open(sb_filename, 'wb'))
                        sb_count += 1
                        stored_batches = list()

                except Exception:
                    break
                finally:
                    first = False
                    with batch_q_cv:
                        batch_q_cv.notify_all()

            if len(stored_batches) > 0:
                sb_filename = 'batches/%s.b%d' % (base, sb_count)
                pkl.dump(stored_batches, open(sb_filename, 'wb'))
                sb_count += 1
                stored_batches = list()

        #batches already produced for next epochs
        else:
            if sb_count == 0:
                num_epoch += 1
                sb_count = 1

            i = 0
            while True:
                sb_filename = 'batches/%s.b%d' % (base, i)
                i += 1
                if os.path.exists(sb_filename):
                    stored_batches = pkl.load(open(sb_filename, 'rb'))

                    for batches in stored_batches:
                        batch_count, loss_acc = train(batches, batch_count, loss_acc, epoch, print_every, save_every)

                    stored_batches = list()

                else:
                    break

    #training over
    msg = ('%s - Epochs: %d Batches: %d Loss: %f' % (data.elapsed(start), num_epoch, batch_count, loss_acc / (batch_count + 1)))
    msg = ('%s\nTraining ended on %s.\n' % (msg, data.format_date(time.time())))
    print(msg)
    with log_lock:
        with open(logfile, 'a') as f_log:
            f_log.write('%s\n' % (msg))
            f_log.flush()

    bakfile = modelfile + ".bak"
    copyfile(modelfile, bakfile)
    torch.save(model.state_dict(), modelfile)

print("Initializing...")
start = time.time()
with open(logfile, 'w') as f_log:
    f_log.write('Inizialization on %s\n' % (data.format_date(start)))
    f_log.flush()

word_to_ix, ix_to_word = data.init_dictionary(dictionary_dim)
model = WordGuesser(hidden_units, context_dim, embedding_dim, len(word_to_ix), batch_dim)
if len(sys.argv) == 3:
    modelfile = sys.argv[2]
    model.load_state_dict(torch.load(modelfile))

model.train()
model = model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
model.hidden = model.init_hidden()

training_set = sys.argv[1]
read_thread = threading.Thread(target=read_routine, args=[training_set, chunk_dim])
split_thread = threading.Thread(target=split_routine)
batch_thread = threading.Thread(target=batch_routine, args=[word_to_ix])
train_thread = threading.Thread(target=train_routine, args=[training_set, print_every, save_every, num_epoch, max_sb_len])

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
