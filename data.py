'''
Contains various utility functions to manipulate training and query data.
'''

from model import *
import random
import spacy
import unicodedata
import string
import time
import math
import copy
from tqdm import tqdm

def custom_pipeline(nlp):
    return (nlp.tagger, nlp.parser)

nlp = spacy.load('en', create_pipeline=custom_pipeline)
all_letters = string.ascii_letters + "1234567890-.,;'"
n_letters = len(all_letters)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
            and c in all_letters
    )

def init_dictionary(dictionary_dim):
    word_to_ix = {'PAD' : 0, '$' : 1, 'UNK' : 2}
    ix_to_word = ["PAD", "$", "UNK"]

    with open("training/dictionary.txt") as f_in:
        counter = 0
        for line in f_in:
            if counter == dictionary_dim:
                break

            w = unicode_to_ascii(line)
            if w not in word_to_ix:
                word_to_ix[w] = len(word_to_ix)
                ix_to_word.append(w)
                counter += 1

    return (word_to_ix, ix_to_word)

def split_to_sentences(lines):
    sentences = []
    for line in lines:
        if line.startswith("TITLE:") or line.startswith("=="):
            continue

        tokens = nlp(line)
        for sent in tokens.sents:
            words = []
            for word in sent:
                words.append(unicode_to_ascii(word.text))
            sentences.append(' '.join(words).strip())

    return sentences

def mask_words(sent):
    sent_list = sent.split()
    if len(sent_list) <= 0:
        return None, None

    sents = list()
    targets = list()
    for i in range(len(sent_list)):
        new_list = copy.deepcopy(sent_list)
        new_list[i] = "$"
        sents.append(new_list)
        targets.append(sent_list[i])
    return sents, targets

def prepare_sequence(sent_list, word, to_ix):
    sent_tensor = prepare_tensor(sent_list, to_ix)
    if sent_tensor is None:
        return None, None

    target = to_ix[word] if word in to_ix else to_ix['UNK']

    return sent_tensor, target

def prepare_tensor(seq, to_ix):
    try:
        ids = [to_ix[w] if w in to_ix else to_ix['UNK'] for w in seq]
    except:
        ids = None
    finally:
        return ids

def _prepare_sequence(seq, to_ix):
    try:
        ids = [to_ix[w] if w in to_ix else to_ix['UNK'] for w in seq]
    except:
        ids = None
    finally:
        return ids

def elapsed(start):
    now = time.time()
    s = now - start
    m = math.floor(s/60)
    s -= m*60
    s = float('%.3f' % (s))
    if s < 10:
        s = ('0%.3f' % (s))
    ret = ('Time elapsed: %sm %ss' % (m, s))
    return ret

def format_date(t):
    date = time.localtime(t)
    m = int(date.tm_mon)
    if m < 10:
        m = ('0%d' % (m))
    d = int(date.tm_mday)
    if d < 10:
        d = ('0%d' % (d))
    hr = int(date.tm_hour)
    if hr < 10:
        hr = ('0%d' % (hr))
    mn = int(date.tm_min)
    if mn < 10:
        mn = ('0%d' % (mn))
    sc = int(date.tm_sec)
    if sc < 10:
        sc = ('0%d' % (sc))
    ret = ('%s/%s/%s %s:%s:%s' % (date.tm_year, m, d, hr, mn, sc))
    return ret
