# Neural WSD

This project aims to replicate Google's ["Semi-supervised Word Sense Disambiguation with Neural Models"](https://research.google.com/pubs/pub45729.html?authuser=0), covering only the LSTM language modeler part with
unsupervised training.

Dictionary built using [Google English One Million 1-grams](http://storage.googleapis.com/books/ngrams/books/datasetsv2.html), sorting each word by its global frequency.

**This project is for educational purposes only**

## How to use

### Setup:

- Python 3.6.3 (Anaconda custom 64-bit)
- PyTorch 0.3.1 (0.4.0 might not work due to [torch.Tensor and autograd.Variable changes](https://github.com/pytorch/pytorch/releases/tag/v0.4.0))
- CUDA 8
- spaCy v2.0 with English models (more [here](https://spacy.io/usage/))
- project folder must contain a folder named `batches` in the same directory of the train.py file

### Training

Start training by using this command:

  `python train.py <path/to/training_set> <path/to/model>`
  
where:
- the training set file is a UTF-8 encoded .txt file;
- the model file is a pre-existent .pt file (by default: `word_guesser.pt`).

The model file is not mandatory: if not specified, it will assume there is no model and will create a model file
named `word_guesser.pt`, overwriting it in case it already exists. By starting a training specifying a model file,
the training will retrain the model (for example to resume training).

### Testing

Start querying the model by using this command:

  `python query.py <path/to/test_set> <path/to/model>`
  
where:
- the test set file is a UTF-8 encoded .txt file;
- model file: (same as training).

The model file is not mandatory: if not specified, it will assume there is a model stored in `word_guesser.pt`, while
specifying a model file, the model stored in that file will be used for predictions.

## Features

- Multi-threaded operation in order to read from the training corpus, split to sentences, batching, and training simultaneously (_producer-consumer pattern_)
- Low RAM usage due to sized queues between threads and periodic dumps of created batches
- Sentences are never padded, instead they are organized by their length and then created batches from sentences of all the same length
- Dynamic batch size: will try to create batches of maximal size (hyper-parameter `batch_dim`) as much as possible, but batches smaller than the chosen size will not be padded

## Known bugs/problems

- Missing `batches` folder creation if not present
- Training corpus only accepted format is UTF-8 encoded plain text
- Slow on computation of large training corpus, might become faster implementing hierarchical softmax or negative sampling

## Consulted resources

- [PyTorch Tutorials](http://pytorch.org/tutorials/)
- [Practical PyTorch](https://github.com/spro/practical-pytorch)
- [The Incredible PyTorch](https://github.com/ritchieng/the-incredible-pytorch)
- [Optimizing PyTorch training code](https://www.sagivtech.com/2017/09/19/optimizing-pytorch-training-code/)
- [Word Sense Disambiguation with LSTM: Do We Really Need 100 Billion Words?](https://github.com/cltl/wsd-dynamic-sense-vector)

