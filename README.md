### Still work in progress

## Training
Start training by using this command:

  `./train.py <path/to/training_set> <path/to/model>`
  
where:
- the training set file is a UTF-8 encoded .txt file;
- the model file is a pre-existent .pt file (by default: word_guesser.pt).

The model file is not mandatory: if not specified, it will assume there is no model and will create a model file
named `word_guesser.pt`, overwriting it in case it already exists. By starting a training specifying a model file,
the training will retrain the model (for example to resume training).

## Testing
Start querying the model by using this command:

  `./query.py <path/to/test_set> <path/to/model>`
  
where:
- the test set file is a UTF-8 encoded .txt file;
- model file: (same as training).

The model file is not mandatory: if not specified, it will assume there is a model stored in `word_guesser.pt`, while
specifying a model file, the model stored in that file will be used for predictions.
