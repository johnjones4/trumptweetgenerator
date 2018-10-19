from lib import db
from lib.db import Tweet
import re
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

def load_text():
  db.init_db()
  tweets_objects = db.session.query(Tweet).order_by(Tweet.date.desc())
  regex = re.compile(r"((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?", re.IGNORECASE)
  tweet_list = map(lambda tweet: regex.sub("", tweet.text), tweets_objects)
  return "\n".join(tweet_list)

def init_model():
  text = load_text()

  # The unique characters in the file
  vocab = sorted(set(text))
  print ('{} unique characters'.format(len(vocab)))

  # Creating a mapping from unique characters to indices
  char2idx = {u:i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)

  text_as_int = np.array([char2idx[c] for c in text])

  for char,_ in zip(char2idx, range(20)):
    print('{:6s} ---> {:4d}'.format(repr(char), char2idx[char]))

  # Show how the first 13 characters from the text are mapped to integers
  print ('{} ---- characters mapped to int ---- > {}'.format(text[:13], text_as_int[:13]))

  # The maximum length sentence we want for a single input in characters
  seq_length = 280

  # Create training examples / targets
  chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length+1, drop_remainder=True)

  for item in chunks.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

  def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

  dataset = chunks.map(split_input_target)

  for input_example, target_example in  dataset.take(1):
    print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

  for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

  # Batch size 
  BATCH_SIZE = 64

  # Buffer size to shuffle the dataset
  # (TF data is designed to work with possibly infinite sequences, 
  # so it doesn't attempt to shuffle the entire sequence in memory. Instead, 
  # it maintains a buffer in which it shuffles elements).
  BUFFER_SIZE = 10000

  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

  # Length of the vocabulary in chars
  vocab_size = len(vocab)

  # The embedding dimension 
  embedding_dim = 256

  # Number of RNN units
  units = 1024

  model = Model(vocab_size, embedding_dim, units)

  # Using adam optimizer with default arguments
  optimizer = tf.train.AdamOptimizer()

  model.build(tf.TensorShape([BATCH_SIZE, seq_length]))

  model.summary()

  return model, dataset, optimizer

class Model(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, units):
    super(Model, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    if tf.test.is_gpu_available():
      self.gru = tf.keras.layers.CuDNNGRU(self.units, 
                                          return_sequences=True, 
                                          recurrent_initializer='glorot_uniform',
                                          stateful=True)
    else:
      self.gru = tf.keras.layers.GRU(self.units, 
                                     return_sequences=True, 
                                     recurrent_activation='sigmoid', 
                                     recurrent_initializer='glorot_uniform', 
                                     stateful=True)

    self.fc = tf.keras.layers.Dense(vocab_size)
        
  def call(self, x):
    embedding = self.embedding(x)
    
    # output at every time step
    # output shape == (batch_size, seq_length, hidden_size) 
    output = self.gru(embedding)
    
    # The dense layer will output predictions for every time_steps(seq_length)
    # output shape after the dense layer == (seq_length * batch_size, vocab_size)
    prediction = self.fc(output)
    
    # states will be used to pass at every step to the model while training
    return prediction
