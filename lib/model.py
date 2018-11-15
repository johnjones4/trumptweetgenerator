import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

# The embedding dimension 
EMBEDDING_DIM = 256

# Number of RNN units
UNITS = 1024

def vocab_to_char2idx_idx2char(vocab):
  char2idx = {u:i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)
  return (char2idx, idx2char)

class Model(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, units):
    super(Model, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim)

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
