from dotenv import load_dotenv
load_dotenv()
from lib.model import Model, vocab_to_char2idx_idx2char, EMBEDDING_DIM, UNITS
from lib import db
from lib.db import Tweet
import tensorflow as tf
tf.enable_eager_execution()
import time
import os
import re

BATCH_SIZE = 64

# The maximum length sentence we want for a single input in characters
SEQ_LENGTH = 280

print("Starting up")

db.init_db()
tweets_objects = db.session.query(Tweet).order_by(Tweet.date.desc()).limit(200)
regex = re.compile(r"((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?", re.IGNORECASE)
tweet_list = map(lambda tweet: regex.sub("", tweet.text), tweets_objects)
text =  "\n".join(tweet_list)

# The unique characters in the file
vocab = sorted(set(text))
with open(os.getenv('VOCAB_FILE','./data/vocab.txt'), 'w') as text_file:
  text_file.write("\n".join(vocab))
print ('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters to indices
(char2idx, idx2char) = vocab_to_char2idx_idx2char(vocab)

text_as_int = np.array([char2idx[c] for c in text])

for char,_ in zip(char2idx, range(20)):
  print('{:6s} ---> {:4d}'.format(repr(char), char2idx[char]))

# Show how the first 13 characters from the text are mapped to integers
print ('{} ---- characters mapped to int ---- > {}'.format(text[:13], text_as_int[:13]))

# Create training examples / targets
chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(SEQ_LENGTH+1, drop_remainder=True)

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

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences, 
# so it doesn't attempt to shuffle the entire sequence in memory. Instead, 
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

model = Model(len(char2idx), EMBEDDING_DIM, UNITS)

# Using adam optimizer with default arguments
optimizer = tf.train.AdamOptimizer()

model.build(tf.TensorShape([BATCH_SIZE, SEQ_LENGTH]))

model.summary()

# Training step
EPOCHS = 30

# Directory where the checkpoints will be saved
checkpoint_dir = os.getenv('TRAINING_CHECKPOINT_DIR','./data/training_checkpoints')
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# Checkpoint instance
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

# Using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
def loss_function(real, preds):
  return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)

for epoch in range(EPOCHS):
  start = time.time()
  
  # initializing the hidden state at the start of every epoch
  # initally hidden is None
  hidden = model.reset_states()
  
  for (batch, (inp, target)) in enumerate(dataset):
    with tf.GradientTape() as tape:
      # feeding the hidden state back into the model
      # This is the interesting step
      predictions = model(inp)
      loss = loss_function(target, predictions)
        
      grads = tape.gradient(loss, model.variables)
      optimizer.apply_gradients(zip(grads, model.variables))

      if batch % 100 == 0:
        print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,
                                                      batch,
                                                      loss))
      # saving (checkpoint) the model every 5 epochs
      if (epoch + 1) % 5 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

      print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
      print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
