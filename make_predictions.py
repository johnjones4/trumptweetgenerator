from dotenv import load_dotenv
load_dotenv()
from lib.model import Model, EMBEDDING_DIM, UNITS, vocab_to_char2idx_idx2char
import tensorflow as tf
tf.enable_eager_execution()
import time
import os

print("Starting up")

with open(os.getenv('VOCAB_FILE','./data/vocab.txt'), 'r') as text_file:
  vocab = text_file.read().split("\n")
  (char2idx, idx2char) = vocab_to_char2idx_idx2char(vocab)

  model = Model(len(char2idx), EMBEDDING_DIM, UNITS)

  checkpoint_dir = os.getenv('TRAINING_CHECKPOINT_DIR','./data/training_checkpoints')
  model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

  model.build(tf.TensorShape([1, None]))

  # Number of characters to generate
  num_generate = 1000

  # You can change the start string to experiment
  start_string = 'Q'

  # Converting our start string to numbers (vectorizing) 
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Evaluation loop.

  # # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a multinomial distribution to predict the word returned by the model
    predictions = predictions / temperature
    predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
    
    # We pass the predicted word as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)
    
    text_generated.append(idx2char[predicted_id])

  print (start_string + ''.join(text_generated))
