from dotenv import load_dotenv
load_dotenv()
from lib.model import init_model
import tensorflow as tf
tf.enable_eager_execution()
import time
import os


(model, dataset, optimizer) = init_model()

# Training step
EPOCHS = 30

# Directory where the checkpoints will be saved
checkpoint_dir = os.getenv('TRAINING_CHECKPOINT_DIR','./training_checkpoints')
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

model.save_weights(os.getenv('MODEL_WEIGHTS_FILE','./model_weights.h5'))

with open(os.getenv('MODEL_ARCH_FILE','model_architecture.json'), 'w') as f:
  f.write(model.to_json())