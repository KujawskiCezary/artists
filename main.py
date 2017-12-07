from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import random
import os
from PIL import Image
import time



tf.logging.set_verbosity(tf.logging.INFO)

TRAINING_DIR = os.getcwd() + '/training_data/'
TESTING_DIR = os.getcwd() + '/testing_data/'
AUTHORS = tuple(os.walk(TRAINING_DIR))[0][1]

def load_image(image, images_dir=TESTING_DIR, amount_of_segments=10):
    painting = np.asarray(Image.open(images_dir + image['author'] + '/' + image['name']))
    shape = painting.shape
    segments = []
    labels = []
    for _ in range(amount_of_segments):
        random_x = random.randint(0, shape[0] - (SIZE + 1))
        random_y = random.randint(0, shape[1] - (SIZE + 1))
        segment = painting[random_x:random_x+SIZE, random_y:random_y+SIZE]
        flatten_segment = np.concatenate(segment)
        segments.append(flatten_segment)
        labels.append(AUTHORS.index(image['author']))
    return np.asarray(segments, dtype=np.float32), np.asarray(labels, dtype=np.int32)
                 
def get_paintings(images_dir):
    return {author:
                     {painting : np.asarray(Image.open(images_dir + author + '/' + painting)) / 255
                      for painting in tuple(os.walk(images_dir + author))[0][2]}
                 for author in AUTHORS}
SIZE = 128
def create_set(iterations, images_dir):
    images = []
    labels = []
    for i in range(iterations):
        #print(i)
        paintings = get_paintings(images_dir)
        for author in paintings:
            for painting in paintings[author]:
                painting_arr = paintings[author][painting]
                shape = painting_arr.shape
                random_x = random.randint(0, shape[0] - (SIZE + 1))
                random_y = random.randint(0, shape[1] - (SIZE + 1))
                segment = painting_arr[random_x:random_x+SIZE, random_y:random_y+SIZE]
                flatten_segment = np.concatenate(segment)
                #flatten_segment = np.concatenate(segment).ravel()
                #flatten_segment = flatten_segment[::3]
                #print(len(flatten_segment), painting)
                #yield (np.array(segment, dtype=np.float32), np.asarray(AUTHORS.index(author), dtype=np.int32))
                images.append(flatten_segment)
                labels.append(AUTHORS.index(author))
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)
    
def test_set(iterations, images_dir, author_i):
    images = []
    labels = []
    for _ in range(iterations):
        
        paintings = get_paintings(images_dir)
        author = AUTHORS[author_i]
        for painting in paintings[author]:
            painting_arr = paintings[author][painting]
            shape = painting_arr.shape
            random_x = random.randint(0, shape[0] - (SIZE + 1))
            random_y = random.randint(0, shape[1] - (SIZE + 1))
            segment = painting_arr[random_x:random_x+SIZE, random_y:random_y+SIZE]
            flatten_segment = np.concatenate(segment)
            #flatten_segment = flatten_segment[::3]
            #print(len(flatten_segment), painting)
            images.append(flatten_segment)
            labels.append(AUTHORS.index(author))
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)
    
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, SIZE, SIZE, 3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, SIZE * SIZE * 4])
  #pool2_flat = tf.reshape(pool2, [-1, 256])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  print(onehot_labels.get_shape())
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  # Load training and eval data
  train_data, train_labels = create_set(2, images_dir=TRAINING_DIR)
  #print(len(train_data), len(train_labels))
  eval_data, eval_labels = create_set(2, images_dir=TESTING_DIR)
  #print(len(eval_data), len(eval_labels))  
  # Create the Estimator
  classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="tmp/convnet_model")
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
   
  # Train the model
 
  start = time.time()

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=10,
    num_epochs=None,
    shuffle=True)
  classifier.train(
    input_fn=train_input_fn,
    steps=2000,
   hooks=[logging_hook])
    
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  end = time.time()
  print('Training time is {}s'.format(end - start))
    
    
    
if __name__ == "__main__":
  tf.app.run()