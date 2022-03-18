# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Run auditing on the FashionMNIST dataset."""

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized

from absl import app
from absl import flags

import audit

from approximate_minima_perturbation import ApproximateMinimaPerturbationLR, ApproximateMinimaPerturbationSVM


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


from datetime import datetime

#### FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_boolean(
    'op', False, 'If True, train with custom loss with objective perturbation. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('noise_std', 0.0, 'Standard deviation for noise')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 24, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 250, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model', 'lr', 'model to use, pick between lr and nn')
flags.DEFINE_string('attack_type', "clip_aware", 'clip_aware or backdoor')
flags.DEFINE_integer('pois_ct', 1, 'Number of poisoning points')
flags.DEFINE_integer('num_trials', 100, 'Number of trials for auditing')
flags.DEFINE_float('attack_l2_norm', 10, 'Size of poisoning data')
flags.DEFINE_float('alpha', 0.05, '1-confidence')
flags.DEFINE_boolean('load_weights', False,
                     'if True, use weights saved in init_weights.h5')
FLAGS = flags.FLAGS


def compute_epsilon(train_size):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / train_size
  steps = FLAGS.epochs * train_size / FLAGS.batch_size
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to approximate 1 / (number of training points).
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

def build_model(x, y):
  """Build a keras model."""
  input_shape = x.shape[1:]
  num_classes = y.shape[1]

  print("input_shape", input_shape, "num_classes", num_classes)
  l2 = 0
  if FLAGS.model == 'lr':
    print("Building lr model with input_shape: ", input_shape)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(l2))
        ])
  elif FLAGS.model == 'nn':
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu',
                              kernel_initializer='glorot_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(l2))
        ])
  else:
    raise NotImplementedError
  return model


def custom_loss(model, train_x):

  std_dev_obj = FLAGS.noise_std
# noise_obj = np.random.normal(scale=std_dev_obj, size=np.shape(model.trainable_weights[0]))
  noise_obj = np.random.normal(scale=std_dev_obj, size=np.shape(model.trainable_weights[0]))
  
  # print("shape of model.trainable_weights[0]: ", np.shape(model.trainable_weights[0]))
  # print("shape of model.trainable_weights[1]: ", np.shape(model.trainable_weights[1]))
  # print(model.trainable_weights[0])
  # print("noise_obj.shape", noise_obj.shape)
  # print("noise_obj, first few entries", noise_obj[:10])

  # print("train_x.shape", train_x.shape)

  def custom_loss_objective_perturbation(y_true, y_pred):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)

    loss_value = loss_fn(y_true, y_pred)
    # print("loss_value", loss_value)
    # loss_value += noise_obj * model.trainable_weights[0]
    # print("sum of (noise_obj * model.trainable_weights[0])", tf.math.reduce_sum(noise_obj * model.trainable_weights[0]))
    loss_value += tf.math.reduce_sum(noise_obj * model.trainable_weights[0]) / train_x.shape[0]

    # print("loss_value 2", loss_value)

    return tf.reduce_mean(loss_value, axis=-1)

  return custom_loss_objective_perturbation

def train_model(model, train_x, train_y, save_weights=False):
  """Train the model on given data."""
  print("Training model:", model.name)
  print("Using DPSGD: ", FLAGS.dpsgd)
  print(" save_weights: ", save_weights)
  print("Using load_weights: ", FLAGS.load_weights)

  # print("First entry in train_x", train_x[0])
  # print("shape of First entry in train_x", np.shape(train_x[0]))
  
  optimizer = dp_optimizer_vectorized.VectorizedDPSGD(
      l2_norm_clip=FLAGS.l2_norm_clip,
      noise_multiplier=FLAGS.noise_multiplier,
      num_microbatches=FLAGS.microbatches,
      learning_rate=FLAGS.learning_rate)

  if not FLAGS.dpsgd: # Use vanilla optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

  # define custom loss for objective perturbation
  if FLAGS.op:
    loss = custom_loss(model, train_x)
  else: 
    loss = tf.keras.losses.CategoricalCrossentropy(
      from_logits=True, reduction=tf.losses.Reduction.NONE)

  # Compile model with Keras
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  if save_weights:
    wts = model.get_weights()
    np.save('save_model', wts)
    model.set_weights(wts)
    return model

  if FLAGS.load_weights:  # load preset weights
    wts = np.load('save_model.npy', allow_pickle=True).tolist()
    model.set_weights(wts)

  # Train model with Keras
  print("fitting model...")
  model.fit(train_x, train_y,
            epochs=FLAGS.epochs,
            validation_data=(train_x, train_y),
            batch_size=FLAGS.batch_size,
            verbose=0)

  print("training model done...")
  return model


def membership_test(model, pois_x, pois_y):
  """Membership inference - detect poisoning."""

  pois_y = np.array([0, 1]) if pois_y == 1 else np.array([1, 0])

  res_1 = 1.0 / (1 + np.exp(pois_x.dot(model) * -1))
  res_2 = 1.0 / (1 + np.exp(np.zeros_like(pois_x).dot(model) * -1))

  probs = np.array([[1 - res_1, res_1], 
                    [1 - res_2, res_2]])

  res = np.multiply(probs[0, :] - probs[1, :], pois_y).sum()

  return res


def train_and_score(dataset):
  """Complete training run with membership inference score."""
  x, y, pois_x, pois_y, i, eps = dataset
  np.random.seed(i)
  tf.set_random_seed(i)
  tf.reset_default_graph()

  # model = build_model(x, y)
  # model = train_model(model, x, y)

  if len(x.shape) == 3:
    x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
  
  # Get y in range of {-1, 1}
  y_lr = np.argmax(y, axis=1) * 2 - 1 

  # print("y_lr:", y_lr[:10])
  # print("x,shape:", x.shape)
  # print("x:", x[:1])
  # print("pois_x: ", pois_x, "pois_x.shape", pois_x.shape)

  pois_y = 1 if pois_y[1] == 1 else -1

  model, _ = ApproximateMinimaPerturbationLR.run_classification(x, y_lr, epsilon=eps, delta=0.01, lambda_param=None)
  
  # Evaluate accuracy of model on training 
  # print("model:", model)
  # print("x.shape", x.shape)
  # print("model.shape", model.shape)
  # print("x:", x[0])
  n = x.dot(model) * -1
  # print("n:", n[:10])
  res = 1.0 / (1 + np.exp(x.dot(model) * -1))
  # print("res:", res[:10])
  res_one_hot = np.array([[0, 1] if y_i > 0.5 else [1, 0] for y_i in res])

  # print(res_one_hot[:10])
  # print(y[:10])
  accuracy = 1.0 - np.abs(y - res_one_hot).sum() / x.shape[0]

  print("Model accuracy on training set: ", accuracy)
  
  return membership_test(model, pois_x, pois_y)


def main(unused_argv):
  eps_vals = [0.1, 0.3, 1, 3, 10, 100]
  eps_vals = [ 10]
  
  output = open("output/auditing_output", "w")

  del unused_argv
  # Load training and test data.
  np.random.seed(0)

  (train_x, train_y), _ = tf.keras.datasets.fashion_mnist.load_data()
  train_inds = np.where(train_y < 2)[0]

  train_x = train_x[train_inds] / 255. - 0.5 # This was originally included, not sure why

  train_y = np.eye(2)[train_y[train_inds]]


  # subsample dataset
  ss_inds = np.random.choice(train_x.shape[0], train_x.shape[0]//2, replace=False)
  train_x = train_x[ss_inds]
  train_y = train_y[ss_inds]

  train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1] * train_x.shape[2]))

  output.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
  for analysis_eps in eps_vals: 
    auditor = audit.AuditAttack(train_x, train_y, train_and_score)

    thresh, _, _ = auditor.run(FLAGS.pois_ct, FLAGS.attack_type, FLAGS.num_trials,
                              alpha=FLAGS.alpha, threshold=None,
                              l2_norm=FLAGS.attack_l2_norm, eps=analysis_eps)

    _, eps, acc = auditor.run(FLAGS.pois_ct, FLAGS.attack_type, FLAGS.num_trials,
                              alpha=FLAGS.alpha, threshold=thresh,
                              l2_norm=FLAGS.attack_l2_norm, eps=analysis_eps)

    # epsilon_upper_bound = compute_epsilon(train_x.shape[0])

    print("Analysis epsilon is {}.".format(analysis_eps))
    print("At threshold={}, epsilon={}.".format(thresh, eps))
    print("The best accuracy at distinguishing poisoning is {}.".format(acc))

    output.write("Analysis epsilon is {}.\n".format(analysis_eps))
    output.write("At threshold={}, epsilon={}.\n".format(thresh, eps))
    output.write("The best accuracy at distinguishing poisoning is {}.\n".format(acc))

  output.close()
  # print("noise_std: ", FLAGS.noise_std)
if __name__ == '__main__':
  app.run(main)
