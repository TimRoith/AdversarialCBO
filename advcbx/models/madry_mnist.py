"""
This code is copied from:
https://github.com/max-andr/square-attack/tree/master/madry_cifar10

which helps wrapping the tensorflow model of the Madry MNIST challenge

The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch

import tensorflow.compat.v1 as tf
#import tensorflow as tf

tf.disable_eager_execution()




class MadryMNIST(object):
  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = self._weight_variable([5,5,1,32])
    b_conv1 = self._bias_variable([32])

    h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
    h_pool1 = self._max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = self._weight_variable([5,5,32,64])
    b_conv2 = self._bias_variable([64])

    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = self._max_pool_2x2(h_conv2)

    # first fully connected layer
    W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    b_fc1 = self._bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # output layer
    W_fc2 = self._weight_variable([1024,10])
    b_fc2 = self._bias_variable([10])

    self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent_per_point = y_xent
    self.xent = tf.reduce_sum(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')


model_path_dict = {'madry_mnist_robust': '../../weights/madry_models/adv_trained/',
                   }
model_class_dict = {'madry_mnist_robust': MadryMNIST,
                    }

class Model:
    def __init__(self, batch_size, gpu_memory):
        self.batch_size = batch_size
        self.gpu_memory = gpu_memory
        self.device     = 'cpu'

    def __call__(self, x):
        raise NotImplementedError('use ModelTF or ModelPT')

    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = utils.softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()
    
    def to(self, device):
        self.device = device
        return self
        
class ModelTF(Model):
    """
    Wrapper class around TensorFlow models.

    In order to incorporate a new model, one has to ensure that self.model has a TF variable `logits`,
    and that the preprocessing of the inputs is done correctly (e.g. subtracting the mean and dividing over the
    standard deviation).
    """
    def __init__(self, model_name, batch_size, gpu_memory):
        super().__init__(batch_size, gpu_memory)
        print("GPUs:", tf.config.list_physical_devices('GPU'))
        model_folder = model_path_dict[model_name]
        model_file = tf.train.latest_checkpoint(model_folder)
        self.model = model_class_dict[model_name]()
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_file = model_file
        print(model_file)
        if 'logits' not in self.model.__dict__:
            self.model.logits = self.model.pre_softmax

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        tf.train.Saver().restore(self.sess, model_file)

    def __call__(self, x):
        x = x.cpu()
        
        if 'mnist' in self.model_name:
            shape = self.model.x_input.shape[1:].as_list()
            x = np.reshape(x, [-1, *shape])
        elif 'cifar10' in self.model_name:
            x = np.transpose(x, axes=[0, 2, 3, 1])

        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []
        for i in range(n_batches):
            x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
            logits = self.sess.run(self.model.logits, feed_dict={self.model.x_input: x_batch})
            logits_list.append(logits)
        logits = np.vstack(logits_list)
        return torch.tensor(logits, device = self.device)