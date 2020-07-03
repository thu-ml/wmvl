import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import dtypes


def load_cifar10(test):
    from tensorflow.contrib.learn.python.learn.datasets import base
    source_url = 'http://ml.cs.tsinghua.edu.cn/~ziyu/static/cifar10/'
    train_dir = 'data/cifar10'
    TRAIN_IMAGES = 'train.npy'
    TEST_IMAGES = 'test.npy'
    lc = base.maybe_download(
        TRAIN_IMAGES, train_dir, source_url + TRAIN_IMAGES)
    train_images = np.load(lc)
    lc = base.maybe_download(
        TEST_IMAGES, train_dir, source_url + TEST_IMAGES)
    test_images = np.load(lc)
    if not test:
      validation_size = 5000
      np.random.seed(23)
      idcs = np.arange(train_images.shape[0])
      np.random.shuffle(idcs)
      train_images = train_images[idcs]
      validation_images = train_images[-validation_size:]
      train_images = train_images[:-validation_size]
    else:
      validation_images = train_images[:5000]
    train = DataSet(train_images)
    validation = DataSet(validation_images)
    test = DataSet(test_images)
    return base.Datasets(train=train, validation=validation, test=test)


def load_data(dname, test):
  if dname == 'cifar10':
    return load_cifar10(test)

  from tensorflow.examples.tutorials.mnist import input_data
  url = {
    'mnist': 'http://yann.lecun.com/exdb/mnist/',
    'fashion': 'http://ml.cs.tsinghua.edu.cn/~ziyu/static/fashion/',
  }[dname]
  dsets = input_data.read_data_sets(
    'data/'+dname, one_hot=False, source_url=url,
    validation_size=(0 if test else 5000))
  dsets.train._images = dsets.train.images.reshape((-1, 28, 28, 1))
  dsets.validation._images = dsets.validation.images.reshape((-1, 28, 28, 1))
  dsets.test._images = dsets.test.images.reshape((-1, 28, 28, 1))
  return dsets


class DataSet(object):
  """Container class for a dataset. From tf.examples.tutorials.mnist
  """

  def __init__(self,
               images,
               one_hot=False,
               dtype=dtypes.float32,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError(
          'Invalid image dtype %r, expected uint8 or float32' % dtype)
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 128.0) - 1.
    self._images = images
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      ret = np.concatenate(
          (images_rest_part, images_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      ret = self._images[start:end]
    ret = ret + (np.random.uniform(size=ret.shape) - 0.5) / 128.
    return ret

