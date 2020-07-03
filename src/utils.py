import tensorflow as tf
import numpy as np


def add_bool_flag(parser, name):
  parser.add_argument('-'+name, action='store_true', dest=name)
  parser.add_argument('-no_'+name, action='store_false', dest=name)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable(
        'u', [1, w_shape[-1]],
        initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def sn_dense(inp, units, activation, name):
    last_dim = int(inp.shape[-1])
    with tf.name_scope(name):
        w = tf.get_variable(name='kernel', shape=[last_dim, num_units],
                            initializer='glorot_uniform')
        b = tf.get_variable(name='bias', shape=[num_units],
                            initializer=tf.zeros_initializer())
        w1 = spectral_norm(w)
    return inp @ w1 + b


def tile_images(imgs):
    z = int(imgs.shape[0] ** 0.5)
    assert z*z == imgs.shape[0]
    imgs = imgs.reshape([z,z]+list(imgs.shape[1:]))
    return np.concatenate(np.concatenate(imgs, axis=1), axis=1)


def save_image(path, imgs):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    if len(imgs.shape) == 2:
        plt.imsave(path, imgs, cmap='gray')
    else:
        plt.imsave(path, imgs)

