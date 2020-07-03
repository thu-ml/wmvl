import numpy as np
import tensorflow as tf
if float('.'.join(tf.__version__.split('.')[:-1])) > 1.12:
  from tensorflow.compat.v1 import AUTO_REUSE
else:
  from tensorflow import AUTO_REUSE

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from utils import *


ENCODER = 'encoder'
DECODER = 'decoder'
COND_ENERGY = 'cond_energy'
MARGINAL_ENERGY = 'mar_energy'


class ExplicitAE(object):

    def __init__(self, x, h_dim, z_dim, activation=tf.nn.relu, distribution='normal',
                 rescale_sph_latent=False):
        """
        :param x: placeholder for input
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        self.x, self.h_dim, self.z_dim, self.activation, self.distribution = x, h_dim, z_dim, \
            activation, distribution
        self.rescale_sph_latent = rescale_sph_latent

        self.z_mean, self.z_var = self._encoder(self.x)

        if distribution == 'normal':
            self.q_z = tf.distributions.Normal(self.z_mean, self.z_var)
        elif distribution == 'vmf':
            self.q_z = VonMisesFisher(self.z_mean, self.z_var)
        else:
            raise NotImplemented

        self.z = self.q_z.sample()

        self.logits = self._decoder(self.z)

    def _encoder(self, x):
        """
        Encoder network

        :param x: placeholder for input
        :return: tuple `(z_mean, z_var)` with mean and concentration around the mean
        """
        
        with tf.variable_scope(ENCODER, reuse=AUTO_REUSE):
            # 2 hidden layers encoder
            h0 = tf.layers.dense(x, units=self.h_dim, activation=self.activation)
            h1 = tf.layers.dense(h0, units=self.h_dim, activation=self.activation)

            if self.distribution == 'normal':
                # compute mean and std of the normal distribution
                z_mean = tf.layers.dense(h1, units=self.z_dim, activation=None)
                z_var = tf.layers.dense(h1, units=self.z_dim, activation=tf.nn.softplus)
            elif self.distribution == 'vmf':
                # compute mean and concentration of the von Mises-Fisher
                z_mean = tf.layers.dense(h1, units=self.z_dim,
                                         activation=lambda x: tf.nn.l2_normalize(x, axis=-1))
                # the `+ 1` prevent collapsing behaviors
                z_var = tf.layers.dense(h1, units=1, activation=tf.nn.softplus) + 1
            else:
                raise NotImplemented

            return z_mean, z_var

    def _decoder(self, z):
        """
        Decoder network

        :param z: tensor, latent representation of input (x)
        :return: logits, `reconstruction = sigmoid(logits)`
        """
        # 2 hidden layers decoder
        if self.distribution == 'vmf' and self.rescale_sph_latent:
            z = z * tf.sqrt(tf.to_float(self.z_dim))
        with tf.variable_scope(DECODER, reuse=AUTO_REUSE):
            h2 = tf.layers.dense(z, units=self.h_dim, activation=self.activation)
            h2 = tf.layers.dense(h2, units=self.h_dim, activation=self.activation)
            logits = tf.layers.dense(h2, units=self.x.shape[-1], activation=None)

        return logits


class ImplicitAE(object):

    def __init__(self, x, h_dim, z_dim, eps_dim, latent_space,
                 spec_norm=False, ae_activation=tf.nn.tanh,
                 energy_activation=tf.nn.elu, rescale_sph_latent=False,
                 observation=None, gan=False, is_training_ph=None):
        self.x, self.h_dim, self.z_dim = x, h_dim, z_dim
        self.spec_norm = spec_norm
        self.eps_dim = eps_dim
        self.latent = latent_space
        self.ae_activation = ae_activation
        self.energy_activation = energy_activation
        self.rescale_sph_latent = rescale_sph_latent
        self.is_gan = gan
        self.z = self._encoder(self.x)
        self.logits = self._decoder(self.z)
        if observation == 'normal':
            self.qxz_mean = tf.nn.sigmoid(self.logits)

    def _encoder(self, x):
        """
        Encoder network

        :param x: placeholder for input
        :return: samples 
        """
        # 2 hidden layers encoder
        eshape = tf.concat([tf.shape(x)[:-1], [self.eps_dim]], axis=0)
        eps = tf.random_normal(shape=eshape)
        inp = tf.concat([x, eps], axis=-1)
        with tf.variable_scope(ENCODER, reuse=AUTO_REUSE):
            h0 = tf.layers.dense(inp, units=self.h_dim, activation=self.ae_activation)
            h1 = tf.layers.dense(h0, units=self.h_dim, activation=self.ae_activation)
            z_samples = tf.layers.dense(h1, units=self.z_dim)
            if self.latent == 'sph':
                z_samples = tf.nn.l2_normalize(z_samples, axis=-1)
        return z_samples

    def _rescale(self, z):
        if self.latent == 'sph' and self.rescale_sph_latent:
            return z * tf.sqrt(tf.to_float(self.z_dim))
        return z

    def _decoder(self, z):
        """
        Decoder network

        :param z: tensor, latent representation of input (x)
        :return: logits, `reconstruction = sigmoid(logits)`
        """
        z = self._rescale(z)
        # 2 hidden layers decoder
        with tf.variable_scope(DECODER, reuse=AUTO_REUSE):
            h2 = tf.layers.dense(z, units=self.h_dim, activation=self.ae_activation)
            h2 = tf.layers.dense(h2, units=self.h_dim, activation=self.ae_activation)
            logits = tf.layers.dense(h2, units=self.x.shape[-1], activation=None)
        return logits

    def _cond_energy(self, z, x):
        """
        :return: approximate energy -log \\tilde{q}(z|x)
        """
        z = self._rescale(z)
        if self.spec_norm:
            dense = sn_dense
        else:
            dense = tf.layers.dense
        inp = tf.concat([z, x], axis=-1)
        with tf.variable_scope(COND_ENERGY, reuse=AUTO_REUSE):
            h0 = dense(inp, units=self.h_dim, activation=self.energy_activation, name='fc1')
            h1 = dense(h0, units=self.h_dim, activation=self.energy_activation, name='fc2')
            h2 = tf.layers.dense(h1, units=self.z_dim, name='fc3')
            ret = tf.reduce_sum(h2 * z, axis=-1, keepdims=True)
        return ret

    def _marginal_energy(self, z):
        """
        :return: approximate energy -log \\tilde{q}(z|x)
        """
        z = self._rescale(z)
        if self.spec_norm:
            dense = sn_dense
        else:
            dense = tf.layers.dense
        inp = z
        with tf.variable_scope(MARGINAL_ENERGY, reuse=AUTO_REUSE):
            h0 = dense(inp, units=self.h_dim, activation=self.energy_activation, name='fc1')
            h1 = dense(h0, units=self.h_dim, activation=self.energy_activation, name='fc2')
            if self.is_gan:
                ret = tf.layers.dense(h1, units=1, name='fc3')
            else:
                h2 = tf.layers.dense(h1, units=self.z_dim, name='fc3')
                ret = tf.reduce_sum(h2 * z, axis=-1, keepdims=True)
        return ret


class LargeImplicitAE(object):

    def __init__(self, x, h_dim, z_dim, eps_dim, latent_space,
                 spec_norm=False, n_filters=64, gan=False):
        self.x, self.h_dim, self.z_dim = x, h_dim, z_dim
        self.eps_dim = eps_dim
        self.latent = latent_space
        self.n_filters = n_filters
        self.is_gan = gan
        self.z = self._encoder(self.x)
        self.qxz_mean = self._decoder(self.z)

    def _encoder(self, x):
        """
        Encoder network

        :param x: placeholder for input
        :return: samples 
        """
        assert x.shape.ndims == 4
        W = int(self.x.shape[-2])
        assert W in [64, 32]
        LL = [1,2,4,8] if W == 64 else [1,2,4]
        with tf.variable_scope(ENCODER, reuse=AUTO_REUSE):
            h = x
            for j in LL:
                h = tf.pad(h, [[0,0], [2,2], [2,2], [0,0]], 'CONSTANT')
                h = tf.layers.conv2d(h, self.n_filters * j, kernel_size=5, strides=2,
                                     activation=tf.nn.relu)
                print('ENC layer', j, h.shape[1:].as_list())
            h0 = tf.layers.flatten(h)
            h0 = tf.layers.dense(h0, 512, activation=tf.nn.relu)
            z_out = tf.layers.dense(h0, self.z_dim)
            if self.latent == 'sph':
                z_out = tf.nn.l2_normalize(z_out, axis=-1)
            print(z_out.get_shape().as_list())
        return z_out

    def _rescale(self, z):
        if self.latent == 'sph':
            return z * tf.sqrt(tf.to_float(self.z_dim))
        return z

    def _decoder(self, z):
        """
        Decoder network

        :param z: tensor, latent representation of input (x)
        :return: logits, `reconstruction = sigmoid(logits)`
        """
        z = self._rescale(z)
        W = int(self.x.shape[-2])
        assert z.shape.ndims == 2
        with tf.variable_scope(DECODER, reuse=AUTO_REUSE):
            LL = [8, 4, 2, 1] if W == 64 else [4, 2, 1]
            h = tf.layers.dense(z, self.n_filters * LL[0] * 4 * 4, activation=tf.nn.relu)
            h = tf.reshape(h, [-1, 4, 4, self.n_filters * LL[0]])
            for j in LL[1:]:
                h = tf.layers.conv2d_transpose(h, self.n_filters*j, 5, 2, activation=tf.nn.relu)
                h = h[:, 1:-2, 1:-2, :]
                print('DEC layer', j, h.shape[1:].as_list())
            h = tf.layers.conv2d_transpose(h, 3, 5, 2, activation=tf.tanh)
            h = h[:, 1:-2, 1:-2, :]
            print('DEC OUT', j, h.shape[1:].as_list())
        return h

    def _cond_energy(self, z, x):
        raise NotImplemented()

    def _marginal_energy(self, z):
        z = self._rescale(z)
        W = int(self.x.shape[-2])
        LL = [1,2,4,8] if W == 64 else [1,2,4]
        with tf.variable_scope(MARGINAL_ENERGY, reuse=AUTO_REUSE):
            h = tf.layers.dense(z, W**2, activation=tf.nn.softplus)
            h = tf.reshape(h, [tf.shape(h)[0], W, W, 1])
            for j in LL:
                h = tf.pad(h, [[0,0], [2,2], [2,2], [0,0]], 'CONSTANT')
                h = tf.layers.conv2d(h, self.n_filters * j, kernel_size=5, strides=2,
                                     activation=tf.nn.softplus)
                # print('ENERGY layer', j, h.shape[1:].as_list())
            h0 = tf.layers.flatten(h)
            h0 = tf.layers.dense(h0, 512, activation=tf.nn.softplus)
            if self.is_gan:
                ret = tf.layers.dense(h0, 1)
            else:
                z_out = tf.layers.dense(h0, self.z_dim)
                ret = tf.reduce_sum(z_out * z, axis=-1, keepdims=True)
        return ret

