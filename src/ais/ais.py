"""
On an implicit VAE with 8 latents and the first 16 images of the test set,
    AIS/HMC produces a log lh of 93.{5,7,8,4}
    AIS/LD  produces a log lh of 93.{9,7},  92.1, 93.7
in three runs.
"""

import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np
from collections import namedtuple
from ais.hmc import *
from ais.ld import *
from hyperspherical_vae.distributions import HypersphericalUniform

def get_schedule(num, rad=4):
    if num == 1:
        return np.array([0.0, 1.0])
    t = np.linspace(-rad, rad, num)
    s = 1.0 / (1.0 + np.exp(-t))
    return (s - np.min(s)) / (np.max(s) - np.min(s))

def log_mean_exp(x, axis=None):
    m = tf.reduce_max(x, axis=axis, keep_dims=True)
    return m + tf.log(tf.reduce_mean(tf.exp(x - m), axis=axis, keep_dims=True))


McConfig = namedtuple('McConfig',
                      'stepsize n_steps target_acceptance_rate '
                      'avg_acceptance_slowness stepsize_min stepsize_max '
                      'stepsize_dec stepsize_inc')

default_config = {
    'hmc': McConfig(stepsize=0.01, n_steps=10, target_acceptance_rate=0.65,
                    avg_acceptance_slowness=0.9, stepsize_min=0.0001,
                    stepsize_max=0.5, stepsize_dec=0.98, stepsize_inc=1.02),
    'ld':  McConfig(stepsize=0.02, n_steps=5, target_acceptance_rate=0.65,
                    avg_acceptance_slowness=0.9, stepsize_max=0.2, stepsize_min=0.0001,
                    stepsize_dec=0.98, stepsize_inc=1.02)
}
default_config['riem_ld'] = default_config['ld']
default_config['riem_euc_ld'] = default_config['ld']

METHOD_CHOICES = list(default_config)

class AIS(object):
    def __init__(self, x_ph, log_likelihood_fn, dims, num_samples=16,
                 method='hmc', config=None):
        """
        The model implements Hamiltonian AIS.
        Developed by @bilginhalil on top of https://github.com/jiamings/ais/

        Example use case:
        logp(x|z) = |integrate over z|{logp(x|z,theta) + logp(z)}
        p(x|z, theta) -> likelihood function p(z) -> prior
        Prior is assumed to be a normal distribution with mean 0 and identity covariance matrix

        :param x_ph: Placeholder for x
        :param log_likelihood_fn: Outputs the logp(x|z, theta), it should take two parameters: x and z
        :param e.g. {'output_dim': 28*28, 'input_dim': FLAGS.d, 'batch_size': 1} :)
        :param num_samples: Number of samples to sample from in order to estimate the likelihood.

        The following are parameters for HMC.
        :param stepsize:
        :param n_steps:
        :param target_acceptance_rate:
        :param avg_acceptance_slowness:
        :param stepsize_min:
        :param stepsize_max:
        :param stepsize_dec:
        :param stepsize_inc:
        """

        self.dims = dims
        self.log_likelihood_fn = log_likelihood_fn
        self.num_samples = num_samples

        self.z_shape = [dims['batch_size'] * self.num_samples, dims['input_dim']]

        if method != 'riem_ld':
            self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.z_shape),
                                                    scale_diag=tf.ones(self.z_shape))
        else:
            self.prior = HypersphericalUniform(dims['input_dim']-1)

        self.batch_size = dims['batch_size']
        self.x = tf.tile(x_ph, [self.num_samples, 1])

        self.method = method
        self.config = config if config is not None else default_config[method]

    def log_f_i(self, z, t):
        return tf.reshape(- self.energy_fn(z, t), [self.num_samples, self.batch_size])

    def energy_fn(self, z, t):
        e = self.prior.log_prob(z)
        assert e.shape.ndims == 1
        e += t * tf.reshape(self.log_likelihood_fn(self.x, z),
                            [self.num_samples * self.batch_size])
        assert e.shape.ndims == 1
        return -e

    def ais(self, schedule):
        """
            :param schedule: temperature schedule i.e. `p(z)p(x|z)^t`
            :return: [batch_size]
        """
        cfg = self.config
        if isinstance(self.prior, tfd.MultivariateNormalDiag):
            z = self.prior.sample()
        else:
            z = self.prior.sample([self.num_samples * self.batch_size])
        assert z.shape.ndims == 2

        index_summation = (tf.constant(0),
                           tf.zeros([self.num_samples, self.batch_size]),
                           tf.cast(z, tf.float32),
                           cfg.stepsize,
                           cfg.target_acceptance_rate)

        items = tf.unstack(tf.convert_to_tensor(
            [[i, t0, t1] for i, (t0, t1) in enumerate(zip(schedule[:-1], schedule[1:]))]))

        def condition(index, summation, z, stepsize, avg_acceptance_rate):
            return tf.less(index, len(schedule)-1)

        def body(index, w, z, stepsize, avg_acceptance_rate):
            item = tf.gather(items, index)
            t0 = tf.gather(item, 1)
            t1 = tf.gather(item, 2)

            new_u = self.log_f_i(z, t1)
            prev_u = self.log_f_i(z, t0)
            w = tf.add(w, new_u - prev_u)

            def run_energy(z):
                e = self.energy_fn(z, t1)
                if self.method != 'hmc':
                    e = e[:, None]
                with tf.control_dependencies([e]):
                    return e

            # New step:
            if self.method == 'hmc':
                accept, final_pos, final_vel = hmc_move(
                    z, run_energy, stepsize, cfg.n_steps)
                new_z, new_stepsize, new_acceptance_rate = hmc_updates(
                    z,
                    stepsize,
                    avg_acceptance_rate=avg_acceptance_rate,
                    final_pos=final_pos,
                    accept=accept,
                    stepsize_min=cfg.stepsize_min,
                    stepsize_max=cfg.stepsize_max,
                    stepsize_dec=cfg.stepsize_dec,
                    stepsize_inc=cfg.stepsize_inc,
                    target_acceptance_rate=cfg.target_acceptance_rate,
                    avg_acceptance_slowness=cfg.avg_acceptance_slowness
                )

            elif self.method.endswith('ld'):
                new_z, cur_acc_rate = ld_move(
                    z, run_energy, stepsize, cfg.n_steps, self.method)
                new_stepsize, new_acceptance_rate = ld_update(
                    stepsize,
                    cur_acc_rate=cur_acc_rate,
                    hist_acc_rate=avg_acceptance_rate,
                    target_acc_rate=cfg.target_acceptance_rate,
                    ssz_inc=cfg.stepsize_inc,
                    ssz_dec=cfg.stepsize_dec,
                    ssz_min=cfg.stepsize_min,
                    ssz_max=cfg.stepsize_max,
                    avg_acc_decay=cfg.avg_acceptance_slowness)

            return tf.add(index,1), w, new_z, new_stepsize, new_acceptance_rate

        i, w, _, final_stepsize, final_acc_rate = tf.while_loop(
            condition, body, index_summation, parallel_iterations=1, swap_memory=True)
        # w = tf.Print(w, [final_stepsize, final_acc_rate], 'ff')
        return tf.squeeze(log_mean_exp(w, axis=0), axis=0)

