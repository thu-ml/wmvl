import argparse
import json
import os
import numpy as np
import tensorflow as tf
if float('.'.join(tf.__version__.split('.')[:-1])) > 1.12:
  import tensorflow.compat.v1 as tfv1
else:
  tfv1 = tf
tf.logging.set_verbosity(tfv1.logging.ERROR)
import experiments.utils
from experiments.slave import LogContext

from ais import ais
import models, optimizers
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


def log_lh_explicit(model, optimizer, n=10):

    z = model.q_z.sample(n)

    log_p_z = optimizer.p_z.log_prob(z)

    if model.distribution == 'normal':
        log_p_z = tf.reduce_sum(log_p_z, axis=-1)

    log_p_x_z = -tf.reduce_sum(optimizer.bce, axis=-1)

    log_q_z_x = model.q_z.log_prob(z)

    if model.distribution == 'normal':
        log_q_z_x = tf.reduce_sum(log_q_z_x, axis=-1)

    return tf.reduce_mean(tf.reduce_logsumexp(
        tf.transpose(log_p_x_z + log_p_z - log_q_z_x) - np.log(n), axis=-1))


def log_lh_ais(model, optimizer, method):

    def log_lh(x, z):
        logits = model._decoder(z)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits)
        return -tf.reduce_sum(bce, axis=-1)

    a = ais.AIS(
        model.x, log_lh, {'batch_size': tf.shape(model.x)[0], 'input_dim': model.z_dim},
        method=method)
    return a.ais(ais.get_schedule(1000))


def main(args):

    from mnist import load_fid
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('data/', one_hot=False,
                                      source_url='http://yann.lecun.com/exdb/mnist/')

    args_json = json.load(open(os.path.join(args.dir, 'hps.txt')))
    vars(args).update(args_json)
    ckpt_dir = tf.train.get_checkpoint_state(args.dir).model_checkpoint_path
    print('RESTORING MODEL FROM', ckpt_dir)

    # digit placeholder
    x = tf.placeholder(tf.float32, shape=(None, 784))
    
    z_dims = args.z_dims if args.latent == 'euc' else args.z_dims+1

    activation = {
        'elu': tf.nn.elu,
        'tanh': tf.nn.tanh
    }[args.ae_activation]

    # if args.model.startswith('wae'):
    #     args.observation = 'normal'

    if args.implicit:
        model = models.ImplicitAE(
            x=x, h_dim=args.h_dims, z_dim=z_dims, eps_dim=args.eps_dims,
            latent_space=args.latent, ae_activation=args.ae_activation,
            energy_activation=args.energy_activation,
            rescale_sph_latent=args.rescale_sph_latent,
            observation=args.observation) 
        optimizer = {
            'vae': optimizers.ImplicitVAE,
            'wae': optimizers.ImplicitWAE,
            'wae-gan': optimizers.GanWAE
        }[args.model](model, args)
    else:
        dist = {
            'euc': 'normal',
            'sph': 'vmf'
        }[args.latent]
        model = models.ExplicitAE(
            x=x, h_dim=args.h_dims, z_dim=z_dims, distribution=dist,
            rescale_sph_latent=args.rescale_sph_latent)
        optimizer = optimizers.ExplicitVAE(model, args)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=5)
    saver.restore(sess, ckpt_dir)

    test_images = mnist.test.images

    # compute_fid, _ = load_fid(test_images, args)
    
    ais_method = args.ais_method if args.latent == 'euc' else 'riem_ld'
    log_lh_sym = log_lh_ais(model, optimizer, ais_method)
    lsum = 0
    lden = 0

    idc = np.arange(test_images.shape[0])
    np.random.shuffle(idc)
    test_images = test_images[idc]
    with LogContext(test_images.shape[0]//args.batch_size) as ctx:
        for j in ctx:
            ti = test_images[j*args.batch_size: (j+1)*args.batch_size]
            ti = (ti > np.random.random(size=ti.shape)).astype(np.float32)
            lhs = sess.run(log_lh_sym, {model.x: ti})
            lsum += lhs.mean() * ti.shape[0]
            lden += ti.shape[0]
            ctx.log_scalars({'avg': lsum/lden}, ['avg'])

    print('AIS lb = {}'.format(lsum/lden))

    # else:
    #     import IPython; IPython.embed(); raise 1


if __name__ == '__main__':
    from mnist import parser
    parser.add_argument('-resume_from', default=-1, type=int)
    main(parser.parse_args())

