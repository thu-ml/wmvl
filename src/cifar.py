import os
import json
import argparse
import numpy as np
from tqdm import trange
from tensorpack.dataflow import *
import experiments.utils
from experiments.slave import LogContext

import tensorflow as tf
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import models, optimizers, datasets
from mnist import parser
from utils import *
import fid
from mnist import parser


parser.add_argument('-plot_every', default=5000, type=int)
parser.add_argument('-plot_only', action='store_true')
parser.set_defaults(model='wae', z_dims=32, eps_dims=0, observation='normal', latent='sph',
                    batch_size=128, lr=1e-4, grad_penalty=1e-4, wae_lambda=10.,
                    optimizer='rmsprop', n_iter=200001, save_every=5000, plot_only=False)


CIFAR_STATS_PATH = os.path.expanduser('~/inception/cifar.npz')
INCEPTION_PATH = os.path.expanduser('~/inception')


def build_graph(args, im_size=64):
    x_ph = tf.placeholder(tf.float32, shape=(None, im_size, im_size, 3), name='x')
    is_training_ph = tf.placeholder(tf.bool, shape=(), name='is_training')
    z_dims = args.z_dims if args.latent == 'euc' else args.z_dims+1
    model = models.LargeImplicitAE(
        x=x_ph, h_dim=args.h_dims, z_dim=z_dims, eps_dim=args.eps_dims,
        latent_space=args.latent, n_filters=64, gan=(args.model=='wae-gan'))
    optimizer = {
        'wae': optimizers.ImplicitWAE,
        'wae-gan': optimizers.GanWAE,
        'wae-mmd': optimizers.MMDWAE,
    }[args.model](model, args)
    batch_size_sym = tf.placeholder(tf.int32, [])

    if args.latent == 'euc':
        z_sample_sym = tf.random_normal(shape=[batch_size_sym, args.z_dims], dtype=tf.float32)
    else:
        z_sample_sym = optimizer.dist_pz.sample([batch_size_sym])

    x_sample_sym = model._decoder(z_sample_sym)
    return x_ph, is_training_ph, model, optimizer, batch_size_sym, z_sample_sym, x_sample_sym


def load_fid(args):
    act_stats = np.load(CIFAR_STATS_PATH)
    mu0, sig0 = act_stats['mu'], act_stats['sigma']
    inception_path = fid.check_or_download_inception(INCEPTION_PATH)
    inception_graph = tf.Graph()
    with inception_graph.as_default():
        fid.create_inception_graph(str(inception_path))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    inception_sess = tf.Session(config=config, graph=inception_graph)
    def compute(images):
        m, s = fid.calculate_activation_statistics(
            np.array(images), inception_sess, args.batch_size, verbose=True)
        return fid.calculate_frechet_distance(m, s, mu0, sig0)

    return compute, locals()


def train(args):
    ds = datasets.load_cifar10(test=True)
    compute_fid, _ = load_fid(args)
    tf.set_random_seed(args.seed)
    x_ph, is_training_ph, model, optimizer, batch_size_sym, z_sample_sym, x_sample_sym = \
        build_graph(args, im_size=32)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.compat.v1.train.Saver(keep_checkpoint_every_n_hours=3, max_to_keep=6)
    print(args)
    LOG_EVERY = 10
    sess.run(tf.global_variables_initializer())
    with LogContext(args.n_iter // LOG_EVERY, logdir=args.dir, tfsummary=True) as ctx:
        for i in ctx:
            for j in range(LOG_EVERY):
                x_mb = ds.train.next_batch(args.batch_size)
                optimizer.step(sess, {x_ph: x_mb, is_training_ph: True})
    
            to_log = sess.run({**optimizer.print}, {x_ph: x_mb, is_training_ph: False})
            ctx.log_scalars(to_log, list(to_log))

            if (i * LOG_EVERY) % args.plot_every == 0:
                x_samples = sess.run(x_sample_sym, {batch_size_sym: 16, is_training_ph: False})
                x_samples = ((np.clip(x_samples, -1, 1) + 1) / 2 * 255).astype(np.uint8)
                ctx.log_image('generated', tile_images(x_samples))

                images = []
                print('Computing FID')
                for j in trange(100):
                    x_samples = sess.run(x_sample_sym, {batch_size_sym: 100, is_training_ph: False})
                    x_samples = (np.clip(x_samples, -1, 1) + 1) / 2 * 256
                    images.extend(x_samples)
                fscore = compute_fid(images)
                print('FID =', fscore)
                ctx.log_scalars({'fid': fscore}, ['fid'])

            if (i * LOG_EVERY) % args.save_every == 0 and args.save_every > 0:
                saver.save(sess, os.path.join(args.dir, 'model'), global_step=i)
                print('Model saved')

    x_samples = sess.run(x_sample_sym, {batch_size_sym: 16, is_training_ph: False})
    x_samples = ((np.clip(x_samples, -1, 1) + 1) / 2 * 255).astype(np.uint8)
    ctx.log_image('generated', tile_images(x_samples))
    saver.save(sess, os.path.join(args.dir, 'model'), global_step=i)


def generate(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if not os.path.exists(CIFAR_STATS_PATH):
        print('Generating FID statistics for test set...')
        print('Building Inception graph')
        with tf.Session(config=config) as sess:
            inception_path = fid.check_or_download_inception(INCEPTION_PATH)
            fid.create_inception_graph(str(inception_path))
            ds = datasets.load_cifar10(True)
            all_test_set = (ds.test.images + 1) * 128
            print(all_test_set.shape)
            m, s = fid.calculate_activation_statistics(
                all_test_set, sess, args.batch_size, verbose=True)
        np.savez(CIFAR_STATS_PATH, mu=m, sigma=s)
        print('Done')

    root_dir = os.path.dirname(args.dir)
    args_json = json.load(open(os.path.join(root_dir, 'hps.txt')))
    ckpt_dir = args.dir
    vars(args).update(args_json)

    model_graph = tf.Graph()
    with model_graph.as_default():
        x_ph, is_training_ph, model, optimizer, batch_size_sym, z_sample_sym, x_sample_sym = build_graph(args)
        saver = tf.compat.v1.train.Saver(keep_checkpoint_every_n_hours=3, max_to_keep=6)

    model_sess = tf.Session(config=config, graph=model_graph)
    print('RESTORING MODEL FROM', ckpt_dir)
    saver.restore(model_sess, ckpt_dir)
    compute_fid, _ = load_fid(args)
    images = []
    for j in range(100):
        x_samples = model_sess.run(x_sample_sym, {batch_size_sym: 100, is_training_ph: False})
        x_samples = (np.clip(x_samples, -1, 1) + 1) / 2 * 256
        images.extend(x_samples)

    fscore = compute_fid(images)
    print('FID score = {}'.format(fscore))
    
    dest = os.path.join(root_dir, 'generated')
    if not os.path.exists(dest):
        os.makedirs(dest)
    for j, im in enumerate(images):
        plt.imsave(os.path.join(dest, '{}.png'.format(j)), im/256)
    

if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    if not args.plot_only:
        experiments.utils.preflight(args)
        train(args)
    else:
        generate(args)

