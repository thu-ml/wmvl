import os
import argparse
import numpy as np
import tensorflow as tf
if float('.'.join(tf.__version__.split('.')[:-1])) > 1.12:
  import tensorflow.compat.v1 as tfv1
else:
  tfv1 = tf
tf.logging.set_verbosity(tfv1.logging.ERROR)
import experiments.utils
from experiments.slave import LogContext

import models, optimizers
from test_ais import *
from utils import tile_images, save_image, add_bool_flag


parser = experiments.utils.parser('svae')
parser.add_argument('-seed', type=int, default=1234)
parser.add_argument('-implicit', action='store_true')
parser.add_argument('-explicit', action='store_false', dest='implicit')
add_bool_flag(parser, 'spec_norm')
parser.add_argument('-model', type=str, choices=['wae', 'wae-gan', 'wae-mmd', 'vae'],
                    default='vae')
parser.add_argument('-latent', type=str, choices=['euc', 'sph'], default='sph')
parser.add_argument('-z_dims', type=int, default=8)
parser.add_argument('-h_dims', type=int, default=256)
parser.add_argument('-batch_size', type=int, default=128)
# For WAEs, set eps_dims to 0 (as in all previous work)
parser.add_argument('-eps_dims', type=int, default=8)
parser.add_argument('-n_iter', type=int, default=100000)
parser.add_argument('-n_particles', type=int, default=1)
parser.add_argument('-optimizer', type=str, default='rmsprop',
                    choices=['adam', 'rmsprop'])
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-mpf_lr', type=float, default=1e-3)
parser.add_argument('-observation', type=str, default='sigmoid')
parser.add_argument('-save_every', type=int, default=2000)
parser.add_argument('-ais_method', type=str, default='hmc', choices=ais.METHOD_CHOICES)
parser.add_argument('--ae_activation',     '-a_act', type=str, default='tanh')
parser.add_argument('--energy_activation', '-e_act', type=str, default='tanh')
# This does not correspond to grad penalty. The regularizer is actually L2
parser.add_argument('--grad_penalty',      '-gp', type=float, default=1e-5)
# for WAE-MMD, wae_lam will be multiplied by 100 following previous work
parser.add_argument('--wae_lambda',        '-wae_lam', type=float, default=10.)
parser.add_argument('-test', action='store_true')
parser.add_argument('-do_fid', action='store_true', default=False)
# The original hyperspherical VAE implementation does not rescale the latents to have L2
# norm ~ O(sqrt(d)). Here we scale it to match the Euclidean case. However there is no
# significant difference in performance.
add_bool_flag(parser, 'rescale_sph_latent')
parser.add_argument('-train_score_dupl', type=int, default=1)
parser.add_argument('-mpf_method', type=str, default='ld', choices=['ld', 'spos'])
parser.add_argument('-mpf_spos_alpha', type=float, default=0.5)
parser.set_defaults(implicit=True, spec_norm=False, rescale_sph_latent=True, test=False)


def log_likelihood(model, optimizer, n=10):
    """

    :param model: model object
    :param optimizer: optimizer object
    :param n: number of MC samples
    :return: MC estimate of log-likelihood
    """
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


def load_fid(mnist_test_images, args, binarize=True):
    import fid
    def transform_for_fid(im):
        assert len(im.shape) == 2 and im.dtype == np.float32
        if binarize:
            im = (im > np.random.random(size=im.shape)).astype(np.float32)
        a = np.array(im) - 0.5
        return a.reshape((-1, 28, 28, 1))
    inception_path = os.path.expanduser('~/lenet/savedmodel')
    inception_graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    inception_sess = tf.Session(config=config, graph=inception_graph)
    with inception_graph.as_default():
        tf.saved_model.loader.load(
            inception_sess, [tf.saved_model.tag_constants.TRAINING], inception_path)
    mu0, sig0 = fid.calculate_activation_statistics(
        transform_for_fid(mnist_test_images), inception_sess, args.batch_size,
        verbose=True, model='lenet')
    def compute(images):
        m, s = fid.calculate_activation_statistics(
            transform_for_fid(images), inception_sess, args.batch_size, verbose=True,
            model='lenet')
        return fid.calculate_frechet_distance(m, s, mu0, sig0)
    return compute, locals()


def main(args):

    from tensorflow.examples.tutorials.mnist import input_data
    if not args.test:
        mnist = input_data.read_data_sets(
            'data/', one_hot=False, source_url='http://yann.lecun.com/exdb/mnist/')
        val_images = test_images = mnist.validation.images
    else:
        mnist = input_data.read_data_sets(
            'data/', one_hot=False, source_url='http://yann.lecun.com/exdb/mnist/',
            validation_size=0)
        val_images = mnist.test.images

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.do_fid:
        compute_fid, _fid_lc = load_fid(
            mnist.test.images, args, binarize=(args.observation!='normal'))

    # digit placeholder
    x_ph = tf.placeholder(tf.float32, shape=(None, 784))

    is_training_ph = tf.placeholder(tf.bool, shape=(), name='is_training')

    z_dims = args.z_dims if args.latent == 'euc' else args.z_dims+1
    activation = {
        'elu': tf.nn.elu,
        'tanh': tf.nn.tanh
    }[args.ae_activation]

    if args.implicit:
        x1 = tf.tile(x_ph, [args.n_particles, 1])
        model_cls = models.ImplicitAE
        model = model_cls(
            x=x1, h_dim=args.h_dims, z_dim=z_dims, eps_dim=args.eps_dims,
            latent_space=args.latent, ae_activation=args.ae_activation,
            energy_activation=args.energy_activation,
            rescale_sph_latent=args.rescale_sph_latent,
            observation=args.observation,
            gan=(args.model=='wae-gan'),
            is_training_ph=is_training_ph)

        optimizer = {
            'vae': optimizers.ImplicitVAE,
            'wae': optimizers.ImplicitWAE,
            'wae-gan': optimizers.GanWAE,
            'wae-mmd': optimizers.MMDWAE,
        }[args.model](model, args)
        ais_method = args.ais_method if args.latent == 'euc' else 'riem_ld'
        log_lh_sym = log_lh_ais(model, optimizer, ais_method)
    else:
        assert not args.model.startswith('wae')
        dist = {
            'euc': 'normal',
            'sph': 'vmf'
        }[args.latent]
        model = models.ExplicitAE(
            x=x_ph, h_dim=args.h_dims, z_dim=z_dims, distribution=dist,
            rescale_sph_latent=args.rescale_sph_latent)
        optimizer = optimizers.ExplicitVAE(model, args)
        log_lh_lb_sym = log_likelihood(model, optimizer, n=1000)
        ais_method = args.ais_method if args.latent == 'euc' else 'riem_ld'
        log_lh_sym = log_lh_ais(model, optimizer, ais_method)

    # === FOR VIS ===
    if args.latent == 'euc':
        dist_pz = tf.distributions.Normal(tf.zeros_like(model.z), tf.ones_like(model.z))
        pz_sample = dist_pz.sample()
    elif args.latent == 'sph':
        dist_pz = HypersphericalUniform(model.z_dim - 1, dtype=model.x.dtype)
        pz_sample = dist_pz.sample([tf.shape(model.z)[0]])
    x_sample_sym = tf.nn.sigmoid(model._decoder(pz_sample))
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tfv1.train.Saver(keep_checkpoint_every_n_hours=2, max_to_keep=2)

    print(args)

    with LogContext(args.n_iter//100, logdir=args.dir, tfsummary=True) as ctx:
        for i in ctx:
            for j in range(100):
                # training
                x_mb, _ = mnist.train.next_batch(args.batch_size)
                if args.observation != 'normal':
                    x_mb = (x_mb > np.random.random(size=x_mb.shape)).astype(np.float32)
                optimizer.step(sess, {x_ph: x_mb, is_training_ph: True})
    
            # plot validation
            x_mb = val_images
            if args.observation != 'normal':
                x_mb = (x_mb > np.random.random(size=x_mb.shape)).astype(np.float32)
            to_log = sess.run(
                {**optimizer.print}, {x_ph: x_mb, is_training_ph: False})
            ctx.log_scalars(to_log, list(to_log))

            if i % 80 == 0 and i > 0:
                if not args.implicit:
                    idcs = np.arange(x_mb.shape[0]); np.random.shuffle(idcs); idcs = idcs[:48]
                    loglh_val = sess.run(
                        log_lh_sym, {model.x: x_mb[idcs], is_training_ph: False})
                    to_log.update({
                        'log_lh_mean': np.mean(loglh_val),
                        'log_lh_sd': np.std(loglh_val) / (idcs.shape[0] ** 0.5)
                    })
                if args.do_fid:
                    sample_images = sess.run(x_sample_sym, {
                        model.x: np.zeros_like(val_images[:10000]).astype('f'),
                        is_training_ph: False
                    })
                    fid_score = compute_fid(sample_images)
                    to_log['fid'] = fid_score
                print(i, to_log)

            if (i * 100) % args.save_every == 0 and args.save_every > 0:
                saver.save(sess, os.path.join(args.dir, 'model'), global_step=i)
                print('Model saved')
    
    print('Test/validation:')
        
    test_images = val_images

    sample_images = sess.run(x_sample_sym, {
        model.x: np.zeros_like(val_images[:10000]).astype('f'),
        is_training_ph: False
    })
    im_tiled = tile_images(sample_images[:100].reshape((100, 28, 28))) 
    save_image(os.path.join(args.dir, 'sampled.png'), im_tiled)

    if args.do_fid:
        fid_score = compute_fid(sample_images)
        print('FID SCORE =', fid_score)
        return

    if not args.implicit:
        x_mb = val_images
        if args.observation != 'normal':
            # dynamic binarization
            x_mb = (x_mb > np.random.random(size=x_mb.shape)).astype(np.float32)
        print_ = {**optimizer.print}
        print_['LL'] = log_lh_lb_sym
        print(sess.run(print_, {model.x: x_mb, is_training_ph: False}))

    lsum = 0
    lden = 0
    idc = np.arange(test_images.shape[0])
    np.random.shuffle(idc)
    test_images = test_images[idc]
    with LogContext(test_images.shape[0]//args.batch_size) as ctx:
        for j in ctx:
            ti = test_images[j*args.batch_size: (j+1)*args.batch_size]
            if args.observation != 'normal':
                ti = (ti > np.random.random(size=ti.shape)).astype(np.float32)
            lhs = sess.run(log_lh_sym, {model.x: ti, is_training_ph: False})
            lsum += lhs.mean() * ti.shape[0]
            lden += ti.shape[0]
            if j % 10 == 0:
                ctx.log_scalars({'avg': lsum/lden}, ['avg'])
    print('AIS lb = {}'.format(lsum/lden))


if __name__ == '__main__':
    args = parser.parse_args()
    experiments.utils.preflight(args)
    main(args)

