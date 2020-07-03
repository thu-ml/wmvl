import numpy as np
import tensorflow as tf

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from models import ENCODER, DECODER, COND_ENERGY, MARGINAL_ENERGY

from score_est import mpf_euc, mpf_sph, mpf_euc_spos
from mmd import mmd


def optimize(loss, scope, args):
    optm = {
        'adam': lambda: tf.train.AdamOptimizer(learning_rate=args.lr),
        'rmsprop': lambda: tf.train.RMSPropOptimizer(learning_rate=args.lr)
    }[args.optimizer]()
    if scope is not None:
        tvars = []
        for s in scope:
            tv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=s)
            assert len(tv)>0
            tvars += tv
        grad_and_vars = optm.compute_gradients(loss, var_list=tvars)
    else:
        grad_and_vars = optm.compute_gradients(loss, var_list=None)
    return optm.apply_gradients(grad_and_vars)


class ExplicitVAE(object):

    def __init__(self, model, args):
        """
        OptimizerVAE initializer

        :param model: a model object
        :param learning_rate: float, learning rate of the optimizer
        """

        # binary cross entropy error
        assert args.observation == 'sigmoid', NotImplemented
        self.bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=model.x, logits=model.logits)
        self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(self.bce, axis=-1))

        if args.latent == 'euc':
            # KL divergence between normal approximate posterior and standard normal prior
            self.p_z = tf.distributions.Normal(tf.zeros_like(model.z), tf.ones_like(model.z))
            kl = model.q_z.kl_divergence(self.p_z)
            self.kl = tf.reduce_mean(tf.reduce_sum(kl, axis=-1))
        elif args.latent == 'sph':
            # KL divergence between vMF approximate posterior and uniform hyper-spherical prior
            self.p_z = HypersphericalUniform(model.z_dim - 1, dtype=model.x.dtype)
            kl = model.q_z.kl_divergence(self.p_z)
            self.kl = tf.reduce_mean(kl)
        else:
            raise NotImplemented

        self.ELBO = - self.reconstruction_loss - self.kl
        self.train_step = optimize(-self.ELBO, None, args)
        self.print = {'loss/recon': self.reconstruction_loss, 'loss/ELBO': self.ELBO, 'loss/KL': self.kl}

    def step(self, sess, fd):
        sess.run(self.train_step, fd)


class ImplicitVAE(object):

    def __init__(self, model, args):

        self.model = model
        self.args = args

        # binary cross entropy error
        assert args.observation == 'sigmoid', NotImplemented
        self.bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=model.x, logits=model.logits)
        self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(self.bce, axis=-1))

        def energ_emb(z):
            return model._cond_energy(z, model.x)

        assert args.mpf_method == 'ld', NotImplemented
        if args.latent == 'euc':
            y, neg_mpf_loss = mpf_euc(model.z, energ_emb, args.mpf_lr)
        elif args.latent == 'sph':
            y, neg_mpf_loss = mpf_sph(model.z, energ_emb, args.mpf_lr)
        self.mpf_loss = -tf.reduce_mean(neg_mpf_loss)
        self.gp_loss = tf.reduce_mean(energ_emb(y)**2) + tf.reduce_mean(energ_emb(model.z)**2)
        self.score_loss = self.mpf_loss + args.grad_penalty * self.gp_loss
        self.score_opt_op = optimize(self.score_loss, [COND_ENERGY], args)

        if args.latent == 'euc':
            self.dist_pz = tf.distributions.Normal(tf.zeros_like(model.z), tf.ones_like(model.z))
            self.log_pz = tf.reduce_sum(self.dist_pz.log_prob(model.z), axis=-1)
        elif args.latent == 'sph':
            self.dist_pz = HypersphericalUniform(model.z_dim - 1, dtype=model.x.dtype)
            self.log_pz = self.dist_pz.log_prob(model.z)
        else:
            raise NotImplemented

        # Eq log(q/p)
        self.kl = -tf.reduce_mean(self.log_pz) - tf.reduce_mean(model._cond_energy(model.z, model.x))
        self.ELBO = -self.reconstruction_loss - self.kl

        self.elbo_opt_op = optimize(-self.ELBO, [ENCODER, DECODER], args)
        self.print = {
            'loss/reconloss': self.reconstruction_loss,
            'loss/ELBO': self.ELBO,
            'loss/approx_KL': self.kl,
            'loss/mpf': self.mpf_loss,
            'loss/gp': self.gp_loss,
            'e/avg': tf.reduce_mean(model._cond_energy(model.z, model.x))
        }
        self.lc = locals()

    def step(self, sess, fd):
        sess.run(self.elbo_opt_op, fd)
        for j in range(self.args.train_score_dupl):
            sess.run(self.score_opt_op, fd)


class ImplicitWAE(object):

    def __init__(self, model, args):

        self.model = model
        self.args = args

        im_axes = list(range(1, model.x.shape.ndims))
        if args.observation == 'normal':
            self.reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum((model.x-model.qxz_mean)**2, axis=im_axes))
        elif args.observation == 'sigmoid':
            self.bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=model.x, logits=model.logits)
            self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(self.bce, axis=-1))
        else:
            raise NotImplemented

        def energ_emb(z):
            return model._marginal_energy(z)

        if args.latent == 'euc':
            if args.mpf_method == 'ld':
                y, neg_mpf_loss = mpf_euc(model.z, energ_emb, args.mpf_lr)
            else:
                y, neg_mpf_loss = mpf_euc_spos(
                    model.z, energ_emb, args.mpf_lr, alpha=args.mpf_spos_alpha)
        elif args.latent == 'sph' and args.mpf_method == 'ld':
            y, neg_mpf_loss = mpf_sph(model.z, energ_emb, args.mpf_lr)
        else:
            raise NotImplemented

        self.mpf_loss = tf.reduce_mean(-neg_mpf_loss) * 1e-3 / args.mpf_lr
        self.gp_loss  = tf.reduce_mean(energ_emb(y)**2) + tf.reduce_mean(energ_emb(model.z)**2)
        self.score_loss = self.mpf_loss + args.grad_penalty * self.gp_loss
        self.score_opt_op = optimize(self.score_loss, [MARGINAL_ENERGY], args)

        if args.latent == 'euc':
            self.dist_pz = tf.distributions.Normal(tf.zeros_like(model.z), tf.ones_like(model.z))
            self.log_pz = tf.reduce_sum(self.dist_pz.log_prob(model.z), axis=-1)
        elif args.latent == 'sph':
            self.dist_pz = HypersphericalUniform(model.z_dim - 1, dtype=model.x.dtype)
            self.log_pz = self.dist_pz.log_prob(model.z)
        else:
            raise NotImplemented

        # KL = Eq(logq - logp) = Eq(-logp - energy_q)
        self.kl = -tf.reduce_mean(self.log_pz) - tf.reduce_mean(model._marginal_energy(model.z))
        self.wae_loss = self.reconstruction_loss + self.kl * args.wae_lambda
        self.wae_opt_op = optimize(self.wae_loss, [ENCODER, DECODER], args)

        self.print = {
            'loss/recon': self.reconstruction_loss,
            'loss/wae': self.wae_loss,
            'loss/mpf': self.mpf_loss,
            'loss/gp': self.gp_loss
        }

        self.lc = locals()

    def step(self, sess, fd):
        sess.run(self.wae_opt_op, fd)
        for j in range(self.args.train_score_dupl):
            sess.run(self.score_opt_op, fd)


class GanWAE(object):

    def __init__(self, model, args):

        self.model = model
        self.args = args

        im_axes = list(range(1, model.x.shape.ndims))
        if args.observation == 'normal':
            self.reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum((model.x-model.qxz_mean)**2, axis=im_axes))
        elif args.observation == 'sigmoid':
            self.bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=model.x, logits=model.logits)
            self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(self.bce, axis=-1))
        else:
            raise NotImplemented

        if args.latent == 'euc':
            self.dist_pz = tf.distributions.Normal(tf.zeros_like(model.z), tf.ones_like(model.z))
            self.log_pz = tf.reduce_sum(self.dist_pz.log_prob(model.z), axis=-1)
            pz_sample = self.dist_pz.sample()
        elif args.latent == 'sph':
            self.dist_pz = HypersphericalUniform(model.z_dim - 1, dtype=model.x.dtype)
            self.log_pz = self.dist_pz.log_prob(model.z)
            pz_sample = self.dist_pz.sample([tf.shape(model.z)[0]])
        else:
            raise NotImplemented

        def energ_emb(z):
            return model._marginal_energy(z)

        assert pz_sample.shape.ndims == 2
        pz_logits = model._marginal_energy(pz_sample)
        qz_logits = model._marginal_energy(model.z)
        self.gp_loss = tf.reduce_mean(energ_emb(model.z)**2) * 2
        self.score_loss = -(tf.reduce_mean(tf.log(tf.nn.sigmoid(pz_logits) + 1e-7)) +\
                            tf.reduce_mean(tf.log(1 - tf.nn.sigmoid(qz_logits) + 1e-7)))
        self.score_opt_op = optimize(
            self.score_loss + args.grad_penalty * self.gp_loss, [MARGINAL_ENERGY], args)

        self.kl = -tf.reduce_mean(tf.math.log_sigmoid(qz_logits)) # non-saturating GAN loss
        self.wae_loss = self.reconstruction_loss + self.kl * args.wae_lambda
        self.wae_opt_op = optimize(self.wae_loss, [ENCODER, DECODER], args)

        self.print = {
            'loss/recon': self.reconstruction_loss,
            'loss/wae': self.wae_loss,
            'loss/disc': self.score_loss,
            'loss/gp': self.gp_loss,
            'loss/kl': self.kl
        }

        self.lc = locals()

    def step(self, sess, fd):
        sess.run(self.wae_opt_op, fd)
        for j in range(self.args.train_score_dupl):
            sess.run(self.score_opt_op, fd)


class MMDWAE(object):

    def __init__(self, model, args):

        self.model = model
        self.args = args

        im_axes = list(range(1, model.x.shape.ndims))
        if args.observation == 'normal':
            self.reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum((model.x-model.qxz_mean)**2, axis=im_axes))
        elif args.observation == 'sigmoid':
            self.bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=model.x, logits=model.logits)
            self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(self.bce, axis=-1))
        else:
            raise NotImplemented

        if args.latent == 'euc':
            self.dist_pz = tf.distributions.Normal(tf.zeros_like(model.z), tf.ones_like(model.z))
            self.log_pz = tf.reduce_sum(self.dist_pz.log_prob(model.z), axis=-1)
            pz_sample = self.dist_pz.sample()
        elif args.latent == 'sph':
            self.dist_pz = HypersphericalUniform(model.z_dim - 1, dtype=model.x.dtype)
            self.log_pz = self.dist_pz.log_prob(model.z)
            pz_sample = self.dist_pz.sample([tf.shape(model.z)[0]])
        else:
            raise NotImplemented

        def energ_emb(z):
            return model._marginal_energy(z)

        assert pz_sample.shape.ndims == 2

        self.kl = matching_loss = mmd(model.z, pz_sample)
        self.wae_loss = self.reconstruction_loss + self.kl * (args.wae_lambda*100)
        self.wae_opt_op = optimize(self.wae_loss, [ENCODER, DECODER], args)

        self.print = {
            'loss/recon': self.reconstruction_loss,
            'loss/wae': self.wae_loss,
            'loss/kl': self.kl
        }

        self.lc = locals()

    def step(self, sess, fd):
        sess.run(self.wae_opt_op, fd)

