import tensorflow as tf
import numpy as np
import tensorflow.distributions as tfd
import svgd


dtype = tf.float32
to_float = lambda x: tf.cast(x, dtype)


class Sn:

    def __init__(self, y):
        self.flip = to_float(tf.stop_gradient(tf.less(y[:, -1], 0)))[:,None]
        y = (-self.flip*2+1) * y
        self.x = y[..., :-1] /  (1+y[..., -1:])
        
    def metric_diag(self, x=None):
        x = self.x if x is None else x
        v = (1 + tf.reduce_sum(x**2, axis=-1, keepdims=True)) ** (-2)
        return tf.ones_like(x) * 4 * v
    
    def invdiag_grad(self, x=None):
        # return \partial_i g^ii
        x = self.x if x is None else x
        return (1 + tf.reduce_sum(x**2, axis=-1, keepdims=True)) * x
    
    def embed_coord(self, x):
        z = tf.reduce_sum(x**2, axis=-1, keepdims=True)
        ylast = (1 - z) / (1 + z)
        y = tf.concat([(1 + ylast) * x, ylast], axis=-1)
        return (-self.flip * 2 + 1) * y
    

class Rn:
    
    def __init__(self, y):
        self.x = y
        
    def metric_diag(self, x=None):
        return tf.ones_like(self.x)
    
    def invdiag_grad(self, x=None):
        return tf.zeros_like(self.x)
    
    def embed_coord(self, x):
        return x
    
    
def transformed_energy(x, energ_emb, manif):
    """
    :return: [None, 1]
    
    p_x = p_y / det|dx/dy|
    d/dx_i  = dy_j/dx_i d/dy_j
    gx = (dy/dx)g_y(dy/dx)^T
    => det|dx/dy| = 1/det|dy/dx| = 1/sqrt|gx|
    log p_x = log p_y + log|gx|/2
    E_x = E_y - log|gx|/2
    """
    e_emb = to_float(energ_emb(manif.embed_coord(x)))
    assert e_emb.shape.ndims == 2
    diag_g = manif.metric_diag(x)
    return e_emb - tf.reduce_sum(tf.log(diag_g), axis=-1, keepdims=True) / 2


def mh_acceptance_step(mh_step_fn, x0, x1):
    # log ratio = log P(x1)g(x0|x1) - {x1|x0} =: mh_step_fn(x0, x1) - mh_step_fn(x1, x0)
    mh_log_ratio = mh_step_fn(x0, x1) - mh_step_fn(x1, x0)
    assert mh_log_ratio.shape.ndims == 1
    unif = tf.random_uniform(shape=tf.shape(mh_log_ratio))
    return tf.stop_gradient(tf.to_float(tf.log(unif + 1e-9) < mh_log_ratio))
    

def riem_ld(y, energ_emb, stepsz0, Manif, energy_scale=1., cv=True, rescale=True):
    """
    :param y: coordinates in embedded space
    :param energy_emb: energy function of the density in embedded space
    :param Manif: Sn or Rn
    :param energy_scale: recaling applied to target distribution when simulating the GF
    :return: (y1, ediff)
    energy scaling and subtraction happens in embedded space
    """
    assert y.shape.ndims == 2 # [BS, ydims]
    m = Manif(y)
    diag_g = m.metric_diag()
    inv_diag_g = 1. / diag_g
    epos_emb = energ_emb(m.embed_coord(m.x))
    xg_emb = tf.gradients(epos_emb, [m.x])[0]
    # for RLD. You have to transform the density as you transformed the space
    # alternatively, the extra term plus the other 2 constitutes the Riem BM.
    xg = tf.gradients(
        transformed_energy(m.x, lambda y: energ_emb(y)*energy_scale, m),
        [m.x])[0]
    drift = -inv_diag_g * xg + m.invdiag_grad()
    if rescale:
        stepsz = tf.stop_gradient(
            stepsz0 / tf.maximum(1., (tf.reduce_sum(drift**2, axis=-1))))
    else:
        stepsz = tf.ones([tf.shape(y)[0]]) * stepsz0

    x = m.x + stepsz[:,None] * drift
    noise = tf.random_normal(
        shape=tf.shape(x), mean=to_float(0), stddev=tf.sqrt(2*stepsz[:,None]*inv_diag_g),
        dtype=dtype)
    x += noise

    # EDIFF = (E1-E0) / stepsize
    ediff = energ_emb(m.embed_coord(x)) - epos_emb
    if cv:
        ediff -= tf.reduce_sum(xg_emb * noise, axis=-1, keepdims=True)
    ediff = ediff * (stepsz0 / stepsz)[:,None]

    # M-H
    def mh_r(x0, x1): # log P(x1)g(x0|x1)
        e_x1 = transformed_energy(x1, energ_emb, m)[:,0]
        inv_diag_g_x1 = 1. / m.metric_diag(x1)
        grad_x1 = tf.gradients(e_x1, [x1])[0]
        nx_mean = x1 + stepsz[:,None] * (-inv_diag_g_x1 * grad_x1 + m.invdiag_grad(x1))
        g_x0_x1 = tfd.Normal(loc=nx_mean, scale=tf.sqrt(2*stepsz[:,None]*inv_diag_g_x1)).log_prob(x0)
        g_x0_x1 = tf.reduce_sum(g_x0_x1, axis=-1)
        return g_x0_x1 - e_x1
    accept = mh_acceptance_step(mh_r, m.x, x)

    return m.embed_coord(x), ediff, accept


def euc_spos(y, energ_emb, stepsz0, Manif, energy_scale=1., cv=True, rescale=True, alpha=0.5):
    assert Manif == Rn and y.shape.ndims == 2
    old_x = y
    epos = energ_emb(old_x)
    xg = tf.gradients(epos, [old_x])[0]
    if rescale:
        stepsz = tf.stop_gradient(
            stepsz0 / tf.maximum(1., (tf.reduce_sum(xg**2, axis=-1))))
    else:
        stepsz = tf.ones([tf.shape(old_x)[0]]) * stepsz0
    svgd_update = svgd.svgd(
        tf.shape(old_x)[0], [-xg * energy_scale], [old_x], svgd.rbf_kernel)
    svgd_grad = svgd_update[0][0]
    noise = tf.random_normal(
        tf.shape(old_x), mean=0.0, stddev=tf.sqrt(2*stepsz[:,None]), dtype=dtype)
    x = old_x + alpha * (stepsz[:,None] * svgd_grad) + \
        (1-alpha) * (-stepsz[:,None] * energy_scale * xg + noise)

    # EDIFF
    ediff = energ_emb(x) - epos
    if cv:
        ediff -= tf.reduce_sum(xg * noise * (1-alpha), axis=-1, keepdims=True)
    ediff = ediff * (stepsz0 / stepsz)[:,None]

    return x, ediff, (lambda: NotImplemented("M-H"))


def euc_ld(y, energ_emb, stepsz0, Manif, energy_scale=1., cv=True, rescale=True):
    """
    :param y: coordinates in embedded space
    :param energy_emb: energy function of the density in embedded space
    :param Manif: Sn or Rn
    :param energy_scale: recaling applied to target distribution when simulating the GF
    :return: (y1, ediff)
    """
    assert y.shape.ndims == 2 # [BS, ydims]
    m = Manif(y)
    diag_g = m.metric_diag()
    epos = transformed_energy(m.x, energ_emb, m)
    xg = tf.gradients(epos, [m.x])[0]
    if rescale:
        stepsz = tf.stop_gradient(
            stepsz0 / tf.maximum(1., (tf.reduce_sum(xg**2, axis=-1))))
    else:
        stepsz = tf.ones([tf.shape(y)[0]]) * stepsz0
    noise = tf.random_normal(
        shape=tf.shape(m.x), mean=to_float(0), stddev=tf.sqrt(2*stepsz[:,None]),
        dtype=dtype)
    x = m.x - stepsz[:,None] * energy_scale * xg + noise

    # EDIFF
    ediff = transformed_energy(x, energ_emb, m) - epos
    if cv:
        ediff -= tf.reduce_sum(xg * noise, axis=-1, keepdims=True)
    ediff = ediff * (stepsz0 / stepsz)[:,None]

    # M/H; **assume energy_scale=1**
    # in euc ld we can ignore the normalizer of mvn; don't do this in riem ld
    def mh_r(x0, x1): # log P(x1)g(x0|x1)
        e_x1 = tf.squeeze(transformed_energy(x1, energ_emb, m), axis=-1)
        grad_x1 = tf.gradients(e_x1, [x1])[0]
        nx_mean = x1 - stepsz[:,None] * grad_x1
        return -tf.reduce_sum((x0 - nx_mean)**2, -1) / (2 * (2*stepsz)) - e_x1
    accept = mh_acceptance_step(mh_r, m.x, x)

    y1 = m.embed_coord(x)
    return y1, ediff, accept


def mpf_euc(y, energ_emb, stepsz):
    return euc_ld(y, energ_emb, stepsz, Rn, to_float(0.5))[:2]


def mpf_sph(y, energ_emb, stepsz):
    return riem_ld(y, energ_emb, stepsz, Sn, to_float(0.5))[:2]


def mpf_euc_spos(y, energ_emb, stepsz, alpha=0.5):
    return euc_spos(y, energ_emb, stepsz, Rn, to_float(0.5), alpha=alpha)[:2]

