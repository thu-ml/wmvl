import tensorflow as tf
import zhusuan as zs
import sys


__all__ = ['svgd', 'spos_step_']

def C(a, d):
    return tf.cast(a, d)


def rbf_kernel(theta_x, theta_y, bandwidth='median', dtype=tf.float32):
    """
    :param theta: tensor of shape [n_particles, n_params]
    :return: tensor of shape [n_particles, n_particles]
    """
    n_x = tf.shape(theta_x)[0]
    pairwise_dists = tf.reduce_sum(
        (tf.expand_dims(theta_x, 1) - tf.expand_dims(theta_y, 0)) ** 2,
        axis=-1)
    if bandwidth == 'median':
        bandwidth = tf.contrib.distributions.percentile(
            tf.squeeze(pairwise_dists), q=50.)
        bandwidth = C(0.5, dtype) * bandwidth / tf.log(tf.cast(n_x, dtype) + 1)
        bandwidth = tf.maximum(tf.stop_gradient(bandwidth), C(1e-5, dtype))
    Kxy = tf.exp(-pairwise_dists / bandwidth / 2)
    return Kxy, None


def _squeeze(tensors, n_particles):
    return tf.concat(
        [tf.reshape(t, [n_particles, -1]) for t in tensors], axis=1)


def _unsqueeze(squeezed, original_tensors):
    ret = []
    offset = 0
    for t in original_tensors:
        size = tf.reduce_prod(tf.shape(t)[1:])
        buf = squeezed[:, offset: offset+size]
        offset += size
        ret.append(tf.reshape(buf, tf.shape(t)))
    return ret 


def svgd(n_particles, loglh_grad, params, kernel, method='svgd', dtype=tf.float32):
    params_squeezed = _squeeze(params, n_particles)
    Kxy, _ = kernel(params_squeezed, tf.stop_gradient(params_squeezed), dtype=dtype)
    # We want dykxy[x] := sum_y\frac{\partial K(x,y)}{\partial y}
    # tf does not support Jacobian, and tf.gradients(Kxy, theta) returns
    # ret[x] = \sum_y\frac{\partial K(x,y)}{\partial x}
    # For stationary kernel ret = -dykxy.
    dykxy = -tf.gradients(Kxy, params_squeezed)[0]
    
    assert params[0].shape.ndims == loglh_grad[0].shape.ndims
    grads = _squeeze(loglh_grad, n_particles)
    svgd_grads = (tf.matmul(Kxy, grads) + dykxy) / C(n_particles, dtype)
    
    _k1 = tf.stop_gradient(tf.reduce_sum(Kxy, axis=-1))
    wsgld_grads = grads + \
        -tf.gradients(Kxy / _k1[None, :], params_squeezed)[0] + \
        dykxy / _k1[:, None]

    if method == 'svgd':
        new_grads = svgd_grads
    elif method == 'gfsf':
        new_grads = (grads + tf.matrix_inverse(Kxy) @ dykxy) / tf.cast(
            n_particles, dtype)
    elif method == 'wsgld':
        new_grads = wsgld_grads
    elif method == 'pisgld':
        new_grads = (wsgld_grads + svgd_grads) / 2
    else:
        raise NotImplementedError()

    return list(zip(_unsqueeze(new_grads, params), params))


def spos_step_(counter, x, inv_energ_scale, model_fwd, lr, mix_prop=0.5, dtype=tf.float32):
    TOF = lambda t: tf.cast(t, dtype)
    energ = model_fwd(x)
    x_grad = tf.gradients(energ, [x])[0]
    lr = lr / tf.stop_gradient(tf.maximum(
        TOF(1),
        tf.reduce_mean(tf.reduce_sum(x_grad**2, axis=-1, keepdims=True)**TOF(0.5))))
    svgd_update = svgd(
        tf.shape(x)[0], [-x_grad / inv_energ_scale], [x], rbf_kernel,
        dtype=dtype)
    svgd_grad = svgd_update[0][0]
    noise = tf.random_normal(
        tf.shape(x), mean=0.0, stddev=tf.sqrt(2 * lr), dtype=dtype)
    nx = x + (1-mix_prop) * lr * svgd_grad + mix_prop * (-lr * x_grad / inv_energ_scale + noise)
    cv_term = -tf.reduce_sum(x_grad * noise * mix_prop, axis=[-1])[:, None]
    return counter+1, nx, cv_term, locals()
