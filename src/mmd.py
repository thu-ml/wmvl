import numpy as np
import tensorflow as tf


def mmd(sample_pz, sample_qz):
    n = tf.shape(sample_qz)[0]
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
    half_size = tf.cast((n * n - n) / 2, tf.int32)

    norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
    dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
    distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

    norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
    dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
    distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

    dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
    distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

    # if opts['verbose']:
    #     distances = tf.Print(
    #         distances,
    #         [tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]],
    #         'Maximal Qz squared pairwise distance:')
    #     distances = tf.Print(distances, [tf.reduce_mean(distances_qz)],
    #                         'Average Qz squared pairwise distance:')

    #     distances = tf.Print(
    #         distances,
    #         [tf.nn.top_k(tf.reshape(distances_pz, [-1]), 1).values[0]],
    #         'Maximal Pz squared pairwise distance:')
    #     distances = tf.Print(distances, [tf.reduce_mean(distances_pz)],
    #                         'Average Pz squared pairwise distance:')

    # Median heuristic for the sigma^2 of Gaussian kernel
    sigma2_k = tf.nn.top_k(
        tf.reshape(distances, [-1]), half_size).values[half_size - 1]
    sigma2_k += tf.nn.top_k(
        tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
    # Maximal heuristic for the sigma^2 of Gaussian kernel
    # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
    # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
    # sigma2_k = opts['latent_space_dim'] * sigma2_p
    res1 = tf.exp( - distances_qz / 2. / sigma2_k)
    res1 += tf.exp( - distances_pz / 2. / sigma2_k)
    res1 = tf.multiply(res1, 1. - tf.eye(n))
    res1 = tf.reduce_sum(res1) / (nf * nf - nf)
    res2 = tf.exp( - distances / 2. / sigma2_k)
    res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
    return res1 - res2

