from score_est import riem_ld, euc_ld, Rn, Sn
import tensorflow as tf
import numpy as np


def ld_move(y0, energy_emb, stepsz, n_steps, method):

    cond = lambda *args: tf.less(args[0], n_steps)

    def body(counter, y, acc_sum):
        if method == 'riem_ld':
            ny, _, acc = riem_ld(y, energy_emb, stepsz, Sn, rescale=False)
        elif method == 'riem_euc_ld':
            ny, _, acc = riem_ld(y, energy_emb, stepsz, Rn, rescale=False)
        elif method == 'ld':
            ny, _, acc = euc_ld(y, energy_emb, stepsz, Rn, rescale=False)
        else:
            raise NotImplemented()
        acc = acc[:, None]
        ny = acc * ny + (1-acc) * y
        return counter+1, ny, acc_sum + tf.reduce_mean(acc)

    init = [tf.constant(0.), y0, tf.constant(0.)]

    _, y1, acc_sum = tf.while_loop(cond, body, init, swap_memory=True)

    return y1, acc_sum / tf.to_float(n_steps)


def ld_update(stepsz, cur_acc_rate, hist_acc_rate, target_acc_rate,
              ssz_inc, ssz_dec, ssz_min, ssz_max, avg_acc_decay):
    new_stepsz = tf.cond(hist_acc_rate > target_acc_rate, lambda: ssz_inc, lambda: ssz_dec) * stepsz
    new_stepsz = tf.maximum(tf.minimum(new_stepsz, ssz_max), ssz_min)
    new_avg_acc_rate = avg_acc_decay * hist_acc_rate + (1 - avg_acc_decay) * cur_acc_rate
    return new_stepsz, new_avg_acc_rate

