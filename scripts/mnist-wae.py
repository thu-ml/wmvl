from experiments.master import runner
import logging
import os
import shutil
import sys


logging.basicConfig(
        stream=sys.stderr, level=logging.DEBUG,
        format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')

slave_working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')
param_specs = {
    'n_iter': [200000],
    'seed': list(range(1234, 1244)),
    'model': ['wae', 'wae-gan'],
    'z_dims': [8],
    'eps_dims': [0],
    'e_act': ['tanh'],
}

log_dir = os.path.expanduser('~/s-vae-tf/run/svae/{}/'.format(__file__.split('.')[0]))

tasks = runner.list_tasks(
    'python mnist.py -observation sigmoid -do_fid -test ',
    param_specs,
    slave_working_dir,
    log_dir + 'prefix')

print('\n'.join([t.cmd for t in tasks]))

r = runner.Runner(
    n_max_gpus=6, n_multiplex=3, n_max_retry=-1)
r.run_tasks(tasks)
