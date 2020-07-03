'''
This script does grid search. 
The runner module determines log dir names, spawns process in the most vacant 
GPU, and redirects stdout/stderr of the slave.
'''

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
    'seed': list(range(1234, 1239)),
    ('z_dims', 'eps_dims'): [8, 32],
    'lr': [1e-3],
    'n_iter': [100000],
    'optimizer': ['rmsprop'],
    'model': ['vae']
}

log_dir = os.path.expanduser('~/s-vae-tf/run/svae/vae-explicit/'.format(__file__.split('.')[0]))

tasks = runner.list_tasks(
    'python mnist.py -explicit -test ',
    param_specs,
    slave_working_dir,
    log_dir + 'prefix')
    
print('\n'.join([t.cmd for t in tasks]))

r = runner.Runner(
    n_max_gpus=6, n_multiplex=3, n_max_retry=-1)
r.run_tasks(tasks)
