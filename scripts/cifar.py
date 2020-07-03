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
    'z_dims': [16, 32, 64],
    'seed': list(range(1334, 1337)),
    'model': ['wae', 'wae-gan', 'wae-mmd'],
    'n_iter': [200001],
}
# NOTE CHANGE THIS
log_dir = os.path.expanduser('~/s-vae-tf/run/svae/cifar/'.format(__file__.split('.')[0]))
tasks = runner.list_tasks(
    'python cifar.py -eps_dim 0 ',
    param_specs,
    slave_working_dir,
    log_dir + 'prefix')

print('\n'.join([t.cmd for t in tasks]))
print(len(tasks))

# NOTE CHANGE THIS
r = runner.Runner(
    n_max_gpus=6, n_multiplex=1, n_max_retry=-1)
r.run_tasks(tasks)
