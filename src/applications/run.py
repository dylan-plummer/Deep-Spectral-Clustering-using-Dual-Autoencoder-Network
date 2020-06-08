import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

from core.data import get_data
from applications.DSCDAN import run_net
from utils import anchor_list_to_dict
from rotated_cell_data_generator import DataGenerator


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--dset', type=str, help='gpu number to use', default='pfc')
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--anchor_file', required=True, type=str)
parser.add_argument('--reference', required=True, type=str)
parser.add_argument('--min_depth', default=75000, type=int)
parser.add_argument('--load_cells', default=True, type=bool)
args = parser.parse_args()

print('Dataset:', args.dset)

data_dir = args.data_dir
anchor_file = args.anchor_file
file_reads = args.reference
min_depth = args.min_depth

load_data = args.load_cells
sparse_matrices = {}
if load_data:
    if 'sparse_matrices.sav' in os.listdir('.'):
        with open('sparse_matrices.sav', 'rb') as f:
            sparse_matrices = joblib.load(f)

anchor_list = pd.read_csv(anchor_file, sep='\t', names=['chr', 'start', 'end', 'anchor', 'length'],
                          usecols=['chr', 'start', 'end', 'anchor'], engine='python')  # read anchor list file
anchor_dict = anchor_list_to_dict(anchor_list['anchor'].values)  # convert to anchor --> index dictionary
reference = pd.read_csv(file_reads, sep='\t', names=['cell', 'depth', 'cluster'])
reference = reference[reference['depth'] >= min_depth]
#reference['cell'] = reference['cell'].apply(lambda r: r.replace('.remap', '.50kb'))  # add file ending to all cell names if using different resolution
cluster_names = list(reference['cluster'].unique())
np.random.seed(36)

print('Cluster names: ', cluster_names)
print('%d total cells' % len(reference))
reference.set_index('cell', inplace=True)

train_generator = DataGenerator(sparse_matrices, anchor_list, anchor_dict, data_dir, reference)

input_shape = (train_generator.matrix_len + train_generator.matrix_pad, train_generator.limit2Mb, 1)
print('Input shape:', input_shape)

# SELECT GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

params = defaultdict(lambda: None)

# SET GENERAL HYPERPARAMETERS
general_params = {
        'dset': args.dset,                  # dataset: reuters / mnist
        }
params.update(general_params)

print('Dataset:', params['dset'])

# SET DATASET SPECIFIC HYPERPARAMETERS
if args.dset == 'mnist':
    mnist_params = {
        'n_clusters': 10,                   # number of clusters in data
        'n_nbrs': 5,                        # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
        'scale_nbr': 2,                     # neighbor used to determine scale of gaussian graph Laplacian; calculated by
        'batch_size': 1024,                 # batch size for spectral net
        'use_approx': False,                # enable / disable approximate nearest neighbors
        'use_all_data': True,               # enable to use all data for training (no test set)
        'latent_dim': 120,
        'img_dim': 28,
        'filters': 16
        }
    params.update(mnist_params)
elif args.dset == 'pfc':
    pfc_params = {
        'n_clusters': 14,  # number of clusters in data
        'n_nbrs': 5,  # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
        'scale_nbr': 2,  # neighbor used to determine scale of gaussian graph Laplacian; calculated by
        'batch_size': 64,  # batch size for spectral net
        'use_approx': False,  # enable / disable approximate nearest neighbors
        'use_all_data': True,  # enable to use all data for training (no test set)
        'latent_dim': 128,
        'input_shape': input_shape,
        'filters': 16
    }
    params.update(pfc_params)

data = get_data(params, train_generator)

# RUN EXPERIMENT
run_net(data, params)


