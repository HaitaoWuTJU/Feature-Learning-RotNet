from __future__ import print_function
import argparse
import os
# import imp
import importlib
import warnings

# warnings.filterwarnings('ignore')
import algorithms as alg
from dataloader import CifarDataLoader, CifarDataset

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, default='', help='config file with parameters of the experiment')
parser.add_argument('--evaluate', default=False, action='store_true')
parser.add_argument('--checkpoint', type=int, default=0, help='checkpoint (epoch id) that will be loaded')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--disp_step', type=int, default=50, help='display step during training')
args_opt = parser.parse_args()

exp_config_file = os.path.join('.', 'config', args_opt.exp + '.py')
# if args_opt.semi == -1:
exp_directory = os.path.join('.', 'experiments', args_opt.exp)
# else:
#    assert(args_opt.semi>0)
#    exp_directory = os.path.join('.','experiments/unsupervised',args_opt.exp+'_semi'+str(args_opt.semi))

# Load the configuration params of the experiment
print('Launching experiment: %s' % exp_config_file)
config = importlib.machinery.SourceFileLoader("", exp_config_file).load_module().config
# print('config-----------------',config)
# config = importlib.machinery.SourceFileLoader("", exp_config_file).load_module().config
config['exp_dir'] = exp_directory  # the place where logs, models, and other stuff will be stored
print("Loading experiment %s from file: %s" % (args_opt.exp, exp_config_file))
print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))

# Set train and test datasets and the corresponding data loaders
data_train_opt = config['data_train_opt']
data_test_opt = config['data_test_opt']
num_imgs_per_cat = data_train_opt['num_imgs_per_cat'] if ('num_imgs_per_cat' in data_train_opt) else None

dataset_train = CifarDataset(
    dataset_name=data_train_opt['dataset_name'],
    split=data_train_opt['split'],
    random_sized_crop=data_train_opt['random_sized_crop'],
    num_imgs_per_cat=num_imgs_per_cat)
dataset_test = CifarDataset(
    dataset_name=data_test_opt['dataset_name'],
    split=data_test_opt['split'],
    random_sized_crop=data_test_opt['random_sized_crop'])

dloader_train = CifarDataLoader(
    dataset=dataset_train,
    batch_size=data_train_opt['batch_size'],
    unsupervised=data_train_opt['unsupervised'],
    epoch_size=data_train_opt['epoch_size'],
    num_workers=args_opt.num_workers,
    shuffle=True)

dloader_test = CifarDataLoader(
    dataset=dataset_test,
    batch_size=data_test_opt['batch_size'],
    unsupervised=data_test_opt['unsupervised'],
    epoch_size=data_test_opt['epoch_size'],
    num_workers=args_opt.num_workers,
    shuffle=False)

# from tqdm import tqdm
# for idx, batch in enumerate(tqdm(dloader_train(0))):
#         # print(label)
#         print(batch)
#         pass

config['disp_step'] = args_opt.disp_step
algorithm = getattr(alg, config['algorithm_type'])(config)
if args_opt.cuda:  # enable cuda
    algorithm.load_to_gpu()
if args_opt.checkpoint > 0:  # load checkpoint
    algorithm.load_checkpoint(args_opt.checkpoint, train=(not args_opt.evaluate))

if not args_opt.evaluate:  # train the algorithm
    algorithm.solve(dloader_train, dloader_test)
else:
    algorithm.evaluate(dloader_test)  # evaluate the algorithm
