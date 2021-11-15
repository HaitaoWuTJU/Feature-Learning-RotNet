from __future__ import print_function
import paddle
import random
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv

from paddle.fluid.dataloader.collate import default_collate_fn
from paddle.io import Dataset
from paddle.vision import transforms, Cifar10, datasets
from pdb import set_trace as breakpoint
from paddle.io import DataLoader

import warnings
warnings.filterwarnings('ignore')
BATCH_SIZE = 128
_CIFAR_DATASET_DIR = './datasets/CIFAR'


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class CifarDataset(Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False,
                 num_imgs_per_cat=None):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop

        # The num_imgs_per_cats input argument specifies the number
        # of training examples per category that would be used.
        # This input argument was introduced in order to be able
        # to use less annotated examples than what are available
        # in a semi-superivsed experiment. By default all the
        # available training examplers per category are being used.
        self.num_imgs_per_cat = num_imgs_per_cat

        if self.dataset_name == 'cifar10':
            self.mean_pix = [x / 255.0 for x in [125.3, 123.0, 113.9]]
            self.std_pix = [x / 255.0 for x in [63.0, 62.1, 66.7]]

            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the CIFAR dataset')

            transform = []
            if (split != 'test'):
                transform.append(transforms.RandomCrop(32, padding=4))
                transform.append(transforms.RandomHorizontalFlip())
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = Cifar10(mode=self.split, transform=self.transform)
            # self.data = datasets.__dict__[self.dataset_name.upper()](
            #     _CIFAR_DATASET_DIR, train=self.split == 'train',
            #     download=True, transform=self.transform)
        else:
            raise ValueError('Not recognized dataset {0}'.format(self.dataset_name))

        if num_imgs_per_cat is not None:
            self._keep_first_k_examples_per_category(num_imgs_per_cat)

    def _keep_first_k_examples_per_category(self, num_imgs_per_cat):
        print('num_imgs_per_category {0}'.format(num_imgs_per_cat))

        if self.dataset_name == 'cifar10':
            labels = self.data.test_labels if (self.split == 'test') else self.data.train_labels
            data = self.data.test_data if (self.split == 'test') else self.data.train_data
            label2ind = buildLabelIndex(labels)
            all_indices = []
            for cat in label2ind.keys():
                label2ind[cat] = label2ind[cat][:num_imgs_per_cat]
                all_indices += label2ind[cat]
            all_indices = sorted(all_indices)
            data = data[all_indices]
            labels = [labels[idx] for idx in all_indices]
            if self.split == 'test':
                self.data.test_labels = labels
                self.data.test_data = data
            else:
                self.data.train_labels = labels
                self.data.train_data = data

            label2ind = buildLabelIndex(labels)
            for k, v in label2ind.items():
                assert (len(v) == num_imgs_per_cat)
        else:
            raise ValueError('Not recognized dataset {0}'.format(self.dataset_name))

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for i, (t, m, s) in enumerate(zip(tensor, self.mean, self.std)):
            tensor[i] = t * s + m
        return tensor


def rotate_img(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 90:  # 90 degrees rotation
        return np.flipud(np.transpose(img, (1, 0, 2)))
    elif rot == 180:  # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270:  # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1, 0, 2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class CifarDataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        mean_pix = self.dataset.mean_pix
        std_pix = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1, 2, 0).astype(np.uint8),
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)

        # if self.unsupervised:
        # if in unsupervised mode define a loader function that given the
        # index of an image it returns the 4 rotated copies of the image
        # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
        # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
        if self.unsupervised:
            # if in unsupervised mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(batch):
                imgs=batch[0]
                rotated_imgs = []
                rotation_labels = []
                for img in imgs:
                    rotated_imgs.append([
                        self.transform(img),
                        self.transform(rotate_img(img, 90).copy()),
                        self.transform(rotate_img(img, 180).copy()),
                        self.transform(rotate_img(img, 270).copy())
                    ])
                    rotation_labels.append(paddle.to_tensor([0, 1, 2, 3], dtype='int64'))
                return [paddle.to_tensor(rotated_imgs), paddle.to_tensor(rotation_labels)]

            def _collate_fun(batch):
                batch = default_collate_fn(batch)
                # print(batch[1])
                assert(len(batch)==2)
                batch = _load_function(batch)
                assert (len(batch) == 2)
                # print(batch[1])
                batch_size, rotations, channels, height, width = batch[0].shape
                batch[0] = batch[0].reshape([batch_size * rotations, channels, height, width])
                batch[1] = batch[1].reshape([batch_size * rotations])
                return batch
        else:  # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            # def _load_function(batch):
                
            #     img, categorical_label = batch[0],batch[1]

            #     img = self.transform(img)
            
            #     return [img, categorical_label]
                
            def _load_function(batch):
                rotated_imgs = []
                categorical_label = []
                # print(batch[0][0])
                # print('1.1')
                for img, label in zip(batch[0], batch[1]):
                    # print('1.1.1')
                    # print(type(img))
                    img=[self.transform(img)]
                    # print('1.1.2')
                    rotated_imgs.extend(img)
                    
                    categorical_label.extend([label])
                # print(type(rotated_imgs))
                # print(rotated_imgs[0])
                # print('1.2')
                return [paddle.to_tensor(rotated_imgs), paddle.to_tensor(categorical_label)]
            def _collate_fun(batch):
                batch = default_collate_fn(batch)
                # print('----1')
                batch = _load_function(batch)
                # print('----2')
                # print('batch[0].shape',batch[0].shape)
                # print('batch[1].shape',batch[1].shape)
                return batch
            
        data_loader = DataLoader(self.dataset,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle,
                                 num_workers=self.num_workers,
                                 drop_last=False,
                                 collate_fn=_collate_fun)

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = CifarDataset(dataset_name='cifar10', split='train', random_sized_crop=False)
    data_loader = CifarDataLoader(dataset, batch_size=128,num_workers=1,shuffle=True, unsupervised=False)
    epoch=0
    from tqdm import tqdm
    for idx, batch in enumerate(tqdm(data_loader(epoch))):
        x,label=batch
        # print(x.shape)
        # print(label)
        # print(label.shape)
        # print(batch)
        # break
        pass
    epoch+=1
    plt.show()

