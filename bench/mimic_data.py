import sys
import os

import ml_collections
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import pickle
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from tqdm import tqdm
from PIL import Image

import h5py
from timm.data import create_transform
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as ArrowDataset
from transformers import AutoTokenizer
from collections import defaultdict

NUM_CLASSES = {"Retina": 2, "COVID": 3, "Cifar-10": 10, "openImg": 600}


def non_iid_split_dirichlet(y_train, n_clients, n_classes, beta=0.4, seed=2022):
    '''
    Utility function for data splitting
    Inputs:
        y_train: label of each image (corresponding with image_id) ===> 1-D array
        n_clients: the number of clients in each split
        n_classes: the number of classes in the dataset
        beta: the degree of non-IID based on Dirichlet dist. Smaller beta -> higher heterogeneity
    '''
    min_size = 0
    min_require_size = 8

    N = y_train.shape[0]
    generator = np.random.default_rng(seed)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(n_classes):
            idx_k = np.where(y_train == k)[0]
            generator.shuffle(idx_k)
            proportions = generator.dirichlet(np.repeat(beta, n_clients))
            proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_clients):
        generator.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    print('Data splitting finished!')
    print('Sum of labels in each client: ',
          [np.sum([y_train[idx] for idx in idx_batch]) for key, idx_batch in net_dataidx_map.items()])

    return net_dataidx_map


def iid_split(y_train, n_clients, n_classes, seed=2022):
    generator = np.random.default_rng(seed)
    net_dataidx_map = defaultdict(list)

    chosen_idx = []
    for k in range(n_classes):
        idx_k = np.where(y_train == k)[0]
        chosen_idx_k = generator.choice(idx_k, n_clients)
        chosen_idx.append(chosen_idx_k)

    chosen_idx = np.array(chosen_idx).T
    for j in range(n_clients):
        net_dataidx_map[j] = chosen_idx[j].tolist()

    print('Data splitting finished!')
    print('Sum of labels in each client: ',
          [np.sum([y_train[idx] for idx in idx_batch]) for key, idx_batch in net_dataidx_map.items()])

    return net_dataidx_map


def load_hf_data(dataset_name, data_path, phase):
    if dataset_name == 'COVID':

        hf = h5py.File(
            os.path.join(data_path, 'COVID_data',
                         dataset_name + '_' + phase + '.h5'), 'r')
        pixels = hf['pixels']
        label = hf['label']

        ## change to uint8, original in the range of [0-1]
        pixels = np.asarray(pixels) * 255

    elif dataset_name == "Cifar-10":

        data = np.load(os.path.join(data_path, "cifar10.npy"), allow_pickle=True)
        pixels, label = None, None
        if phase == "train":
            pixels = data.item()["central"]["data"]["train_central"]
            label = data.item()["central"]["target"]["train_central"]
        elif phase == "val":
            pixels = data.item()["union_val"]["data"]
            label = data.item()["union_val"]["target"]
        elif phase == "test":
            pixels = data.item()["union_test"]["data"]
            label = data.item()["union_test"]["target"]
        assert pixels is not None and label is not None
        pixels, label = np.asarray(pixels), np.asarray(label)

    elif dataset_name == "openImg":
        data = np.load(os.path.join(data_path, "openImg.npy"), allow_pickle=True)
        pixels, label = None, None
        if phase == "train":
            pixels = {k: data.item()[k]["data"] for k in data.item().keys() if k not in ["union_val", "union_test"]}
            label = {k: data.item()[k]["target"] for k in data.item().keys() if k not in ["union_val", "union_test"]}
        elif phase == "val":
            pixels = data.item()["union_val"]["data"]
            label = data.item()["union_val"]["target"]
            pixels, label = np.asarray(pixels), np.asarray(label)
        elif phase == "test":
            pixels = data.item()["union_test"]["data"]
            label = data.item()["union_test"]["target"]
            pixels, label = np.asarray(pixels), np.asarray(label)

    else:
        raise NotImplementedError()

    return pixels, label


class ImageDataset(Dataset):

    def __init__(self, data, label, phase='train'):
        self.data = data
        self.label = label

        if phase == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            ])

        self.transform = transform

    def __getitem__(self, index):

        x = self.data[index]
        y = self.label[index]
        ## if x is grayimage, then broadcast to 3 images
        ## ADNI to 3 channels
        if len(x.shape) < 3:
            x = np.stack([x, x, x], 2)

        x = np.asarray(x).astype('uint8')

        img = Image.fromarray(x)
        y = np.asarray(y).astype('int64')

        if self.transform is not None:
            img = self.transform(img)

        return img, y

    def __len__(self):
        return len(self.data)


def get_dataloader(dataset_name, data_path, n_client, iid=True, sub_dataset_size=None, beta=0.5, seed=42):
    train_pixels, train_label = load_hf_data(dataset_name, data_path, 'train')
    if dataset_name == 'COVID':
        train_pixels = train_pixels[::3, :, :, :]
        train_label = train_label[::3, ]
    test_pixels, test_label = load_hf_data(dataset_name, data_path, 'test')
    val_pixels, val_label = load_hf_data(dataset_name, data_path, 'val')

    groups = []
    trainsep = {}

    if dataset_name in ["openImg"]:  # pre-split datasets
        for i, split in enumerate(train_pixels):
            trainsep[str(i)] = [train_pixels[split], train_label[split]]
    else:  # regular sample
        if sub_dataset_size is not None:
            num_classes = len(np.unique(train_label))
            inds = [np.where(train_label == label)[0] for label in range(num_classes)]
            _train_pixels = []
            _train_label = []
            cur_size = 0
            for ind in inds[:-1]:
                offset = int(sub_dataset_size / len(inds))
                cur_size += offset
                _train_pixels.append(train_pixels[ind[:offset], :, :, :])
                _train_label.append(train_label[ind[:offset],])
            _train_pixels.append(train_pixels[inds[-1][:sub_dataset_size - cur_size], :, :, :])
            _train_label.append(train_label[inds[-1][:sub_dataset_size - cur_size],])
            train_pixels = np.concatenate(_train_pixels)
            train_label = np.concatenate(_train_label)
            print('current train set size: ', train_pixels.shape[0])

        user_pixels = {}
        total_pixels = len(train_pixels)
        if n_client is not None:
            if iid:
                data_idx_map = iid_split(train_label, n_client, NUM_CLASSES[dataset_name],
                                         seed=seed)
            else:
                data_idx_map = non_iid_split_dirichlet(train_label, n_client,
                                                       NUM_CLASSES[dataset_name], beta=beta,
                                                       seed=seed)
            for i in range(len(data_idx_map)):
                user_pixels[i] = data_idx_map[i]
        else:
            for i in range(total_pixels):
                user_pixels[i] = [i]
        # num_users = len(train_pixels) // cfg.client_data_num
        # users = [i for i in range(num_users)]
        for i in user_pixels:
            trainsep[str(i)] = [[train_pixels[data_idx] for data_idx in user_pixels[i]],
                                [train_label[data_idx] for data_idx in user_pixels[i]]]

    users = list(trainsep.keys())

    test_set = ImageDataset(test_pixels,
                            test_label,
                            phase='test')
    tests = DataLoader(dataset=test_set,
                       batch_size=20,
                       shuffle=False,
                       drop_last=True)  # Miao used batch size 20

    val_set = ImageDataset(val_pixels,
                           val_label,
                           phase='val')
    valids = DataLoader(dataset=val_set,
                        batch_size=40,
                        shuffle=False,
                        drop_last=True)

    index = random.sample(list(range(len(val_pixels))), 40)
    fixed_data = [val_pixels[i] for i in index]
    fixed_label = [val_label[i] for i in index]

    fixed_val_set = ImageDataset(fixed_data,
                                 fixed_label,
                                 phase='val')
    fixed_valids = DataLoader(dataset=fixed_val_set,
                              batch_size=10,
                              shuffle=True,
                              drop_last=False)

    return users, groups, trainsep, valids, fixed_valids, tests
