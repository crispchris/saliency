## -------------------
## --- Third-Party ---
## -------------------
import logging
import torch as t
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split

## -------------
## --- Class ---
## -------------
class Dataset:
    """Use for UCR Datasets
        For Torch Model Training
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data    ## data = [Num, Dim, Len]
        self.labels = labels

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, item):
        xt = self.data[item, :, :]
        y = self.labels[item]
        return xt, y


class TrainValSplit:
    """
    Use for Trainset and Validation Set split
    For the dataset which was already splitted into Train and Testset.
    """
    def __init__(self, dataset, val_train_split: float = 0.1, shuffle: bool =True):
        self.dataset = dataset
        dataset_size = len(dataset)
        random_seed = 42
        train_indices, val_indices = train_test_split(np.arange(dataset_size), random_state=random_seed,
                                                      shuffle=shuffle,
                                                      stratify=self.dataset.labels,
                                                      test_size=val_train_split)

        train_data = self.dataset.data[train_indices]
        train_labels = self.dataset.labels[train_indices]
        val_data = self.dataset.data[val_indices]
        val_labels = self.dataset.labels[val_indices]
        self.trainset = Dataset(train_data, train_labels)
        self.valset = Dataset(val_data, val_labels)

        _, train_classcounts = np.unique(self.trainset.labels, return_counts=True)
        _, val_classcounts = np.unique(self.valset.labels, return_counts=True)

        print(f"the Trainset size: {len(self.trainset)}")
        print(f"the number of each class in Trainset: {train_classcounts}")
        print(f"the Validation set size: {len(self.valset)}")
        print(f"the number of each class in Validation set: {val_classcounts}")

        weights = 1. / t.tensor(train_classcounts, dtype=t.float)
        sample_weights = weights[self.trainset.labels]
        self.trainsampler = WeightedRandomSampler(weights=sample_weights,
                                                  num_samples=len(self.trainset),
                                                  replacement=True)

    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader

    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        train_loader = t.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                                sampler=self.trainsampler,
                                                shuffle=False, num_workers=num_workers)
        return train_loader

    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        val_loader = t.utils.data.DataLoader(self.valset, batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)
        return val_loader


class DataSplit:
    """
    Data Split
    Split the dataset into Train, Validation, Test sets
    """
    def __init__(self, dataset, test_train_split: float = 0.2, val_train_split: float =0.1,
                 shuffle: bool =False, densely_labels: bool = False, majority_labels = None):
        """
        Parameters
        ----------
        dataset (torch.utils.data.Dataset) : Pytorch Dataset with (len, getitem) ...
        test_train_split (float) : the percentage for testset (first separate testset from trainset)
        val_train_split (float) : the percentage for validation set from testset (second split from trainset)
        shuffle (bool) : whether random shuffle or not
        densely_labels (bool) : whether the labels of the dataset are more than one in a sample
        majority_labels: the majority labels for each sample, used in densely labels data
        """
        self.dataset = dataset
        dataset_size = len(dataset)
        random_seed = 42
        if not densely_labels:
            train_indices, test_indices = train_test_split(np.arange(dataset_size), random_state=random_seed,
                                                           shuffle=shuffle,
                                                           stratify=self.dataset.labels,
                                                           test_size=test_train_split)
        else:
            train_indices, test_indices = train_test_split(np.arange(dataset_size), random_state=random_seed,
                                                           shuffle=shuffle,
                                                           stratify=majority_labels,
                                                           test_size=test_train_split)
        train_data = self.dataset.data[train_indices]
        train_labels = self.dataset.labels[train_indices]
        test_data = self.dataset.data[test_indices]
        test_labels = self.dataset.labels[test_indices]

        self.trainset = Dataset(train_data, train_labels)
        trainset_size = len(self.trainset)
        self.testset = Dataset(test_data, test_labels)

        if not densely_labels:
            train_indices, val_indices = train_test_split(np.arange(trainset_size), random_state=random_seed,
                                                          shuffle=shuffle,
                                                          stratify=self.trainset.labels,
                                                          test_size=val_train_split)
        else:
            majority_labels = []
            for i in range(len(self.trainset.labels)):
                values, counts = np.unique(self.trainset.labels[i], return_counts=True)
                pos = np.argmax(counts)
                majority_labels.append(values[pos])
            train_indices, val_indices = train_test_split(np.arange(trainset_size), random_state=random_seed,
                                                          shuffle=shuffle,
                                                          stratify=np.array(majority_labels),
                                                          test_size=val_train_split)
        train_data = self.trainset.data[train_indices]
        train_labels = self.trainset.labels[train_indices]
        val_data = self.trainset.data[val_indices]
        val_labels = self.trainset.labels[val_indices]

        self.trainset = Dataset(train_data, train_labels)
        self.valset = Dataset(val_data, val_labels)

    def get_split(self, batch_size=50, val_batch_size=100, num_workers=4):
        _, train_classcounts = np.unique(self.trainset.labels, return_counts=True)
        _, val_classcounts = np.unique(self.valset.labels, return_counts=True)
        _, test_classcounts = np.unique(self.testset.labels, return_counts=True)
        print(f"the Trainset size: {len(self.trainset)}")
        print(f"the number of each class in Trainset: {train_classcounts}")
        print(f"the Validation set size: {len(self.valset)}")
        print(f"the number of each class in Validation set: {val_classcounts}")
        print(f"the Test set size: {len(self.testset)}")
        print(f"the number of each class in Test set: {test_classcounts}")

        if self.trainset.labels[0].size < 2: ## only for windowed labeling
            weights = 1. / t.tensor(train_classcounts, dtype=t.float)
            sample_weights = weights[self.trainset.labels]
            self.trainsampler = WeightedRandomSampler(weights=sample_weights,
                                                      num_samples=len(self.trainset),
                                                      replacement=True)

        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=val_batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=val_batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        ## sampler only for windowed labeling
        ## not for densely labeling
        train_loader = t.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                               sampler=self.trainsampler if self.trainset.labels[0].size < 2 else None,
                                               shuffle=False, num_workers=num_workers)
        return train_loader

    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        val_loader = t.utils.data.DataLoader(self.valset, batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)
        return val_loader

    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        test_loader = t.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)
        return test_loader