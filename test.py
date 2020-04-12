import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import numpy as np
import os
import zipfile
import math
import collections


class MyDict(dict):
    def __missing__(self, key):
        return self['UNK']


class CBOWDataset(Dataset):
    def __init__(self, filename, window_size, vocabulary_size=None):
        handle = open(filename, "r")
        lines = handle.readlines()  # read ALL the lines!corpus
        self.corpus = [w for l in lines for w in l.split()]
        self.window_size = window_size
        self.num_corpus = len(self.corpus)
        self.corpus_words = set(self.corpus)
        if vocabulary_size is not None:
            c = collections.Counter()
            for w in self.corpus_words:
                c[w] += 1
            self.corpus_words = [w for w, _ in c.most_common(vocabulary_size)]
        self.corpus_words = sorted(self.corpus_words)
        self.corpus_words.insert(0, 'UNK')
        self.num_corpus_words = len(self.corpus_words)
        self.word2Ind = MyDict({w: i for i, w in enumerate(self.corpus_words)})

    def __getitem__(self, i):
        x = np.zeros(self.num_corpus_words, dtype=np.int64)
        for nb in range(np.maximum(i - self.window_size, 0), i):
            x[self.word2Ind[self.corpus[nb]]] = 1 / (2 * self.window_size)
        for n in range(i + 1, np.minimum(i + self.window_size + 1, self.num_corpus)):
            x[self.word2Ind[self.corpus[n]]] = 1 / (2 * self.window_size)

        # y = np.zeros(self.num_corpus_words, dtype=np.int64)
        y = self.word2Ind[self.corpus[i]]
        return x, y

    def __len__(self):
        return self.num_corpus

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor()
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cbow_train = CBOWDataset('text8.txt', 4)
loader_train = DataLoader(cbow_train, batch_size=64,
                          sampler=sampler.SubsetRandomSampler(range(int(len(cbow_train)*0.8))))
loader_val = DataLoader(cbow_train, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(int(len(cbow_train)*0.8), int(len(cbow_train)*0.9))))
loader_test=DataLoader(cbow_train, batch_size=64,
                       sampler=sampler.SubsetRandomSampler(range(int(len(cbow_train)*0.9), len(cbow_train))))
USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)
def flatten(x):
    ################################################################################
    # TODO: Implement flatten function.                                            #
    ################################################################################
    x_flat = x.flatten(start_dim=1)
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    return x_flat
def check_accuracy_part34(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(flatten(x))
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            print("t x y:", t, x.shape, y.shape)
            scores = model(flatten(x))
            print("scores:", scores.shape)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                #check_accuracy_part34(loader_val, model)
                print()

# We need to wrap `flatten` function in a module in order to stack it
# in nn.Sequential
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

hidden_layer_size = 100
learning_rate = 1e-2

########################################################################
# TODO: use nn.Sequential to make the same network as before           #
########################################################################
model = nn.Sequential(
          Flatten(),
          nn.Linear(cbow_train.num_corpus_words,hidden_layer_size),
          nn.ReLU(),
          nn.Linear(hidden_layer_size,cbow_train.num_corpus_words)
        )
########################################################################
#                          END OF YOUR CODE                            #
########################################################################

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_part34(model, optimizer)