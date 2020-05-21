import os
import shutil
import requests
import traceback
import tarfile
import pickle
import torch

import numpy as np

from torch import Tensor


cifar_dataset_path = "./cifar_dataset/"

def getCifar10Dataset(path=None):
    if path is not None:
        global cifar_dataset_path
        cifar_dataset_path = path

    # constants
    cifar_10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tarfile_name = 'cifar-10.tar.gz'

    # check if the directory that holds the dataset exists
    if os.path.isdir(cifar_dataset_path):
        # remove directory to make sure we start clean
        try:
            shutil.rmtree(cifar_dataset_path)
            # create directory
            os.mkdir(cifar_dataset_path)
        except:
            print("Deletion of folder not possible - something went wrong.")
            traceback.print_exc()
    else:
        # create directory
        os.mkdir(cifar_dataset_path)

    # download the cifar-10 dataset
    r = requests.get(cifar_10_url, allow_redirects=True)
    open(cifar_dataset_path + tarfile_name, 'wb').write(r.content)

    cifar_tar_file = tarfile.open(cifar_dataset_path + tarfile_name)
    cifar_tar_file.extractall(cifar_dataset_path)
    cifar_tar_file.close()

    # remove tar file
    os.remove(cifar_dataset_path + tarfile_name)

    # rearrange file structure
    dir_to_batches = cifar_dataset_path + 'cifar-10-batches-py/'
    files = os.listdir(dir_to_batches)

    for f in files:
        shutil.move(dir_to_batches+f, cifar_dataset_path)

    try:
        # remove 'cifar-10-batches-py' directory
        os.rmdir(dir_to_batches)
    except:
        print("Deletion of folder not possible - something went wrong.")
        traceback.print_exc()


    # remove .html file in "cifar_dataset_path"
    for f in os.listdir(cifar_dataset_path):
        if f.endswith('html'):
            os.remove(cifar_dataset_path + f)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def getTrainDatasets():
    # create an empty tensor that it's going to be used to concatenate
    # on it all the training images
    train_images = torch.zeros(0,3,32,32)
    train_labels = np.array([], dtype=int)
    for i in range(1,6):
        # unpickle the 'data_batch'.py in turn
        # and convert it to a torch.Tensor
        dicc = unpickle(cifar_dataset_path + 'data_batch_{}'.format(i))
        # the dicc list has a shape of (10000,3072), so it needs to be reshaped
        dict_tensor = torch.Tensor(dicc[b'data']).view(10000,3,32,32)
        train_images = torch.cat((train_images, dict_tensor), dim=0)
        train_labels = np.concatenate((train_labels, dicc[b'labels']), axis=0)
    
    return train_images, train_labels


def getTestDataset():
         
    # now unpickle the test batch and process it as it was done for a training batch
    tdicc = unpickle(cifar_dataset_path + 'test_batch')
    test_images = torch.Tensor(tdicc[b'data']).view(10000,3,32,32)
    test_labels = np.array(tdicc[b'labels'])

    return test_images, test_labels
