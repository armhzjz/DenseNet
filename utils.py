import os
import shutil
import requests
import traceback
import tarfile
import pickle
import torch

from torch import Tensor


cifar_dataset_path = "./cifar_dataset/"

def getCifar10Dataset():
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


def getTrainDataset(path=None):
    if path is not None:
        global cifar_dataset_path
        cifar_dataset_path = path

    # download the cifar10 dataset
    getCifar10Dataset()

    # create the tensor we will return
    #train_images = Tensor()
    images_list = [[[Tensor(x).view(3,32,32) for x in dicc[b'data']] for dicc in \
                    unpickle(cifar_dataset_path + 'data_batch_{}'.format(i))] for i in range(1,6)]
    # for i in range(1,6):
    #     dicc = unpickle(cifar_dataset_path + 'data_batch_{}'.format(i))
    #     images_list.append([x for x in dicc[b'data']])
    #     #train_images = torch.cat([train_images, [Tensor(x).view(3,32,32) for x in dicc[b'data']]], 0)

    return images_list
