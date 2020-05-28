import utils
import torch
import DenseNet

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# -------------------------------------------------------------------------------------
#   1st. Section - Preparations
#           In this section we
#               * define some constants
#               * download the Cifar-10 dataset & get training and testing datasets
#                   (we create the torch datasets out of the cifar-10 sets)
#               * define transformations to be applied to the dataset's pictures
#               * create training, validation and testing dataloaders
# -------------------------------------------------------------------------------------

print("1st. section:")
print("     Definition of constants, download of Cifar-10 datasets,")
print("     definition of torch transformations and creation of")
print("     training, validation and testing dataloaders")
# some constants...
cifar10_datdaset_path = './cifar10_dataset/'
batch_size = 64 # samples per batch
num_workers = 48 #number of subprocesses to use for data loading
valid_percentage = .20
nEpochs = 50
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# Download cifar 10 dataserts
utils.getCifar10Dataset(cifar10_datdaset_path)
# get the training and testing data sets.
# the labels on the testing data set are used to measure the accuracy on the resulting model
train_images, train_labels = utils.getTrainDatasets()
test_images, test_labels = utils.getTestDataset()

# we need to correctly  transpose the numpy arrays, so
# they can be converted to PIL images later while they are being
# drawn from the data loaders
train_images = train_images.transpose([0,2,3,1])
test_images = test_images.transpose([0,2,3,1])

# define transformations for the training images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# define transformations for the test iamges - which means only
# converting them to Tensors and normalizing them
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# create the cifar-10 dataset
x_train = utils.CifarDataset(train_images, train_labels, transform=transform)
x_test = utils.CifarDataset(test_images, test_labels, transform=test_transform)

# training indices to be used for validation purposes
indices = list(range(len(x_train)))
# split the training dataset into a -real- training dataset and validation set
np.random.shuffle(indices)
split = int(np.floor(len(x_train) * valid_percentage)) # we take 24 percent of the training dataset for validation
train_indices, valid_indices = indices[split:], indices[:split]

# define samplers - they sample/obtain batches from a list of indices.
# batches are built out of random sampled elements
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

# prepare loaders
train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
# the following test loader is used to load all the test images into the model for their classification
test_loader  = torch.utils.data.DataLoader(x_test, batch_size=batch_size, num_workers=num_workers)


# -------------------------------------------------------------------------------------
#   2nd. Section - DenseNet model & training
#           In this section we
#               * create a DenseNet model
#               * define a function loss & optimizer
#               * train the model
# -------------------------------------------------------------------------------------

print("2nd. section:")
print("     Create a DenseNet model, define a function loss")
print("     and optimizer and train the model")

# create a DenseNet model
modDN = DenseNet.Models.DenseNet([12,18,16], tlayer='H_layer', k=32, nClasses=10)
# create a loss function
crit = nn.CrossEntropyLoss()
# and the optimizer ...
optimizer = optim.SGD(modDN.parameters(), lr=0.01)
modDN.trainMe(modDN, crit, optimizer, nEpochs,
                bestModelName='best_modDN_SGD',
                lr_update_at_Epoch_perc=0.2,
                minLr_val_at_Epoch_perc=0.4,
                train_loader=train_loader,
                valid_loader=valid_loader
            )


# -------------------------------------------------------------------------------------
#   3rd. Section - Evaluate the best parameters gotten during training
# -------------------------------------------------------------------------------------

print("3rd. section:")
print("     Creata a new DenseNet model with the same architecture")
print("     as the last DenseNet model.")
print("     Load the best parameters produced during training and")
print("     evaluate them using a completely new image dataset")
print("     (i.e. the test dataset)")

# empty the GPU cache
torch.cuda.empty_cache()
# create a new model with the same architecture as the module used for training
eval_module = DenseNet.Models.DenseNet([12,18,16], tlayer="H_layer", k=32, nClasses=10)
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("Evaluaton of model will take place on CUDA!")
    # load the parameters trained
    eval_module.load_state_dict(torch.load('best_modDN_SGD.pt'))
    eval_module.cuda()
else:
    eval_module.load_state_dict(torch.load('best_modDN_SGD.pt',map_location=torch.device('cpu')))
# set the module for evaluation purposes
eval_module.eval()

# numpy array to hold all the test predictions
# (i.e. the name of the class an image belongs to)
predictions_classes = np.array([], dtype=str)
predictions_numbers = np.array([])

# draw batches of images from the test loader 
# until all of them have been drawn
test_labels = np.array([]) # labes will be concatenated in this array in the order they are drawn
for image_batch, labels in test_loader:
    image_batch = image_batch.cuda()
    # get predictions of the classes of the iamges
    # on this batch
    mod_output = eval_module(image_batch)
    _, batch_pred = torch.max(mod_output, 1)
    predictions_numbers = np.append(predictions_numbers, batch_pred.cpu())
    predictions_classes = np.append(predictions_classes, [classes[x] for x in batch_pred])

    test_labels = np.concatenate((test_labels, labels), axis=0)

# release the GPU memory
torch.cuda.empty_cache()

# print the accuracy gotten
equals = predictions_numbers == test_labels
accuracy = np.mean(equals)

print("\nAccuracy gotten after training: {:.4f}".format(accuracy))
