import utils
import torch


cifar10_datdaset_path = './cifar10_dataset/'


# Download cifar 10 dataserts
utils.getCifar10Dataset(cifar10_datdaset_path)
# get the training and testing data sets.
# the labels on the testing data set are used to measure the accuracy on the resulting model
train_images, train_labels = utils.getTrainDatasets()
test_images, test_labels = utils.getTestDataset()

print("training shape is: {}".format(train_images.shape))
print("training labels: {}".format(train_labels.shape))
