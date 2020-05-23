## About this DenseNet implementation

After reading the [DenseNet paper](https://arxiv.org/abs/1608.06993) I was very surprised for it being so simple and yet so powerfull. So I decided to make my own implementation of it and give it a try.<br>
This implementation of DenseNet was done under python version 3.6.10.

I used [this Cifar-10 datase](https://www.kaggle.com/emadtolba/cifar10-comp) from kaggle to test the performance of my implementation. For that purpose, I trained a total of 4 DenseNet networks to the data; 2 of them were BC variants and the other to were none BC networks. The quick comparison I did on these four networks can be found in [this jupyter notebook](https://github.com/armhzjz/DenseNet/blob/master/performance_Analysis/Cifar-10_performanceTest.ipynb).

A test script may be found [here](https://github.com/armhzjz/DenseNet/tree/master/tests/Cifar-10). I used this test sctipt to ensure the implementation works also out of the context of kaggle notebooks. It downloads the Cifar-10 dataset directly from its official webpage, prepares the training, validation and test data sets, trains a DenseNet model and evaluates it using the best parameters produced during its training. The execution of this script will take a considerable amount of time depending on the GPU hardware you use, so beware of this and don't get puzzled if it seems to take forever until the script is completely executed.

<br>Finally, a [kaggle kernel is found in here](https://www.kaggle.com/ahernandez1/mydensenet-implementation) in case the reader is interested in a cifar-10 evaluation (i.e. not only a rough comparison as the one provided in this repository).

Note: In order to use this module, pytorch must be install in your system.