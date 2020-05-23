<p align="center">
  <a href="https://arxiv.org/abs/1608.06993" rel="noopener">
 <img width=350px height=320px src="./densenet-arch.png" alt="DenseNet architecture image"></a>
</p>

<h3 align="center">DenseNet - my implementation</h3>


---

<p align="center"> This is my own implementation of the <a href="https://arxiv.org/abs/1608.06993" rel="noopener">DenseNet</a>. I tried to make this implementation as close to the paper as my understanding allowed me. Please leave a comment and or suggestions (raise an issue)!
    <br> 
</p>

## Table of Contents

- [About](#about)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## About <a name = "about"></a>

After reading the [DenseNet paper](https://arxiv.org/abs/1608.06993) I was very surprised for it being so simple and yet so powerfull. So I decided to make my own implementation of it and give it a try.<br>
This implementation of DenseNet was done under python version 3.6.10.

I used [this Cifar-10 datase](https://www.kaggle.com/emadtolba/cifar10-comp) from kaggle to test the performance of my implementation. For that purpose, I trained a total of 4 DenseNet networks to the data; 2 of them were BC variants and the other to were none BC networks. The quick comparison I did on these four networks can be found in [this jupyter notebook](performance_Analysis/Cifar-10_performanceTest.ipynb).

A test script may be found [here](https://github.com/armhzjz/DenseNet/tree/master/tests/Cifar-10). I used this test sctipt to ensure the implementation works also out of the context of kaggle notebooks. It download the Cifar-10 dataset directly from its official webpage, prepares the training, validation and test data sets, trains a DenseNet model and evaluates it using the best parameters produced during its training. The execution of this script will take a considerable amount of time depending on the GPU hardware you use, so beware of this and don't get puzzled if it seems to take forever until the script is completely executed.

<br>Finally, a [kaggle kernel is found in here](https://www.kaggle.com/ahernandez1/mydensenet-implementation) in case the reader is interested in a cifar-10 evaluation (i.e. not only a rough comparison as the one provided in this repository).

## Content <a name = "content"></a>

* [DenseNet implementation as a Package](https://github.com/armhzjz/DenseNet/tree/master/DenseNet)
* [Cifar-10 training script (script used as test)](https://github.com/armhzjz/DenseNet/tree/master/tests/Cifar-10)

## Authors <a name = "authors"></a>

- [@armhzjz](https://github.com/armhzjz)

Any hint or advice on how this could be improved are welcome!

## Acknowledgements <a name = "acknowledgement"></a>

While implementing my DenseNet solution I stumble upon [this other DenseNet](https://github.com/weiaicunzai/pytorch-cifar100) implementation from [@weiaicunzai](https://github.com/weiaicunzai). Even though my implmentation was almost finish (and very alike I must admit), it was useful to have another reference apart from the paper.
