<p align="center">
  <a href="https://arxiv.org/abs/1608.06993" rel="noopener">
 <img width=350px height=320px src="./densenet-arch.png" alt="DenseNet architecture image"></a>
</p>

<h3 align="center">DenseNet - my implementation</h3>


---

<p align="center"> This is my own implementation of the <a href="https://arxiv.org/abs/1608.06993" rel="noopener">DenseNet</a>. I tried to make this implementation as close to the paper as my understanding allowed me. Please leave a comment and or suggestions (raise an issue)!
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## About <a name = "about"></a>

After reading the [DenseNet paper](https://arxiv.org/abs/1608.06993) I was very surprised for it being so simple and yet so powerfull. So I decided to make my own implementation of it and give it a try.<br>
I used [this Cifar-10 datase](https://www.kaggle.com/emadtolba/cifar10-comp) from kaggle to test the performance of my implementation. For that purpose, I trained a total of 4 DenseNet networks to the data; 2 of them were BC variants and the other to were none BC networks. The quick analysis I did on these four networks can be found in [this jupyter notebook](performance_Analysis/Cifar-10_performanceTest.ipynb).

## Authors <a name = "authors"></a>

- [@armhzjz](https://github.com/armhzjz)

Any hint or advice on how this could be improved are welcome!

## Acknowledgements <a name = "acknowledgement"></a>

While implementing my DenseNet solution I stumble upon [this other DenseNet](https://github.com/weiaicunzai/pytorch-cifar100) implementation from [@weiaicunzai](https://github.com/weiaicunzai). Even though my implmentation was almost finish (and very alike I must admit), it was useful to have another reference apart from the paper.
