import sys
import torch
import numpy as np
import torch.nn as nn
from DenseNet import Layers as lyr
from DenseNet import Training




class DenseNet(nn.Module):
    def __init__(self, nLayers, tlayer="Bottleneck", k=32, compression_factor=0.5, nClasses=100, default_trainer=True):
        super(DenseNet, self).__init__()
        
        # validate parameters...
        nLayers, layer, self.compression_factor = self.__Validate_params(nLayers, tlayer, compression_factor)
        # add a function pointer to the package's default training method if required
        self.trainMe = Training.TrainModel if default_trainer else lambda: print("This instance doesn't use a default trainer")

        # create the main sequentail module
        self.densenet = nn.Sequential()

        # Before entering the first dense block, a convolution with 16 (or twice the growth rate for DenseNet-BC)
        # ouput channels is performed on the input images.
        preprocess_outmaps = 2 * k if (layer is lyr.Bottleneck and self.compression_factor < 1.) else 16
        self.densenet.add_module("preprocessInput", nn.Conv2d(3, preprocess_outmaps, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True))

        # create the dense blocks according to the size of the 'nLayers' list
        # I define - for clarity/readability reasons - a 'innerChanns' variable whose 
        # value is initialized to 'preprocess_outmaps'
        innerChanns = preprocess_outmaps
        
        for indx in range(len(nLayers) - 1): # we skip here the creation of the last dense block ....
                
            # build a dense block with the number of layers according
            # to the index 'indx' of the 'nLayers' list
            locals()['DenseBlock_{}'.format(indx)] = nn.Sequential()
            for f in range(nLayers[indx]):
                locals()['DenseBlock_{}'.format(indx)].add_module('H{}'.format(f), layer(innerChanns, k))
                innerChanns += k
            
            # add the just built dense block to the main sequential module (i.e. densenet)
            self.densenet.add_module('DN_block{}'.format(indx), locals()['DenseBlock_{}'.format(indx)])
            
            """ We use (...) transition layers between two contiguous dense blocsk """
            # add a transition layer right after a dense block - do not forget to explicitly add the compression factor argument!
            self.densenet.add_module('TransitionLayer_{}'.format(indx), lyr.Transition_layer(innerChanns, self.compression_factor))
            # update the number of input feature maps of the next Dense Block
            innerChanns = int(innerChanns * self.compression_factor)

        # create and add the last dense block. This last dense block was previously left aside because
        # after this last dense block comes no transition layer. Instead a global average pooling
        # takes place together with a fully connected network performing a softmax classifier.
        locals()['DenseBlock_{}'.format(len(nLayers) - 1)] = nn.Sequential()
        for f in range( nLayers[len(nLayers) - 1] ):
            locals()['DenseBlock_{}'.format(len(nLayers) - 1)].add_module('H{}'.format(f), layer(innerChanns, k))
            innerChanns += k
        # add the just built dense block to the main sequential module (i.e. densenet)
        self.densenet.add_module('DN_block{}'.format(len(nLayers) - 1), locals()['DenseBlock_{}'.format(len(nLayers) - 1)])

        """ At the end of the last dense block, a global average pooling is performed
            and then a softmax classifier is attached. """
        # With adaptive pooling the output can be reduced to any feature map size,
        # although in practice it is often choosen size 1, in which case
        # it does the same thing as global pooling
        # - but first a batch and relu layer (I included this two layers after checking the implementation 
        # I refer to at the begining of this notebook. I checked the evaluation loss without these layer (first)
        # and with these layers (after) and it works best with them.
        preSoftmax_layer = nn.Sequential(
            nn.BatchNorm2d(innerChanns),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.densenet.add_module('preSoftmax_layer', preSoftmax_layer)
        
        # a linear transformation is used here as a Softmax layer. 
        self.fakeSoftmax = nn.Linear(innerChanns, nClasses, bias=True)

        # initialize all weights and biases
        self.densenet.apply(self.__InitW_uniCenter)
        self.fakeSoftmax.apply(self.__InitW_uniCenter)
        
        
    def forward(self, x):
        y = self.densenet(x)
        y = y.view(y.size()[0], -1)
        return self.fakeSoftmax(y)


    def __InitW_uniCenter(self, m):
        """ General rule for setting the weights in a neural network is to set
            them to be close to zero without being too small. A uniform gaussian
            distribution centered at zero is used towards this end. """
        classname = m.__class__.__name__
        # for every linear layer in a model ...
        if classname.find('Linear') != -1:
            # get the number of inputs
            n = m.in_features
            y = 1. / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)
        
        
    def __Validate_params(self, nLayers, tlayer, compression_factor):
        # validate the parameters given to the main class creator, to
        # ensure a minimum degree of sane functionality

        # check for the type of layer to be used....
        if tlayer == "Bottleneck" or tlayer is None:
            layer = lyr.Bottleneck
            # save the compression factor value - needed to further build the network
            real_compression_factor = compression_factor
        elif tlayer == "H_layer":
            layer = lyr.H_layer
            # save the compression factor value - needed to further build the network
            if compression_factor < 1:
                print("Compression factor smaller than 1.0 is exclusive of DenseNet BC.")
                print("Compression factor has been set to 1.0")
                real_compression_factor = 1.0
            else:
                real_compression_factor = compression_factor
        else:
            print("Layer type not supported in DenseNet.")
            print("Must be either 'Bottleneck' of 'H_layer'")
            print("For mor information, refer to the DenseNet paper:")
            print("     https://arxiv.org/abs/1608.06993v5")
            sys.exit(1)
            
        # check that nLayers is either of type int or list
        # if nLayer is of type list, check that it is not empty
        if isinstance(nLayers, int):
            nLayers = [nLayers]
        elif isinstance(nLayers, list) and 0 < len(nLayers):
            nLayers = nLayers
        else:
            print("nLayer must be an int or a list containing the")
            print("number of layers to be created per dense block.")
            print("If a list is given as argument, so many dense blocks")
            print("will be created as elements on the list.")
            sys.exit(1)
        
        return nLayers, layer, real_compression_factor




def DenseNet121():
    return DenseNet([6,12,24,16], "Bottleneck", k=32)

def DenseNet169():
    return DenseNet([1,12,32,32], "Bottleneck", k=32)

def DenseNet201():
    return DenseNet([6,12,48,32], "Bottleneck", k=32)

def DenseNet161():
    return DenseNet([6,12,36,24], "Bottleneck", k=48)
