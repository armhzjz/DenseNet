import torch
import torch.nn as nn


""" Pooling layers. The concatenation operation used in 
    Eq. (2) is not viable when the size of feature-maps changes.
    However, an essential part of convolutional networks is
    down-sampling layers that change the size of feature-maps.
    To facilitate down-sampling in our architecture we divide 
    the network into multiple densely connected dense blocks;
    see Figure 2. We refer to layers between blocks as transition
    layers, which do convolution and pooling. The transition
    layers used in our experiments consist of a batch normalization
    layer and an 1 x 1 convolutional layer followed by a
    2 x 2 average pooling layer. """
class Transition_layer(nn.Module):
    def __init__(self, chann_in, compression_factor=1):
        # constructor of the class
        super().__init__()
        """ Compression. To further improve model compactness,
            we can reduce the number of feature-maps at transition
            layers. If a dense block contains m feature-maps, we let
            the following transition layer generate [θm] output feature-
            maps, where 0 < θ ≤ 1is referred to as the compression fac-
            tor.  When θ= 1, the number of feature-maps across transi-
            tion layers remains unchanged.  We refer the DenseNet with
            θ < 1 as DenseNet-C, and we set θ = 0.5 in our experiment.
            When both the bottleneck and transition layers with θ < 1
            are used, we refer to our model as DenseNet-BC. """
        chann_out = int(chann_in * compression_factor)
        self.__Transition_layer = nn.Sequential(
            nn.BatchNorm2d(chann_in),
            nn.Conv2d(
                chann_in, chann_out, kernel_size=1, stride=1, padding=0, bias=True
                    ),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        
    def forward(self,x):
        return self.__Transition_layer(x)


""" Bottleneck layers. Although each layer only produces k
    output feature-maps, it typically has many more inputs. It
    has been noted in [37, 11] that a 1 x 1 convolution can be introduced
    as bottleneck layer before each 3 x 3 convolution
    to reduce the number of input feature-maps, and thus to
    improve computational efficiency. We find this design especially
    effective for DenseNet and we refer to our network
    with such a bottleneck layer, i.e., to the BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)
    vresion of H_l, as DensNet-B. In our experiments, we let each 1x1
    convolution produce 4k feature-maps (where k = Growth rate). """
class Bottleneck(nn.Module):
    """ Bottleneck layer is an exclusive layer
        of DenseNet-B - a version of DenseNet. """
    def __init__(self, chann_in, growth_rate):
        # constructor of the class
        super().__init__()
        self.__Bottleneck = nn.Sequential(
            nn.BatchNorm2d(chann_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(chann_in, 4 * growth_rate, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1,
                    padding_mode='zeros', bias=True)
        )
        
        
    def forward(self, x):
        return torch.cat([x, self.__Bottleneck(x)], 1)


""" Composite function.Motivated by [12], we define H_l(·)
    as  a  composite  function  of  three  consecutive  operations:
    batch normalization (BN) [14], followed by a rectified lin-
    ear unit (ReLU) [6] and a3×3convolution (Conv). """
class H_layer(nn.Module):
    """ Composite function. This layer is used always when
        DenseNet-B is not. """
    def __init__(self, chann_in, growth_rate):
        # constructor of the class
        super().__init__()
        self.h = nn.Sequential(
            nn.BatchNorm2d(chann_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(chann_in, growth_rate, kernel_size=3, stride=1, padding=1,
                    padding_mode='zeros', bias=True)
        )
    
    def forward(self, x):
        return torch.cat([x, self.h(x)], 1)