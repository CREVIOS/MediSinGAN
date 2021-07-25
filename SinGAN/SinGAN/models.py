import numpy as np
import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from typing import Union, Tuple, Sequence, Callable, Any, List
from dataclasses import field

import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def ConvNd(out_channel, kernel_size, strides, padding, dim=2, name=None):
    kernel_size=[kernel_size for i in range(0,dim)]
    strides=[strides for i in range(0,dim)]
    padding = padding if isinstance(padding, str) else [[padding, padding] for i in range(0,dim)]

    return nn.Conv(out_channel,kernel_size=kernel_size,strides=strides,padding=padding, name='conv')


class Sequential(nn.Module):
    modules: List[nn.Module] # = field(default_factory=list)

    @nn.compact
    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x


class ActivationLayer(nn.Module):
    activation: Callable[[Any], Any]

    @nn.compact
    def __call__(self, x):
        return self.activation(x)

class ConvBlock(nn.Module):
    out_channel: int
    kernel_size: int
    padding: Union[str, Tuple[int, int]]
    strides: int
    dim: int
    
    @nn.compact
    def __call__(self, x):
        # print("Conv block", kernel_size, strides, padding)
        x = ConvNd(self.out_channel,kernel_size=self.kernel_size,strides=self.strides,padding=self.padding, dim=self.dim, name='conv')(x)
        x = nn.BatchNorm(name="norm", use_running_average=False)(x)
        x = nn.leaky_relu(x, 0.2)
        return x


class WDiscriminator(nn.Module):
    opt: Any


    def setup(self):
        N = int(self.opt.nfc)
        self.head = ConvBlock(N,kernel_size=self.opt.kernel_size,strides=1,padding=self.opt.padding, dim=2)
        body = []
        for i in range(self.opt.num_layer-2):
            N = int(self.opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(N,self.opt.min_nfc),kernel_size=self.opt.kernel_size,padding=self.opt.padding,strides=1, dim=2)
            body.append(block)
        self.body = Sequential(body)
        
        self.tail = ConvNd(1,kernel_size=self.opt.kernel_size,strides=1,padding=self.opt.padding)

    def __call__(self,x):
        x = x.transpose([0, 2, 3, 1])
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = x.transpose([0, 3, 1, 2])
#         print(f"Discriminator FINAL SHAPE: {x.shape}")
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    opt: Any

    def setup(self):

        N = self.opt.nfc
        self.head = ConvBlock(N,kernel_size=self.opt.kernel_size,padding=self.opt.padding,strides=1, dim=2) #GenConvTransBlock(self.opt.nc_z,N,opt.kernel_size,opt.padding,opt.stride)
        body = []
        for i in range(self.opt.num_layer-2):
            N = int(self.opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(N,self.opt.min_nfc),kernel_size=self.opt.kernel_size,padding=self.opt.padding,strides=1, dim=2)
            body.append(block)
        self.body = Sequential(body)
        self.tail = Sequential(modules=[
            ConvNd(self.opt.nc_im,kernel_size=self.opt.kernel_size,strides=1,padding=self.opt.padding),
            ActivationLayer(activation=jnp.tanh)]
        )
    def __call__(self,x,y):
        # TODO Change order of channels in code instead of doing transposition here
        x = x.transpose([0, 2, 3, 1])
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = x.transpose([0, 3, 1, 2])
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
#         print(f"X shape: {x.shape} , Y shape: {y.shape}")
        return x+y




