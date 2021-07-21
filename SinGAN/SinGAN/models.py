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


class Sequential(nn.Module):
    modules: List[nn.Module] # = field(default_factory=list)

    def __call__(self, x):
        for module in modules:
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
        kernel_size=[self.kernel_size for i in range(0,self.dim)]
        strides=[self.strides for i in range(0,self.dim)]
        padding = self.padding if isinstance(self.padding, str) else [[self.padding, self.padding] for i in range(0,self.dim)]
        print(kernel_size, strides, padding)
        x = nn.Conv(self.out_channel,kernel_size=kernel_size,strides=strides,padding=padding, name='conv')(x)
        x = nn.BatchNorm(name="norm")(x)
        x = nn.leaky_relu(x, 0.2)
        return x


class WDiscriminator(nn.Module):
    opt: Any


    def setup(self):
        N = int(self.opt.nfc)
        self.head = ConvBlock(N,kernel_size=self.opt.kernel_size,strides=1,padding=self.opt.padding_size, dim=2)
        body = []
        for i in range(self.opt.num_layer-2):
            N = int(self.opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(N,self.opt.min_nfc),kernel_size=self.opt.kernel_size,padding=self.opt.padding_size,strides=1, dim=2)
            body.append(block)
        self.body = Sequential(self.body)
        padding = self.opt.padding_size if isinstance(self.opt.padding_size, str) else (self.opt.padding_size, self.opt.padding_size)
        self.tail = nn.Conv(1,kernel_size=(self.opt.kernel_size, self.opt.kernel_size),strides=(1,1),padding=padding)

    def __call__(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    opt: Any

    def setup(self):
        devices = jax.devices()
        if not str(devices[0]).startswith('gpu'):
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(devices[0]))

        N = self.opt.nfc
        self.head = ConvBlock(N,kernel_size=self.opt.kernel_size,padding=self.opt.padding_size,strides=1, dim=2) #GenConvTransBlock(self.opt.nc_z,N,opt.kernel_size,opt.padding_size,opt.stride)
        body = []
        for i in range(self.opt.num_layer-2):
            N = int(self.opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(N,self.opt.min_nfc),kernel_size=self.opt.kernel_size,padding=self.opt.padding_size,strides=1, dim=2)
            body.append(block)
        self.body = Sequential(body)
        padding = self.opt.padding_size if isinstance(self.opt.padding_size, str) else (self.opt.padding_size, self.opt.padding_size)
        self.tail = Sequential(modules=[
            nn.Conv(self.opt.nc_im,kernel_size=(self.opt.kernel_size, self.opt.kernel_size),strides=(1,1),padding=padding),
            ActivationLayer(activation=jnp.tanh)]
        )
    def __call__(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y




