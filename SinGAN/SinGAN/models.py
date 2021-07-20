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
  modules: list = field(default_factory=lambda: [])

  def add_module(self, new_module:nn.Module) -> None:
    print(self.modules)
    self.modules.append(new_module)

  def __call__(self, x):
    for module in modules:
        x = module(x)
    return x


class ActivationLayer(nn.Module):
    activation: Callable[[Any], Any]

    @nn.compact
    def __call__(self, x):
        return self.activation(x)

class ConvBlockBase(nn.Module):
    out_channel: int
    kernel_size: int
    padding: Union[str, Tuple[int, int]]
    strides: int
    dim: int

class ConvBlock(Sequential, ConvBlockBase):
    # out_channel: int
    # kernel_size: int
    # padding: Union[str, Tuple[int, int]]
    # stride: int
    # dim: int
  
    def setup(self):
        self.kernel_size=(self.kernel_size for i in range(0,dim))
        self.strides=(self.strides for i in range(0,dim))
        self.padding = self.padding if isinstance(self.padding, str) else (self.padding for i in range(0,dim))
        self.add_module(nn.Conv(out_channel,kernel_size=self.kernel_size,strides=self.strides,padding=self.padding, name='conv')),
        self.add_module(nn.BatchNorm(name="norm")),
        self.add_module(ActivationLayer(activation=lambda x : nn.leaky_relu(x, 0.2)))


class WDiscriminator(nn.Module):
    is_cuda: bool
    opt: Any


    def setup(self):
        devices = jax.devices()
        if not str(devices[0]).startswith('gpu'):
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(devices[0]))

        N = int(self.opt.nfc)
        self.head = ConvBlock(N,kernel_size=self.opt.kernel_size,strides=1,padding=self.opt.padding_size, dim=2)
        self.body = Sequential()
        for i in range(self.opt.num_layer-2):
            N = int(self.opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(N,self.opt.min_nfc),kernel_size=self.opt.kernel_size,padding=self.opt.padding_size,strides=1, dim=2)
            self.body.add_module(block)
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
        self.body = Sequential()
        for i in range(self.opt.num_layer-2):
            N = int(self.opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(N,self.opt.min_nfc),kernel_size=self.opt.kernel_size,padding=self.opt.padding_size,strides=1, dim=2)
            self.body.add_module(block)
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




