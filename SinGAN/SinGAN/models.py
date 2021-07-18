import numpy as np
import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn

import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Sequential(nn.Module):
  modules: List[nn.Module] = []

  def add_module(self, new_module:nn.Module) -> None:
    self.modules.add(new_module)

  def __call__(self, x):
    for module in modules:
        x = module(x)
    return x


class ActivationLayer(nn.Module):
    activation: Callable[[Array], Array]

    @nn.compact
    def __call__(self, x):
        return self.activation(x)


class ConvBlock(Sequential):
    out_channel: int
    ker_size: int
    padd: Union[str, Tuple[int, int]]
    stride: int
    dim: int
  
  def setup(self):
    self.ker_size=(self.ker_size for i in range(0,dim))
    self.stride=(self.stride for i in range(0,dim))
    self.padd = self.padd if isinstance(self.padd, str) else (self.padd for i in range(0,dim))
    self.add_module(nn.Conv(out_channel,kernel_size=self.ker_size,strides=self.stride,padding=self.padd, name='conv')),
    self.add_module(nn.BatchNorm(name="norm")),
    self.add_module(ActivationLayer(activation=lambda x : nn.leaky_relu(x, 0.2)))


class WDiscriminator(nn.Module):
    is_cuda: bool
    opt


    def setup(self):
        devices = jax.devices()
        if not str(devices[0]).startswith('gpu'):
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(devices[0]))

        N = int(opt.nfc)
        self.head = ConvBlock(N,kernel_size=opt.ker_size,strides=1,padding=opt.padd_size, dim=2)
        self.body = Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(N,opt.min_nfc),kernel_size=opt.ker_size,padding=opt.padd_size,strides=1, dim=2)
            self.body.add_module('block%d'%(i+1),block)
        padding = opt.padd_size if isinstance(opt.padd_size, str) else (opt.padd_size, opt.padd_size)
        self.tail = nn.Conv(1,kernel_size=(opt.ker_size, opt.ker_size),stride=(1,1),padding=padding)

    def (self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    opt

    def setup(self):
        devices = jax.devices()
        if not str(devices[0]).startswith('gpu'):
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(devices[0]))

        N = opt.nfc
        self.head = ConvBlock(N,kernel_size=opt.ker_size,padding=opt.padd_size,strides=1, dim=2) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(N,opt.min_nfc),kernel_size=opt.ker_size,padding=opt.padd_size,strides=1, dim=2)
            self.body.add_module('block%d'%(i+1),block)
        padding = opt.padd_size if isinstance(opt.padd_size, str) else (opt.padd_size, opt.padd_size)
        self.tail = Sequential(modules=[
            nn.Conv(opt.nc_im,kernel_size=(opt.ker_size, opt.ker_size),stride=(1,1),padding=padding),
            ActivationLayer(activation=jnp.tanh)]
        )
    def __call__(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y




