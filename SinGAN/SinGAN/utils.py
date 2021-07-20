import jax
from jax import lax, random, numpy as jnp
from jax import grad, jit, vmap
import flax
from flax import linen as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
#from skimage import morphology
#from skimage import filters
import os
import random as std_random
import pickle

def np2jax(x,opt):
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = jnp.asarray(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
   
    x = norm(x)
    return x


def jax2uint8(x):
    x = x[0,:,:,:]
    x = x.transpose((1,2,0))
    x = 255*denorm(x)
    x = np.array(x)
    x = x.astype(np.uint8)
    return x


def move_to_gpu(t):
    devices = jax.devices()
    if not str(devices[0]).startswith('gpu'):
        raise SystemError('GPU device not found')
    t = jax.device_put(t,device=devices[0])
    return t

def move_to_cpu(t):
    t = jax.device_put(t,device=jax.devices('cpu')[0])
    return t


def denorm(x):
    out = (x + 1) / 2
    return out.clip(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clip(-1, 1)