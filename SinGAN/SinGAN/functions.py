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
from SinGAN.imresize import imresize
import os
import random as std_random
from sklearn.cluster import KMeans
import pickle
from SinGAN.utils import *
from functools import partial
# from SinGAN.stopwatch import StopwatchPrint


# custom weights initialization called on netG and netD



#def denorm2image(I1,I2):
#    out = (I1-I1.mean())/(I1.max()-I1.min())
#    out = out*(I2.max()-I2.min())+I2.mean()
#    return out#.clamp(I2.min(), I2.max())

#def norm2image(I1,I2):
#    out = (I1-I2.mean())*2
#    return out#.clamp(I2.min(), I2.max())

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = np.asarray(inp.transpose((1,2,0)))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = np.asarray(inp.transpose((0,1)))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    return inp

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        #ax.imshow(convert_image_np(real_cpu[0,:,:,:].cpu()))
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
    # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    # inp = std*
    return inp

# TODO
# @jax.jit
def generate_noise(key,size,num_samp=1,scale=1):
    key, subkey = random.split(key)
    
    noise = random.normal(subkey, shape=(num_samp, size[0], jnp.round(size[1]/scale).astype(int), jnp.round(size[2]/scale).astype(int)))
#     print("1", noise.shape)
    if scale != 1:
        noise = noise.transpose([0, 2, 3, 1])

        noise = upsampling(noise, scale, scale)


        noise = noise.transpose([0, 3, 1, 2])
#     print("2", noise.shape)

    return noise, key

def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    #plt.title('loss')
    #plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)



def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

@jax.jit
def apply_state(state, *args, params=None):
        res, params = state.apply_fn({'params': state.params if params is None else params, 'batch_stats': state.batch_stats}, *args, mutable=['batch_stats'])
        state.replace(batch_stats=["batch_stats"])
    return res, state

@jax.jit
def calc_gradient_penalty(paramsD, stateD, key, real_data, fake_data, LAMBDA):
    #print real_data.size()
    key, subkey = random.split(key)
    alpha = random.uniform(subkey)
#     print(fake_data)
#     print("fake shape", fake_data.shape)
#     print("real shape", real_data.shape)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    
    def func(interpolates):
        val, _ = stateD.apply_fn({'params': paramsD, 'batch_stats': stateD.batch_stats}, interpolates, mutable=['batch_stats'])
        return val
    # TODO
    disc_interpolates = func(interpolates)
    gradients = jax.jvp(func, (interpolates,), (jnp.ones(interpolates.shape),))
    gradients = gradients[0]
#     print(gradients.shape)
#     gradients, stateD = jax.grad(partial(stateD.apply_fn, mutable=['batch_stats']), argnums=1, has_aux=True)({'params': paramsD, 'batch_stats': stateD.batch_stats}, interpolates)
    #LAMBDA = 1
    gradient_penalty = ((jax.numpy.linalg.norm(gradients, ord=2, axis=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty, stateD, key

def read_image(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = np2jax(x,opt)
    x = x[:,0:3,:,:]
    return x

def read_image_dir(dir,opt):
    x = img.imread('%s' % (dir))
    x = np2jax(x,opt)
    x = x[:,0:3,:,:]
    return x


def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = x[:, :, 0:3]
    return x

def save_networks(paramsG,paramsD,z,opt):
    pickle_save(paramsG, '%s/netG.pth' % (opt.outf))
    pickle_save(paramsD, '%s/netD.pth' % (opt.outf))
    pickle_save(z, '%s/z_opt.pth' % (opt.outf))

def adjust_scales2image(real_,opt):
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def adjust_scales2image_SR(real_,opt):
    opt.min_size = 18
    opt.num_scales = int((math.log(opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor_init))) + 1
    scale2stop = int(math.log(min(opt.max_size , max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def creat_reals_pyramid(real,reals,opt):
    real = real[:,0:3,:,:]
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize(real,scale,opt)
        reals.append(curr_real)
    return reals

def pickle_load(filepath):
    with open(filepath, 'rb') as fp:
        fl = pickle.load(fp)
    return fl

def pickle_save(obj,filepath):
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp)

def load_trained_pyramid(opt, mode_='train'):
    #dir = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor_init)
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    if(os.path.exists(dir)):
        Gs = pickle_load('%s/Gs.pth' % dir)
        Zs = pickle_load('%s/Zs.pth' % dir)
        reals = pickle_load('%s/reals.pth' % dir)
        NoiseAmp = pickle_load('%s/NoiseAmp.pth' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def generate_in2coarsest(reals,scale_v,scale_h,opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v, scale_h)
    if opt.gen_start_scale == 0:
        in_s = jnp.zeros(real_down.shape)
    return in_s

def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.alpha)
    elif (opt.mode == 'animation_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
    elif (opt.mode == 'paint_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.paint_start_scale)
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out,opt.input_name[:-4], opt.gen_start_scale)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
    elif opt.mode == 'animation':
        dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    elif opt.mode == 'SR':
        dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    elif opt.mode == 'harmonization':
        dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'editing':
        dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'paint2image':
        dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
        if opt.quantization_flag:
            dir2save = '%s_quantized' % dir2save
    return dir2save

def post_config(opt):
    # init fixed parameters
    
    platform = "cpu" if opt.not_cuda else "gpu"
    jax.config.update('jax_platform_name', platform)
    devices = jax.devices(backend=platform)
    
    opt.device = devices[0]
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100

    
    if opt.manualSeed is None:
        opt.manualSeed = std_random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    opt.PRNGKey = random.PRNGKey(opt.manualSeed)
    std_random.seed(opt.manualSeed)
    if str(devices[0]).startswith('gpu') and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def calc_init_scale(opt):
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale,iter_num

def quant(prev,device):
    arr = prev.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    x = centers[labels]
    x = jnp.asarray(x)
    x = move_to_gpu(x)
    x = x.view(prev.shape)
    return x,centers

def quant2centers(paint, centers):
    arr = paint.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
    labels = kmeans.labels_
    #centers = kmeans.cluster_centers_
    x = centers[labels]
    x = jnp.asarray(x)
    x = move_to_gpu(x)
    x = x.view(paint.shape)
    return x

    return paint


def dilate_mask(mask,opt):
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)
    mask = jax2uint8(mask)
    mask = mask[:,:,0]
    mask = morphology.binary_dilation(mask,selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2jax(mask,opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    plt.imsave('%s/%s_mask_dilated.png' % (opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask), vmin=0,vmax=1)
    mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask








@jax.jit
def interpolate_bilinear(im, rows, cols):

  # based on http://stackoverflow.com/a/12729229
  col_lo = jnp.floor(cols).astype(int)
  col_hi = col_lo + 1
  row_lo = jnp.floor(rows).astype(int)
  row_hi = row_lo + 1

  nrows, ncols = im.shape[-3:-1]
  def cclip(cols): return jnp.clip(cols, 0, ncols - 1)
  def rclip(rows): return jnp.clip(rows, 0, nrows - 1)
  Ia = im[..., rclip(row_lo), cclip(col_lo), :]
  Ib = im[..., rclip(row_hi), cclip(col_lo), :]
  Ic = im[..., rclip(row_lo), cclip(col_hi), :]
  Id = im[..., rclip(row_hi), cclip(col_hi), :]

  wa = np.expand_dims((col_hi - cols) * (row_hi - rows), -1)
  wb = np.expand_dims((col_hi - cols) * (rows - row_lo), -1)
  wc = np.expand_dims((cols - col_lo) * (row_hi - rows), -1)
  wd = np.expand_dims((cols - col_lo) * (rows - row_lo), -1)

  return wa*Ia + wb*Ib + wc*Ic + wd*Id

@jax.jit
def upsampling(img, sx, sy):
   nrows, ncols = img.shape[-3:-1]
   delta_x = 0.5/sx
   delta_y = 0.5/sy

   rows = np.linspace(delta_y,nrows-delta_y, jnp.int32(sy*nrows))
   cols = np.linspace(delta_x,ncols-delta_x, jnp.int32(sx*ncols))
   ROWS, COLS = np.meshgrid(rows,cols,indexing='ij')
   
   img_resize_vec = interpolate_bilinear(img, ROWS.flatten(), COLS.flatten())
   img_resize =  img_resize_vec.reshape(img.shape[:-3] + 
                                        (len(rows),len(cols)) + 
                                       img.shape[-1:])
   
   return img_resize
