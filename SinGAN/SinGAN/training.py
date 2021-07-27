import SinGAN.functions as functions
from SinGAN.functions import apply_state
import SinGAN.models as models
import os

import jax
from jax import lax, random, numpy as jnp
from jax import grad, jit, vmap
import optax
import flax
from flax import linen as nn
from flax.training import train_state
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize
from typing import Any
from SinGAN.stopwatch import StopwatchPrint

class TrainState(train_state.TrainState):
    batch_stats: Any

def train(opt,Gs,Zs,reals,NoiseAmp):
    real_ = functions.read_image(opt)
    in_s = 0
    scale_num = 0
    real = imresize(real_,opt.scale1,opt)
    with StopwatchPrint("Creating pyramid..."):
        reals = functions.creat_reals_pyramid(real,reals,opt)
    
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        D_curr, D_params, G_curr, G_params = init_models(opt,reals[scale_num].shape)
#         print(D_curr, D_params)
#         print(G_curr, G_params)
        if (nfc_prev==opt.nfc):
            G_params = pickle_load('%s/%d/netG.pth' % (opt.out_,scale_num-1))
            D_params = pickle_load('%s/%d/netD.pth' % (opt.out_,scale_num-1))
        with StopwatchPrint("Train single scale..."):
            z_curr,in_s,G_curr = train_single_scale(D_curr,D_params, G_curr, G_params,reals,Gs,Zs,in_s,NoiseAmp,opt)

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)
        with StopwatchPrint("Saving models..."):
            functions.pickle_save(Zs, '%s/Zs.pth' % (opt.out_))
            functions.pickle_save(Gs, '%s/Gs.pth' % (opt.out_))
            functions.pickle_save(reals, '%s/reals.pth' % (opt.out_))
            functions.pickle_save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def step_stateD(in_stateD, fake, real, prev, opt):
    @jax.jit
    def discriminator_loss(params):
        
        output, stateD = apply_state(in_stateD, real)

        errD_real = -output.mean()#-a
        D_x = -errD_real


        errD_fake = output.mean()
        D_G_z = output.mean()
       
        gradient_penalty, stateD, new_PRNGKey = functions.calc_gradient_penalty(params, stateD, opt.PRNGKey, real, fake, opt.lambda_grad)

        return errD_real + errD_fake + gradient_penalty, (stateD, new_PRNGKey)
    with StopwatchPrint("apply disc"):
        (errD, (stateD, opt.new_PRNGKey)), grads = jax.value_and_grad(discriminator_loss, has_aux=True)(in_stateD.params)
    with StopwatchPrint("apply grads"):
        stateD = stateD.apply_gradients(grads=grads, batch_stats=stateD.batch_stats)
    return errD, stateD

def step_stateG(in_stateG, z_opt, z_prev, output, alpha, real, opt):
    @jax.jit
    def rec_loss(params):
        if alpha!=0:
            mse_loss = lambda x,y: jnp.mean((x-y)**2)
            Z_opt = opt.noise_amp*z_opt+z_prev
            rec, out_stateG = apply_state(in_stateG, Z_opt, z_prev, params=params)
            rec_loss = alpha*mse_loss(rec,real)        
        else:
            Z_opt = z_opt
            rec_loss = alpha
        return rec_loss, (out_stateG, Z_opt)
    (rec_loss_val, (stateG, Z_opt)), grads = jax.value_and_grad(rec_loss, has_aux=True)(in_stateG.params)
    stateG = stateG.apply_gradients(grads=grads, batch_stats=stateG.batch_stats)
    return rec_loss_val, stateG, Z_opt

def train_single_scale(netD,paramsD,netG,paramsG,reals,Gs,Zs,in_s,NoiseAmp,opt,centers=None):

    real = reals[len(Gs)]
    opt.nzx = real.shape[2]#+(opt.kernel_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]#+(opt.kernel_size-1)*(opt.num_layer)
    opt.receptive_field = opt.kernel_size + ((opt.kernel_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.kernel_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.kernel_size - 1) * opt.num_layer) / 2)
    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2]+(opt.kernel_size-1)*(opt.num_layer)
        opt.nzy = real.shape[3]+(opt.kernel_size-1)*(opt.num_layer)
        pad_noise = 0
    m_noise = lambda arr: (jnp.pad(arr, ((0,0), (0,0), (pad_noise,pad_noise), (pad_noise,pad_noise)))) 
    m_image = lambda arr: (jnp.pad(arr, ((0,0), (0,0), (pad_image,pad_image), (pad_image,pad_image))))

    alpha = opt.alpha
    fixed_noise,opt.PRNGKey = functions.generate_noise(opt.PRNGKey,[opt.nc_z,opt.nzx,opt.nzy])
    z_opt = jnp.zeros(fixed_noise.shape)
    z_opt = m_noise(z_opt)
    
    # setup optimizer
#     optimizerD = flax.optim.Adam(learning_rate=opt.lr_d, beta1=opt.beta1, beta2=0.999).create(paramsD)
#     optimizerG = flax.optim.Adam(learning_rate=opt.lr_g, beta1=opt.beta1, beta2 = 0.999).create(paramsG)
    
    optimizerD = optax.adam(learning_rate=opt.lr_d, b1=opt.beta1, b2=0.999)
    optimizerG = optax.adam(learning_rate=opt.lr_g, b1=opt.beta1, b2 = 0.999)


    stateD = TrainState.create(
    apply_fn=netD.apply, params=paramsD["params"], batch_stats=paramsD["batch_stats"], tx=optimizerD)
    
    stateG = TrainState.create(
    apply_fn=netG.apply, params=paramsG["params"], batch_stats=paramsG["batch_stats"], tx=optimizerG)
    
    # TODO
    # schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    # schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    with StopwatchPrint("Training..."):
        for epoch in range(opt.niter):
            if (Gs == []):
                z_opt,opt.PRNGKey = functions.generate_noise(opt.PRNGKey,[1,opt.nzx,opt.nzy])
                z_opt = m_noise(jnp.tile(z_opt,[1,3,1,1]))
                noise_,opt.PRNGKey = functions.generate_noise(opt.PRNGKey,[1,opt.nzx,opt.nzy])
                noise_ = m_noise(jnp.tile(noise_,[1,3,1,1]))
            else:
                noise_,opt.PRNGKey = functions.generate_noise(opt.PRNGKey,[opt.nc_z,opt.nzx,opt.nzy])
                noise_ = m_noise(noise_)

            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            with StopwatchPrint("Update D..."):
                for j in range(opt.Dsteps):                  

                    # train with fake
                    if (j==0) & (epoch == 0):
                        if Gs == []:
                            prev = jnp.zeros([1,opt.nc_z,opt.nzx,opt.nzy])
                            in_s = prev
                            prev = m_image(prev)
                            z_prev = jnp.zeros([1,opt.nc_z,opt.nzx,opt.nzy])
                            z_prev = m_noise(z_prev)
                            opt.noise_amp = 1
                        # elif opt.mode == 'SR_train':
                        #     z_prev = in_s
                        #     criterion = nn.MSELoss()
                        #     RMSE = torch.sqrt(criterion(real, z_prev))
                        #     opt.noise_amp = opt.noise_amp_init * RMSE
                        #     z_prev = m_image(z_prev)
                        #     prev = z_prev
                        else:
                            
                            prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                            prev = m_image(prev)
                            z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt)
                            criterion = nn.MSELoss()
                            RMSE = torch.sqrt(criterion(real, z_prev))
                            opt.noise_amp = opt.noise_amp_init*RMSE
                            z_prev = m_image(z_prev)
                    else:
                        prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                        prev = m_image(prev)

                    if (Gs == []) & (opt.mode != 'SR_train'):
                        noise = noise_
                    else:
                        noise = opt.noise_amp*noise_+prev
                    
                    
                    fake, stateG = apply_state(stateG, noise, prev)
                    
                    with StopwatchPrint("step state d..."):
                        errD, stateD = step_stateD(stateD, fake, real, prev, opt)

        errD2plot.append(errD)

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        with StopwatchPrint("Update G..."):
            for j in range(opt.Gsteps):

                output, stateD = apply_state(stateD, fake)
                errG = -output.mean()
                rec_loss, stateG, Z_opt = step_stateG(stateG, z_opt, z_prev, errG, alpha, real, opt)

        # errG2plot.append(errG.detach()+rec_loss)
        # D_real2plot.append(D_x)
        # D_fake2plot.append(D_G_z)
        # z_opt2plot.append(rec_loss)

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            GZ_opt, stateG = apply_state(stateG, Z_opt, z_prev)
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(GZ_opt), vmin=0, vmax=1)
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            functions.pickle_save(z_opt, '%s/z_opt.pth' % (opt.outf))

        # schedulerD.step()
        # schedulerG.step()
    with StopwatchPrint("Saving Networks..."):
        functions.save_networks(stateD,stateG,z_opt,opt)
    return z_opt,in_s,stateG    

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.kernel_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z,opt.PRNGKey = functions.generate_noise(opt.PRNGKey,[1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise])
                    z = jnp.tile(z,[1,3,1,1])
                else:
                    z,opt.PRNGKey = functions.generate_noise(opt.PRNGKey,[opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise])
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                # G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                # G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

# def train_paint(opt,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
#     in_s = torch.full(reals[0].shape, 0, device=opt.device)
#     scale_num = 0
#     nfc_prev = 0

#     while scale_num<opt.stop_scale+1:
#         if scale_num!=paint_inject_scale:
#             scale_num += 1
#             nfc_prev = opt.nfc
#             continue
#         else:
#             opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
#             opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

#             opt.out_ = functions.generate_dir2save(opt)
#             opt.outf = '%s/%d' % (opt.out_,scale_num)
#             try:
#                 os.makedirs(opt.outf)
#             except OSError:
#                     pass

#             #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
#             #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
#             plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

#             D_curr,G_curr = init_models(opt)

#             z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],opt,centers=centers)

#             G_curr = functions.reset_grads(G_curr,False)
#             G_curr.eval()
#             D_curr = functions.reset_grads(D_curr,False)
#             D_curr.eval()

#             Gs[scale_num] = G_curr
#             Zs[scale_num] = z_curr
#             NoiseAmp[scale_num] = opt.noise_amp

#             torch.save(Zs, '%s/Zs.pth' % (opt.out_))
#             torch.save(Gs, '%s/Gs.pth' % (opt.out_))
#             torch.save(reals, '%s/reals.pth' % (opt.out_))
#             torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

#             scale_num+=1
#             nfc_prev = opt.nfc
#         del D_curr,G_curr
#     return


def init_models(opt, img_shape):
    opt.PRNGKey, subkey = jax.random.split(opt.PRNGKey)
    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt)
    paramsG = netG.init(subkey, jnp.ones(img_shape), jnp.ones(img_shape))
    
    
    # netG.apply(models.weights_init)
    if opt.netG != '':
        paramsG = functions.pickle_load(opt.netG)
#     print(netG)

    opt.PRNGKey, subkey = jax.random.split(opt.PRNGKey)
    #discriminator initialization:
    netD = models.WDiscriminator(opt)
    paramsD = netD.init(subkey, jnp.ones(img_shape))
    # netD.apply(models.weights_init)
    if opt.netD != '':
        paramsD = functions.pickle_load(opt.netD)
#     print(netD)

    return netD, paramsD, netG, paramsG

