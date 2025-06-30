import argparse
import gc
# import copy
import json
import os
import sys
import h5py
# import hdf5storage
import time

import numpy as np
import scipy.io as io
import torch
import torch.distributions
from aux import *
from baselines import bounded_random_walk
# from interpolate import CubicHermiteMAkima as CubicHermiteInterp
from types import SimpleNamespace

sys.path.append('../')


def baselines(config: dict,
              ppm) -> tuple:
    '''
    Simulate baseline offsets
    '''
    cfg = SimpleNamespace(**config)
    baselines = batch_smooth(
                    bounded_random_walk(cfg.start, cfg.end, 
                                        cfg.std, cfg.lower_bnd, 
                                        cfg.upper_bnd, 
                                        cfg.length), 
                                cfg.windows, 'constant')

    # Subtract the trend lines 
    trend = batch_linspace(baselines[...,0].unsqueeze(-1),
                            baselines[...,-1].unsqueeze(-1), 
                            cfg.length)
    baselines = baselines - trend

    baselines, _ = normalize(signal=baselines, fid=False, denom=None, noisy=-1)

    if cfg.rand_omit>0: 
        baselines, _ = rand_omit(baselines, 0.0, cfg.rand_omit)

    # Convert simulated residual water from local to clinical range before 
    # Hilbert transform makes the imaginary component. Then resample 
    # acquired range to cropped range.
    # ppm_range =  [torch.as_tensor(val) for val in cfg.ppm_range]
    # print('ppm.shape: ', ppm.shape)
    raw_baseline = HilbertTransform(
                    sim2acquired(baselines * config['scale'], 
                                 [cfg.ppm_range[0], cfg.ppm_range[1]],
                                 ppm.squeeze(-1))
                    )
    
    flp = torch.distributions.bernoulli.Bernoulli(0.5).sample([raw_baseline.shape[0]]).long()
    raw_baseline[flp,...] = raw_baseline[flp,...].fliplr()
    
    return raw_baseline


def normalize(signal: torch.Tensor,
              fid: bool=False,
              denom: torch.Tensor=None,
              noisy: int=-1, # dim for noisy/clean
              ) -> torch.Tensor:
    '''
    Normalize each sample of single or multi-echo spectra. 
        Step 1: Find the max of the real and imaginary components separately
        Step 2: Pick the larger value for each spectrum
    If the signal is separated by metabolite, then an additional max() 
        is necessary
    Reimplemented according to: https://stackoverflow.com/questions/4157653
        6/normalizing-complex-values-in-numpy-python
    '''
    denom = torch.amax(signal[...,0,:].unsqueeze(-2).abs(),
                        dim=-1, keepdim=True)

    denom[denom.isnan()] = 1e-6
    denom[denom==0.0] = 1e-6
    # denom = torch.amax(denom, dim=noisy, keepdim=True)

    for _ in range(denom.ndim-signal.ndim): signal = signal.unsqueeze(1)

    return signal / denom, denom


def scale_offsets(spec: torch.Tensor,
                  baselines: tuple=None,
                  scale: torch.Tensor=None,
                  drop_prob: float=0.0,
                  ) -> dict:        
    '''
    Used for adding residual water and baselines. config dictionaries are 
    needed for each one.
    '''
    # print('spec.shape: ',spec.shape)
    max_val = np.amax(spec, axis=(-1,-2), keepdims=True) # 10**(OrderOfMagnitude(fid) - OrderOfMagnitude(out))
    # out, ind = rand_omit(out, 0.0, drop_prob)
    # print('baselines.shape {}, max_val.shape {}, scale.shape {}'.format(baselines.shape, max_val.shape, scale.shape))

    if scale.ndim==2: scale = scale.unsqueee(-1)

    return baselines.clone() * max_val * scale


def simulate(config_file, 
             totalEntries, 
             xfAll,
             xfNuisance,
             xfMeta,
             brainMask,
             ppm,
             args=None):
    with open(config_file) as file:
        config = json.load(file)
        baseline_cfg = config["baseline_cfg"]
        # del config
    
    
    params = torch.zeros((totalEntries, 1, 1))
    p = 1 - 0.0

    # Scaling
    # This range needs to be symmetric, therefore zero scaling occurs at 0.5
    print('>>> Scale')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[:,0].uniform_(-1,1) # [-0.1,0.1]
    params[sign,0].fill_(0)
    
    first = True
    step = 10000
    for i in range(0,totalEntries,step):
        n = 1+step if 1+step<=params.shape[0] else params.shape[0]-i
        print('step: ',n)
        outputs = baselines(sample_baselines(n, **baseline_cfg), ppm)

        if first:
            # _, _, baseline, _, _, _ = outputs
            baseline = outputs
            first = False
        else:
            baseline = torch.cat([baseline, outputs], dim=0)# if baseline else None

    '''After baselines have been generated,...'''
    ind = np.squeeze(brainMask)
    temp = np.zeros_like(xfAll)
    baseline = baseline[0:totalEntries,...]
    # baseline = normalize(baseline[0:totalEntries,...])
    # print('xfMeta.shape: ', xfMeta.shape)
    # print('ind.shape: ',ind.shape)
    # baseline = scale_offsets(spec=xfMeta[ind,...], baselines=baseline, scale=params, drop_prob=0.0).numpy()
    baseline = scale_offsets(spec=xfMeta[ind,...], baselines=baseline, scale=torch.ones_like(baseline), drop_prob=0.0).numpy()
    
    
    xfAll[ind,...] += baseline
    xfNuisance[ind,...] += baseline
    xfMeta[ind,...] += baseline
    temp[ind,...] += baseline
    
    return xfAll, xfNuisance, xfMeta, temp
    
    
    
def convert(file) -> dict:
    data = {}
    for k, v in file.items():
        data[k] = v
    return data

def h5py_to_dict(obj):
    """ Recursively convert h5py object to a Python dictionary. """
    if isinstance(obj, h5py.Dataset):
        # Convert dataset to a numpy array
        data = obj[()]
        # Convert byte strings to normal strings if necessary
        if data.dtype.type is np.bytes_:
            data = data.astype(str)
        return data
    elif isinstance(obj, h5py.Group):
        group_dict = {}
        for key, item in obj.items():
            group_dict[key] = h5py_to_dict(item)
        return group_dict
    else:
        raise TypeError(f"Unsupported h5py object type: {type(obj)}")
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--savedir', type=str, default='./dataset/')
    parser.add_argument('--batchSize', type=int, default=10000)
    parser.add_argument('--config_file', type=str, default='./src/configurations/DL_PRESS_144_ge.json')
    parser.add_argument('--subjectdir', type=str, default='./dataset/')
    
    args = parser.parse_args()
    
    # os.makedirs(args.savedir, exist_ok=True)
    
    folders = [d for d in args.subjectdir.split(',')]
    for d in folders:
        start = time.time()
        # data = io.loadmat(d)
        with h5py.File(d, 'r') as file:
            # data = convert(file)
            data = h5py_to_dict(file)

        Ns, z, y, x = data['xtAll'].shape
        # print(data['xtAll'].dtype, data['xtAll'].shape)
        print('xtAll FFT')
        xfAll = np.reshape(np.fft.fftshift(np.fft.fft(np.moveaxis(data['xtAll']['real'] + 1j*data['xtAll']['imag'], source=[0,1,2,3], destination=[3,2,1,0]))), [int(x*y*z), 1, Ns])
        # xfAll = np.fft.fftshift(np.fft.fft(np.moveaxis(data['xtAll']['real'] + 1j*data['xtAll']['imag'], source=[0,1,2,3], destination=[3,2,1,0])))
        # print('xfAll: ',xfAll.shape)
        # xfAll = np.reshape(xfAll, [int(x*y*z), 1, Ns])
        # print('xfAll: ',xfAll.shape)
        
        xfAll = np.concatenate([np.real(xfAll), np.imag(xfAll)], axis=1)
        print('xtNuisance FFT')
        xfNuisance = np.reshape(np.fft.fftshift(np.fft.fft(np.moveaxis(data['xtNuisance']['real'] + 1j*data['xtNuisance']['imag'], source=[0,1,2,3], destination=[3,2,1,0]))), [int(x*y*z), 1, Ns])
        xfNuisance = np.concatenate([np.real(xfNuisance), np.imag(xfNuisance)], axis=1)
        print('xtMeta FFT')
        xfMeta = np.reshape(np.fft.fftshift(np.fft.fft(np.moveaxis(data['xtMeta']['real'] + 1j*data['xtMeta']['imag'], source=[0,1,2,3], destination=[3,2,1,0]))), [int(x*y*z), 1, Ns])
        xfMeta = np.concatenate([np.real(xfMeta), np.imag(xfMeta)], axis=1)
        print('T1w')
        Iref = np.reshape(np.moveaxis(data['Iref'], source=[0,1,2], destination=[2,1,0]), [int(x*y*z), 1, 1])
        totalEntries = np.sum(data['brainMask'], axis=(0,1,2))
        brainMask = np.reshape(np.moveaxis(data['brainMask'], source=[0,1,2], destination=[2,1,0]), [int(x*y*z), 1]).astype(bool)
        dt = data['t'][1] - data['t'][0]
        bw = 1 / dt
        ppm = torch.from_numpy(np.linspace(-0.5*bw, 0.5*bw, Ns) / data['hzpppm'] + data['ppmoff']).unsqueeze(0)#.unsqueeze(0)
        print('bandwidth: {}, dwell time: {}, 0.5*bw: {}, 0.5*bw/centerFreq: {}'.format(bw, dt, 0.5*bw, 0.5*bw/data['hzpppm']))
        print('[ppm.min() {}, ppm.max() {}]'.format(ppm.min(), ppm.max()))

        print('Generating baselines')
        outputs = simulate(config_file=args.config_file, 
                        totalEntries=totalEntries,
                        xfAll=xfAll,
                        xfNuisance=xfNuisance,
                        xfMeta=xfMeta,
                        brainMask=brainMask,
                        ppm=ppm,
                        args=args)
        
        del xfAll, xfMeta, xfNuisance
    #     print('Finished')
    #     print('Unpacking and preparing to save')
    #     xfAll, xfNuisance, xfMeta, baseline = outputs
    #     xtAll_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfAll)), [x,y,z,2,Ns])
    #     xtAll_b = xtAll_b[...,0,:] + 1j*xtAll_b[...,1,:]
    #     data.create_dataset(name='xtAll_b', data=xtAll_b, maxshape=(xtAll_b.shape))
    #     del xtAll_b
        
    #     xtNuisance_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfNuisance)), [x,y,z,2,Ns])
    #     xtNuisance_b = xtNuisance_b[...,0,:] + 1j*xtNuisance_b[...,1,:]
    #     data.create_dataset(name='xtNuisance_b', data=xtNuisance_b, maxshape=(xtNuisance_b.shape))
    #     del xtNuisance_b
        
    #     xtMeta_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfMeta)), [x,y,z,2,Ns])
    #     xtMeta_b = xtMeta_b[...,0,:] + 1j*xtMeta_b[...,1,:]
    #     data.create_dataset(name='xtMeta_b', data=xtMeta_b, maxshape=(xtMeta_b.shape))
    #     del xtMeta_b
        
    #     baseline = np.reshape(np.fft.ifft(np.fft.ifftshift(baseline)), [x,y,z,2,Ns])
    #     baseline = baseline[...,0,:] + 1j*baseline[...,1,:]
    #     data.create_dataset(name='baseline', data=baseline, maxshape=(baseline.shape))
    #     del baseline
    # # hdf5storage.savemat(d, data, format='7.3', oned_as='column', matlab_compatible=True)
    
    
        print('Finished')
        print('Unpacking and preparing to save')
        data1 = {}
        xfAll, xfNuisance, xfMeta, baseline = outputs
        xtAll_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfAll)), [x,y,z,2,Ns])
        data1['xtAll_b'] = xtAll_b[...,0,:] + 1j*xtAll_b[...,1,:]
        del xtAll_b
        
        xtNuisance_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfNuisance)), [x,y,z,2,Ns])
        data1['xtNuisance_b'] = xtNuisance_b[...,0,:] + 1j*xtNuisance_b[...,1,:]
        del xtNuisance_b
        
        xtMeta_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfMeta)), [x,y,z,2,Ns])
        data1['xtMeta_b'] = xtMeta_b[...,0,:] + 1j*xtMeta_b[...,1,:]
        del xtMeta_b
        
        baseline = np.reshape(np.fft.ifft(np.fft.ifftshift(baseline)), [x,y,z,2,Ns])
        data1['baseline'] = baseline[...,0,:] + 1j*baseline[...,1,:]
        del baseline
        
        data1['brainMask'] = np.reshape(brainMask, [x,y,z,1])
        data1['Iref'] = np.reshape(Iref, [x,y,z])
        
        
        print('Finished. Ready to save: ',time.time() - start)
        start = time.time()
        base, ext = os.path.splitext(d)
        new_name = base + '_baselines' + ext
        io.savemat(new_name, mdict=data1, do_compression=False)
        print(time.time() - start)
    #     with h5py.File(new_name, 'w') as file:
    #         for k, v in data.items():
    #             file.create_dataset(k, data=v)
    # # hdf5storage.savemat(d, data, format='7.3', oned_as='column', matlab_compatible=True)import argparse
# import copy
import json
import os
import sys
import h5py
# import hdf5storage
import time

import numpy as np
import scipy.io as io
import torch
import torch.distributions
from aux import *
from baselines import bounded_random_walk
# from interpolate import CubicHermiteMAkima as CubicHermiteInterp
from types import SimpleNamespace

sys.path.append('../')


def baselines(config: dict,
              ppm) -> tuple:
    '''
    Simulate baseline offsets
    '''
    cfg = SimpleNamespace(**config)
    baselines = batch_smooth(
                    bounded_random_walk(cfg.start, cfg.end, 
                                        cfg.std, cfg.lower_bnd, 
                                        cfg.upper_bnd, 
                                        cfg.length), 
                                cfg.windows, 'constant')

    # Subtract the trend lines 
    trend = batch_linspace(baselines[...,0].unsqueeze(-1),
                            baselines[...,-1].unsqueeze(-1), 
                            cfg.length)
    baselines = baselines - trend

    baselines, _ = normalize(signal=baselines, fid=False, denom=None, noisy=-1)

    if cfg.rand_omit>0: 
        baselines, _ = rand_omit(baselines, 0.0, cfg.rand_omit)

    # Convert simulated residual water from local to clinical range before 
    # Hilbert transform makes the imaginary component. Then resample 
    # acquired range to cropped range.
    # ppm_range =  [torch.as_tensor(val) for val in cfg.ppm_range]
    # print('ppm.shape: ', ppm.shape)
    raw_baseline = HilbertTransform(
                    sim2acquired(baselines * config['scale'], 
                                 [cfg.ppm_range[0], cfg.ppm_range[1]],
                                 ppm.squeeze(-1))
                    )
    
    flp = torch.distributions.bernoulli.Bernoulli(0.5).sample([raw_baseline.shape[0]]).long()
    raw_baseline[flp,...] = raw_baseline[flp,...].fliplr()
    
    return raw_baseline


def normalize(signal: torch.Tensor,
              fid: bool=False,
              denom: torch.Tensor=None,
              noisy: int=-1, # dim for noisy/clean
              ) -> torch.Tensor:
    '''
    Normalize each sample of single or multi-echo spectra. 
        Step 1: Find the max of the real and imaginary components separately
        Step 2: Pick the larger value for each spectrum
    If the signal is separated by metabolite, then an additional max() 
        is necessary
    Reimplemented according to: https://stackoverflow.com/questions/4157653
        6/normalizing-complex-values-in-numpy-python
    '''
    denom = torch.amax(signal[...,0,:].unsqueeze(-2).abs(),
                        dim=-1, keepdim=True)

    denom[denom.isnan()] = 1e-6
    denom[denom==0.0] = 1e-6
    # denom = torch.amax(denom, dim=noisy, keepdim=True)

    for _ in range(denom.ndim-signal.ndim): signal = signal.unsqueeze(1)

    return signal / denom, denom


def scale_offsets(spec: torch.Tensor,
                  baselines: tuple=None,
                  scale: torch.Tensor=None,
                  drop_prob: float=0.0,
                  ) -> dict:        
    '''
    Used for adding residual water and baselines. config dictionaries are 
    needed for each one.
    '''
    # print('spec.shape: ',spec.shape)
    max_val = np.amax(spec, axis=(-1,-2), keepdims=True) # 10**(OrderOfMagnitude(fid) - OrderOfMagnitude(out))
    # out, ind = rand_omit(out, 0.0, drop_prob)
    # print('baselines.shape {}, max_val.shape {}, scale.shape {}'.format(baselines.shape, max_val.shape, scale.shape))

    if scale.ndim==2: scale = scale.unsqueee(-1)

    return baselines.clone() * max_val * scale


def simulate(config_file, 
             totalEntries, 
             xfAll,
             xfNuisance,
             xfMeta,
             brainMask,
             ppm,
             args=None):
    with open(config_file) as file:
        config = json.load(file)
        baseline_cfg = config["baseline_cfg"]
        # del config
    
    
    params = torch.zeros((totalEntries, 1, 1))
    p = 1 - 0.0

    # Scaling
    # This range needs to be symmetric, therefore zero scaling occurs at 0.5
    print('>>> Scale')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[:,0].uniform_(-1,1) # [-0.1,0.1]
    params[sign,0].fill_(0)
    
    first = True
    step = 10000
    for i in range(0,totalEntries,step):
        n = 1+step if 1+step<=params.shape[0] else params.shape[0]-i
        print('step: ',n)
        outputs = baselines(sample_baselines(n, **baseline_cfg), ppm)

        if first:
            # _, _, baseline, _, _, _ = outputs
            baseline = outputs
            first = False
        else:
            baseline = torch.cat([baseline, outputs], dim=0)# if baseline else None

    '''After baselines have been generated,...'''
    ind = np.squeeze(brainMask)
    temp = np.zeros_like(xfAll)
    baseline = baseline[0:totalEntries,...]
    # baseline = normalize(baseline[0:totalEntries,...])
    # print('xfMeta.shape: ', xfMeta.shape)
    # print('ind.shape: ',ind.shape)
    # baseline = scale_offsets(spec=xfMeta[ind,...], baselines=baseline, scale=params, drop_prob=0.0).numpy()
    baseline = scale_offsets(spec=xfMeta[ind,...], baselines=baseline, scale=torch.ones_like(baseline), drop_prob=0.0).numpy()
    
    
    xfAll[ind,...] += baseline
    xfNuisance[ind,...] += baseline
    xfMeta[ind,...] += baseline
    temp[ind,...] += baseline
    
    return xfAll, xfNuisance, xfMeta, temp
    
    
    
def convert(file) -> dict:
    data = {}
    for k, v in file.items():
        data[k] = v
    return data

def h5py_to_dict(obj):
    """ Recursively convert h5py object to a Python dictionary. """
    if isinstance(obj, h5py.Dataset):
        # Convert dataset to a numpy array
        data = obj[()]
        # Convert byte strings to normal strings if necessary
        if data.dtype.type is np.bytes_:
            data = data.astype(str)
        return data
    elif isinstance(obj, h5py.Group):
        group_dict = {}
        for key, item in obj.items():
            group_dict[key] = h5py_to_dict(item)
        return group_dict
    else:
        raise TypeError(f"Unsupported h5py object type: {type(obj)}")
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--savedir', type=str, default='./dataset/')
    parser.add_argument('--batchSize', type=int, default=10000)
    parser.add_argument('--config_file', type=str, default='./src/configurations/DL_PRESS_144_ge.json')
    parser.add_argument('--subjectdir', type=str, default='./dataset/')
    
    args = parser.parse_args()
    
    # os.makedirs(args.savedir, exist_ok=True)
    
    folders = [d for d in args.subjectdir.split(',')]
    for d in folders:
        start = time.time()
        # data = io.loadmat(d)
        with h5py.File(d, 'r') as file:
            # data = convert(file)
            data = h5py_to_dict(file)

        Ns, z, y, x = data['xtAll'].shape
        # print(data['xtAll'].dtype, data['xtAll'].shape)
        print('xtAll FFT')
        xfAll = np.reshape(np.fft.fftshift(np.fft.fft(np.moveaxis(data['xtAll']['real'] + 1j*data['xtAll']['imag'], source=[0,1,2,3], destination=[3,2,1,0]))), [int(x*y*z), 1, Ns])
        # xfAll = np.fft.fftshift(np.fft.fft(np.moveaxis(data['xtAll']['real'] + 1j*data['xtAll']['imag'], source=[0,1,2,3], destination=[3,2,1,0])))
        # print('xfAll: ',xfAll.shape)
        # xfAll = np.reshape(xfAll, [int(x*y*z), 1, Ns])
        # print('xfAll: ',xfAll.shape)
        
        xfAll = np.concatenate([np.real(xfAll), np.imag(xfAll)], axis=1)
        print('xtNuisance FFT')
        xfNuisance = np.reshape(np.fft.fftshift(np.fft.fft(np.moveaxis(data['xtNuisance']['real'] + 1j*data['xtNuisance']['imag'], source=[0,1,2,3], destination=[3,2,1,0]))), [int(x*y*z), 1, Ns])
        xfNuisance = np.concatenate([np.real(xfNuisance), np.imag(xfNuisance)], axis=1)
        print('xtMeta FFT')
        xfMeta = np.reshape(np.fft.fftshift(np.fft.fft(np.moveaxis(data['xtMeta']['real'] + 1j*data['xtMeta']['imag'], source=[0,1,2,3], destination=[3,2,1,0]))), [int(x*y*z), 1, Ns])
        xfMeta = np.concatenate([np.real(xfMeta), np.imag(xfMeta)], axis=1)
        print('T1w')
        Iref = np.reshape(np.moveaxis(data['Iref'], source=[0,1,2], destination=[2,1,0]), [int(x*y*z), 1, 1])
        totalEntries = np.sum(data['brainMask'], axis=(0,1,2))
        brainMask = np.reshape(np.moveaxis(data['brainMask'], source=[0,1,2], destination=[2,1,0]), [int(x*y*z), 1]).astype(bool)
        dt = data['t'][1] - data['t'][0]
        bw = 1 / dt
        ppm = torch.from_numpy(np.linspace(-0.5*bw, 0.5*bw, Ns) / data['hzpppm'] + data['ppmoff']).unsqueeze(0)#.unsqueeze(0)
        print('bandwidth: {}, dwell time: {}, 0.5*bw: {}, 0.5*bw/centerFreq: {}'.format(bw, dt, 0.5*bw, 0.5*bw/data['hzpppm']))
        print('[ppm.min() {}, ppm.max() {}]'.format(ppm.min(), ppm.max()))

        print('Generating baselines')
        outputs = simulate(config_file=args.config_file, 
                        totalEntries=totalEntries,
                        xfAll=xfAll,
                        xfNuisance=xfNuisance,
                        xfMeta=xfMeta,
                        brainMask=brainMask,
                        ppm=ppm,
                        args=args)
        
        del xfAll, xfMeta, xfNuisance
    #     print('Finished')
    #     print('Unpacking and preparing to save')
    #     xfAll, xfNuisance, xfMeta, baseline = outputs
    #     xtAll_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfAll)), [x,y,z,2,Ns])
    #     xtAll_b = xtAll_b[...,0,:] + 1j*xtAll_b[...,1,:]
    #     data.create_dataset(name='xtAll_b', data=xtAll_b, maxshape=(xtAll_b.shape))
    #     del xtAll_b
        
    #     xtNuisance_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfNuisance)), [x,y,z,2,Ns])
    #     xtNuisance_b = xtNuisance_b[...,0,:] + 1j*xtNuisance_b[...,1,:]
    #     data.create_dataset(name='xtNuisance_b', data=xtNuisance_b, maxshape=(xtNuisance_b.shape))
    #     del xtNuisance_b
        
    #     xtMeta_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfMeta)), [x,y,z,2,Ns])
    #     xtMeta_b = xtMeta_b[...,0,:] + 1j*xtMeta_b[...,1,:]
    #     data.create_dataset(name='xtMeta_b', data=xtMeta_b, maxshape=(xtMeta_b.shape))
    #     del xtMeta_b
        
    #     baseline = np.reshape(np.fft.ifft(np.fft.ifftshift(baseline)), [x,y,z,2,Ns])
    #     baseline = baseline[...,0,:] + 1j*baseline[...,1,:]
    #     data.create_dataset(name='baseline', data=baseline, maxshape=(baseline.shape))
    #     del baseline
    # # hdf5storage.savemat(d, data, format='7.3', oned_as='column', matlab_compatible=True)
    
    
        print('Finished')
        print('Unpacking and preparing to save')
        data1 = {}
        xfAll, xfNuisance, xfMeta, baseline = outputs
        xtAll_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfAll)), [x,y,z,2,Ns])
        data1['xtAll_b'] = xtAll_b[...,0,:] + 1j*xtAll_b[...,1,:]
        del xtAll_b
        
        xtNuisance_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfNuisance)), [x,y,z,2,Ns])
        data1['xtNuisance_b'] = xtNuisance_b[...,0,:] + 1j*xtNuisance_b[...,1,:]
        del xtNuisance_b
        
        xtMeta_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfMeta)), [x,y,z,2,Ns])
        data1['xtMeta_b'] = xtMeta_b[...,0,:] + 1j*xtMeta_b[...,1,:]
        del xtMeta_b
        
        baseline = np.reshape(np.fft.ifft(np.fft.ifftshift(baseline)), [x,y,z,2,Ns])
        data1['baseline'] = baseline[...,0,:] + 1j*baseline[...,1,:]
        del baseline
        
        data1['brainMask'] = np.reshape(brainMask, [x,y,z,1])
        data1['Iref'] = np.reshape(Iref, [x,y,z])
        
        
        print('Finished. Ready to save: ',time.time() - start)
        start = time.time()
        base, ext = os.path.splitext(d)
        new_name = base + '_baselines' + ext
        io.savemat(new_name, mdict=data1, do_compression=False)
        print(time.time() - start)
    #     with h5py.File(new_name, 'w') as file:
    #         for k, v in data.items():
    #             file.create_dataset(k, data=v)
    # # hdf5storage.savemat(d, data, format='7.3', oned_as='column', matlab_compatible=True)import argparse
# import copy
import json
import os
import sys
import h5py
# import hdf5storage
import time

import numpy as np
import scipy.io as io
import torch
import torch.distributions
from aux import *
from baselines import bounded_random_walk
# from interpolate import CubicHermiteMAkima as CubicHermiteInterp
from types import SimpleNamespace

sys.path.append('../')


def baselines(config: dict,
              ppm) -> tuple:
    '''
    Simulate baseline offsets
    '''
    cfg = SimpleNamespace(**config)
    baselines = batch_smooth(
                    bounded_random_walk(cfg.start, cfg.end, 
                                        cfg.std, cfg.lower_bnd, 
                                        cfg.upper_bnd, 
                                        cfg.length), 
                                cfg.windows, 'constant')

    # Subtract the trend lines 
    trend = batch_linspace(baselines[...,0].unsqueeze(-1),
                            baselines[...,-1].unsqueeze(-1), 
                            cfg.length)
    baselines = baselines - trend

    baselines, _ = normalize(signal=baselines, fid=False, denom=None, noisy=-1)

    if cfg.rand_omit>0: 
        baselines, _ = rand_omit(baselines, 0.0, cfg.rand_omit)

    # Convert simulated residual water from local to clinical range before 
    # Hilbert transform makes the imaginary component. Then resample 
    # acquired range to cropped range.
    # ppm_range =  [torch.as_tensor(val) for val in cfg.ppm_range]
    # print('ppm.shape: ', ppm.shape)
    raw_baseline = HilbertTransform(
                    sim2acquired(baselines * config['scale'], 
                                 [cfg.ppm_range[0], cfg.ppm_range[1]],
                                 ppm.squeeze(-1))
                    )
    
    flp = torch.distributions.bernoulli.Bernoulli(0.5).sample([raw_baseline.shape[0]]).long()
    raw_baseline[flp,...] = raw_baseline[flp,...].fliplr()
    
    return raw_baseline


def normalize(signal: torch.Tensor,
              fid: bool=False,
              denom: torch.Tensor=None,
              noisy: int=-1, # dim for noisy/clean
              ) -> torch.Tensor:
    '''
    Normalize each sample of single or multi-echo spectra. 
        Step 1: Find the max of the real and imaginary components separately
        Step 2: Pick the larger value for each spectrum
    If the signal is separated by metabolite, then an additional max() 
        is necessary
    Reimplemented according to: https://stackoverflow.com/questions/4157653
        6/normalizing-complex-values-in-numpy-python
    '''
    denom = torch.amax(signal[...,0,:].unsqueeze(-2).abs(),
                        dim=-1, keepdim=True)

    denom[denom.isnan()] = 1e-6
    denom[denom==0.0] = 1e-6
    # denom = torch.amax(denom, dim=noisy, keepdim=True)

    for _ in range(denom.ndim-signal.ndim): signal = signal.unsqueeze(1)

    return signal / denom, denom


def scale_offsets(spec: torch.Tensor,
                  baselines: tuple=None,
                  scale: torch.Tensor=None,
                  drop_prob: float=0.0,
                  ) -> dict:        
    '''
    Used for adding residual water and baselines. config dictionaries are 
    needed for each one.
    '''
    # print('spec.shape: ',spec.shape)
    max_val = np.amax(spec, axis=(-1,-2), keepdims=True) # 10**(OrderOfMagnitude(fid) - OrderOfMagnitude(out))
    # out, ind = rand_omit(out, 0.0, drop_prob)
    # print('baselines.shape {}, max_val.shape {}, scale.shape {}'.format(baselines.shape, max_val.shape, scale.shape))

    if scale.ndim==2: scale = scale.unsqueee(-1)

    return baselines.clone() * max_val * scale


def simulate(config_file, 
             totalEntries, 
             xfAll,
             xfNuisance,
             xfMeta,
             brainMask,
             ppm,
             args=None):
    with open(config_file) as file:
        config = json.load(file)
        baseline_cfg = config["baseline_cfg"]
        # del config
    
    
    params = torch.zeros((totalEntries, 1, 1))
    p = 1 - 0.0

    # Scaling
    # This range needs to be symmetric, therefore zero scaling occurs at 0.5
    print('>>> Scale')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[:,0].uniform_(-1,1) # [-0.1,0.1]
    params[sign,0].fill_(0)
    
    first = True
    step = 10000
    for i in range(0,totalEntries,step):
        n = 1+step if 1+step<=params.shape[0] else params.shape[0]-i
        print('step: ',n)
        outputs = baselines(sample_baselines(n, **baseline_cfg), ppm)

        if first:
            # _, _, baseline, _, _, _ = outputs
            baseline = outputs
            first = False
        else:
            baseline = torch.cat([baseline, outputs], dim=0)# if baseline else None

    '''After baselines have been generated,...'''
    ind = np.squeeze(brainMask)
    temp = np.zeros_like(xfAll)
    baseline = baseline[0:totalEntries,...]
    # baseline = normalize(baseline[0:totalEntries,...])
    # print('xfMeta.shape: ', xfMeta.shape)
    # print('ind.shape: ',ind.shape)
    # baseline = scale_offsets(spec=xfMeta[ind,...], baselines=baseline, scale=params, drop_prob=0.0).numpy()
    baseline = scale_offsets(spec=xfMeta[ind,...], baselines=baseline, scale=torch.ones_like(baseline), drop_prob=0.0).numpy()
    
    
    xfAll[ind,...] += baseline
    xfNuisance[ind,...] += baseline
    xfMeta[ind,...] += baseline
    temp[ind,...] += baseline
    
    return xfAll, xfNuisance, xfMeta, temp
    
    
    
def convert(file) -> dict:
    data = {}
    for k, v in file.items():
        data[k] = v
    return data

def h5py_to_dict(obj):
    """ Recursively convert h5py object to a Python dictionary. """
    if isinstance(obj, h5py.Dataset):
        # Convert dataset to a numpy array
        data = obj[()]
        # Convert byte strings to normal strings if necessary
        if data.dtype.type is np.bytes_:
            data = data.astype(str)
        return data
    elif isinstance(obj, h5py.Group):
        group_dict = {}
        for key, item in obj.items():
            group_dict[key] = h5py_to_dict(item)
        return group_dict
    else:
        raise TypeError(f"Unsupported h5py object type: {type(obj)}")
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--savedir', type=str, default='./dataset/')
    parser.add_argument('--batchSize', type=int, default=10000)
    parser.add_argument('--config_file', type=str, default='./src/configurations/DL_PRESS_144_ge.json')
    parser.add_argument('--subjectdir', type=str, default='./dataset/')
    
    args = parser.parse_args()
    
    # os.makedirs(args.savedir, exist_ok=True)
    
    folders = [d for d in args.subjectdir.split(',')]
    for d in folders:
        start = time.time()
        # data = io.loadmat(d)
        with h5py.File(d, 'r') as file:
            # data = convert(file)
            data = h5py_to_dict(file)

        Ns, z, y, x = data['xtAll'].shape
        # print(data['xtAll'].dtype, data['xtAll'].shape)
        print('xtAll FFT')
        xfAll = np.reshape(np.fft.fftshift(np.fft.fft(np.moveaxis(data['xtAll']['real'] + 1j*data['xtAll']['imag'], source=[0,1,2,3], destination=[3,2,1,0]))), [int(x*y*z), 1, Ns])
        # xfAll = np.fft.fftshift(np.fft.fft(np.moveaxis(data['xtAll']['real'] + 1j*data['xtAll']['imag'], source=[0,1,2,3], destination=[3,2,1,0])))
        # print('xfAll: ',xfAll.shape)
        # xfAll = np.reshape(xfAll, [int(x*y*z), 1, Ns])
        # print('xfAll: ',xfAll.shape)
        
        xfAll = np.concatenate([np.real(xfAll), np.imag(xfAll)], axis=1)
        print('xtNuisance FFT')
        xfNuisance = np.reshape(np.fft.fftshift(np.fft.fft(np.moveaxis(data['xtNuisance']['real'] + 1j*data['xtNuisance']['imag'], source=[0,1,2,3], destination=[3,2,1,0]))), [int(x*y*z), 1, Ns])
        xfNuisance = np.concatenate([np.real(xfNuisance), np.imag(xfNuisance)], axis=1)
        print('xtMeta FFT')
        xfMeta = np.reshape(np.fft.fftshift(np.fft.fft(np.moveaxis(data['xtMeta']['real'] + 1j*data['xtMeta']['imag'], source=[0,1,2,3], destination=[3,2,1,0]))), [int(x*y*z), 1, Ns])
        xfMeta = np.concatenate([np.real(xfMeta), np.imag(xfMeta)], axis=1)
        print('T1w')
        Iref = np.reshape(np.moveaxis(data['Iref'], source=[0,1,2], destination=[2,1,0]), [int(x*y*z), 1, 1])
        totalEntries = np.sum(data['brainMask'], axis=(0,1,2))
        brainMask = np.reshape(np.moveaxis(data['brainMask'], source=[0,1,2], destination=[2,1,0]), [int(x*y*z), 1]).astype(bool)
        dt = data['t'][1] - data['t'][0]
        bw = 1 / dt
        ppm = torch.from_numpy(np.linspace(-0.5*bw, 0.5*bw, Ns) / data['hzpppm'] + data['ppmoff']).unsqueeze(0)#.unsqueeze(0)
        print('bandwidth: {}, dwell time: {}, 0.5*bw: {}, 0.5*bw/centerFreq: {}'.format(bw, dt, 0.5*bw, 0.5*bw/data['hzpppm']))
        print('[ppm.min() {}, ppm.max() {}]'.format(ppm.min(), ppm.max()))

        print('Generating baselines')
        outputs = simulate(config_file=args.config_file, 
                        totalEntries=totalEntries,
                        xfAll=xfAll,
                        xfNuisance=xfNuisance,
                        xfMeta=xfMeta,
                        brainMask=brainMask,
                        ppm=ppm,
                        args=args)
        
        del xfAll, xfMeta, xfNuisance
    #     print('Finished')
    #     print('Unpacking and preparing to save')
    #     xfAll, xfNuisance, xfMeta, baseline = outputs
    #     xtAll_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfAll)), [x,y,z,2,Ns])
    #     xtAll_b = xtAll_b[...,0,:] + 1j*xtAll_b[...,1,:]
    #     data.create_dataset(name='xtAll_b', data=xtAll_b, maxshape=(xtAll_b.shape))
    #     del xtAll_b
        
    #     xtNuisance_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfNuisance)), [x,y,z,2,Ns])
    #     xtNuisance_b = xtNuisance_b[...,0,:] + 1j*xtNuisance_b[...,1,:]
    #     data.create_dataset(name='xtNuisance_b', data=xtNuisance_b, maxshape=(xtNuisance_b.shape))
    #     del xtNuisance_b
        
    #     xtMeta_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfMeta)), [x,y,z,2,Ns])
    #     xtMeta_b = xtMeta_b[...,0,:] + 1j*xtMeta_b[...,1,:]
    #     data.create_dataset(name='xtMeta_b', data=xtMeta_b, maxshape=(xtMeta_b.shape))
    #     del xtMeta_b
        
    #     baseline = np.reshape(np.fft.ifft(np.fft.ifftshift(baseline)), [x,y,z,2,Ns])
    #     baseline = baseline[...,0,:] + 1j*baseline[...,1,:]
    #     data.create_dataset(name='baseline', data=baseline, maxshape=(baseline.shape))
    #     del baseline
    # # hdf5storage.savemat(d, data, format='7.3', oned_as='column', matlab_compatible=True)
    
    
        print('Finished')
        print('Unpacking and preparing to save')
        data1 = {}
        xfAll, xfNuisance, xfMeta, baseline = outputs
        xtAll_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfAll)), [x,y,z,2,Ns])
        data1['xtAll_b'] = xtAll_b[...,0,:] + 1j*xtAll_b[...,1,:]
        del xtAll_b
        
        xtNuisance_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfNuisance)), [x,y,z,2,Ns])
        data1['xtNuisance_b'] = xtNuisance_b[...,0,:] + 1j*xtNuisance_b[...,1,:]
        del xtNuisance_b
        
        xtMeta_b = np.reshape(np.fft.ifft(np.fft.ifftshift(xfMeta)), [x,y,z,2,Ns])
        data1['xtMeta_b'] = xtMeta_b[...,0,:] + 1j*xtMeta_b[...,1,:]
        del xtMeta_b
        
        baseline = np.reshape(np.fft.ifft(np.fft.ifftshift(baseline)), [x,y,z,2,Ns])
        data1['baseline'] = baseline[...,0,:] + 1j*baseline[...,1,:]
        del baseline
        
        data1['brainMask'] = np.reshape(brainMask, [x,y,z,1])
        data1['Iref'] = np.reshape(Iref, [x,y,z])
        
        
        print('Finished. Ready to save: ',time.time() - start)
        start = time.time()
        base, ext = os.path.splitext(d)
        new_name = base + '_baselines' + ext
        io.savemat(new_name, mdict=data1, do_compression=False)
        print(time.time() - start)
    #     with h5py.File(new_name, 'w') as file:
    #         for k, v in data.items():
    #             file.create_dataset(k, data=v)
    # # hdf5storage.savemat(d, data, format='7.3', oned_as='column', matlab_compatible=True)import argparse
# import copy
import json
import os
import sys
import h5py
# import hdf5storage
import time

import numpy as np
import scipy.io as io
import torch
import torch.distributions
from aux import *
from baselines import bounded_random_walk
# from interpolate import CubicHermiteMAkima as CubicHermiteInterp
from types import SimpleNamespace

sys.path.append('../')


def baselines(config: dict,
              ppm) -> tuple:
    '''
    Simulate baseline offsets
    '''
    cfg = SimpleNamespace(**config)
    baselines = batch_smooth(
                    bounded_random_walk(cfg.start, cfg.end, 
                                        cfg.std, cfg.lower_bnd, 
                                        cfg.upper_bnd, 
                                        cfg.length), 
                                cfg.windows, 'constant')

    # Subtract the trend lines 
    trend = batch_linspace(baselines[...,0].unsqueeze(-1),
                            baselines[...,-1].unsqueeze(-1), 
                            cfg.length)
    baselines = baselines - trend

    baselines, _ = normalize(signal=baselines, fid=False, denom=None, noisy=-1)

    if cfg.rand_omit>0: 
        baselines, _ = rand_omit(baselines, 0.0, cfg.rand_omit)

    # Convert simulated residual water from local to clinical range before 
    # Hilbert transform makes the imaginary component. Then resample 
    # acquired range to cropped range.
    # ppm_range =  [torch.as_tensor(val) for val in cfg.ppm_range]
    # print('ppm.shape: ', ppm.shape)
    raw_baseline = HilbertTransform(
                    sim2acquired(baselines * config['scale'], 
                                 [cfg.ppm_range[0], cfg.ppm_range[1]],
                                 ppm.squeeze(-1))
                    )
    
    flp = torch.distributions.bernoulli.Bernoulli(0.5).sample([raw_baseline.shape[0]]).long()
    raw_baseline[flp,...] = raw_baseline[flp,...].fliplr()
    
    return raw_baseline


def normalize(signal: torch.Tensor,
              fid: bool=False,
              denom: torch.Tensor=None,
              noisy: int=-1, # dim for noisy/clean
              ) -> torch.Tensor:
    '''
    Normalize each sample of single or multi-echo spectra. 
        Step 1: Find the max of the real and imaginary components separately
        Step 2: Pick the larger value for each spectrum
    If the signal is separated by metabolite, then an additional max() 
        is necessary
    Reimplemented according to: https://stackoverflow.com/questions/4157653
        6/normalizing-complex-values-in-numpy-python
    '''
    denom = torch.amax(signal[...,0,:].unsqueeze(-2).abs(),
                        dim=-1, keepdim=True)

    denom[denom.isnan()] = 1e-6
    denom[denom==0.0] = 1e-6
    # denom = torch.amax(denom, dim=noisy, keepdim=True)

    for _ in range(denom.ndim-signal.ndim): signal = signal.unsqueeze(1)

    return signal / denom, denom


def scale_offsets(spec: torch.Tensor,
                  baselines: tuple=None,
                  scale: torch.Tensor=None,
                  drop_prob: float=0.0,
                  ) -> dict:        
    '''
    Used for adding residual water and baselines. config dictionaries are 
    needed for each one.
    '''
    # print('spec.shape: ',spec.shape)
    max_val = np.amax(spec, axis=(-1,-2), keepdims=True) # 10**(OrderOfMagnitude(fid) - OrderOfMagnitude(out))
    # out, ind = rand_omit(out, 0.0, drop_prob)
    # print('baselines.shape {}, max_val.shape {}, scale.shape {}'.format(baselines.shape, max_val.shape, scale.shape))

    if scale.ndim==2: scale = scale.unsqueee(-1)

    return baselines.clone() * torch.from_numpy(max_val) * scale


def simulate(config_file, 
             totalEntries, 
             xfAll,
             xfNuisance,
             xfMeta,
             brainMask,
             ppm,
             args=None):
    with open(config_file) as file:
        config = json.load(file)
        baseline_cfg = config["baseline_cfg"]
        # del config
    
    
    params = torch.zeros((totalEntries, 1, 1))
    p = 1 - 0.0

    # Scaling
    # This range needs to be symmetric, therefore zero scaling occurs at 0.5
    print('>>> Scale')
    sign = torch.tensor([True if torch.rand([1]) > p else False for _ in range(params.shape[0])])
    params[:,0].uniform_(-1,1) # [-0.1,0.1]
    params[sign,0].fill_(0)
    params = torch.ones_like(torch.from_numpy(brainMask)).unsqueeze(0).unsqueeze(0).float().uniform_(0.2,1)
    
    first = True
    step = 10000
    for i in range(0,totalEntries,step):
        n = 1+step if 1+step<=params.shape[0] else params.shape[0]-i
        print('step: ',n)
        outputs = baselines(sample_baselines(n, **baseline_cfg), ppm)

        if first:
            # _, _, baseline, _, _, _ = outputs
            baseline = outputs
            first = False
        else:
            baseline = torch.cat([baseline, outputs], dim=0)# if baseline else None

    '''After baselines have been generated,...'''
    ind = np.squeeze(brainMask)
    temp = np.zeros_like(xfAll)
    baseline = baseline[0:totalEntries,...]
    print(sum(ind))
    # baseline = normalize(baseline[0:totalEntries,...])
    # print('xfMeta.shape: ', xfMeta.shape)
    # print('ind.shape: ',ind.shape)
    # baseline = scale_offsets(spec=xfMeta[ind,...], baselines=baseline, scale=params, drop_prob=0.0).numpy()
    # baseline = scale_offsets(spec=xfMeta[ind,...], baselines=baseline, scale=torch.ones_like(baseline), drop_prob=0.0).numpy()
    baseline = scale_offsets(spec=xfMeta[ind,...], baselines=baseline, scale=params, drop_prob=0.0).numpy()
    print('baseline.shape: ',baseline.shape)
    
    
    # xfAll[ind,...] += baseline
    # xfNuisance[ind,...] += baseline
    # xfMeta[ind,...] += baseline
    # temp[ind,...] += baseline
    
    # return xfAll, xfNuisance, xfMeta, temp
    return baseline
    
    
def convert(file) -> dict:
    data = {}
    for k, v in file.items():
        data[k] = v
    return data

def h5py_to_dict(obj):
    """ Recursively convert h5py object to a Python dictionary. """
    if isinstance(obj, h5py.Dataset):
        # Convert dataset to a numpy array
        data = obj[()]
        # Convert byte strings to normal strings if necessary
        if data.dtype.type is np.bytes_:
            data = data.astype(str)
        return data
    elif isinstance(obj, h5py.Group):
        group_dict = {}
        for key, item in obj.items():
            group_dict[key] = h5py_to_dict(item)
        return group_dict
    else:
        raise TypeError(f"Unsupported h5py object type: {type(obj)}")
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--savedir', type=str, default='./dataset/')
    parser.add_argument('--batchSize', type=int, default=10000)
    parser.add_argument('--config_file', type=str, default='./src/configurations/DL_PRESS_144_ge.json')
    parser.add_argument('--subjectdir', type=str, default='./dataset/')
    
    args = parser.parse_args()
    
    gc.collect()
    
    # os.makedirs(args.savedir, exist_ok=True)
    
    folders = [d for d in args.subjectdir.split(',')]
    for d in folders:
        start = time.time()
        # data = io.loadmat(d)
        with h5py.File(d, 'r') as file:
            # data = convert(file)
            data = h5py_to_dict(file)

        Ns, z, y, x = data['xtAll'].shape
        # print(data['xtAll'].dtype, data['xtAll'].shape)
        print('xtAll FFT')
        xfAll = np.reshape(np.fft.fft(np.moveaxis(data['xtAll']['real'] + 1j*data['xtAll']['imag'], source=[0,1,2,3], destination=[3,2,1,0])), [int(x*y*z), 1, Ns])
        xfAll = np.concatenate([np.real(xfAll), np.imag(xfAll)], axis=1)
        print('xtNuisance FFT')
        xfNuisance = np.reshape(np.fft.fft(np.moveaxis(data['xtNuisance']['real'] + 1j*data['xtNuisance']['imag'], source=[0,1,2,3], destination=[3,2,1,0])), [int(x*y*z), 1, Ns])
        xfNuisance = np.concatenate([np.real(xfNuisance), np.imag(xfNuisance)], axis=1)
        print('xtMeta FFT')
        xfMeta = np.reshape(np.fft.fft(np.moveaxis(data['xtMeta']['real'] + 1j*data['xtMeta']['imag'], source=[0,1,2,3], destination=[3,2,1,0])), [int(x*y*z), 1, Ns])
        xfMeta = np.concatenate([np.real(xfMeta), np.imag(xfMeta)], axis=1)
        print('T1w')
        Iref = np.reshape(np.moveaxis(data['Iref'], source=[0,1,2], destination=[2,1,0]), [int(x*y*z), 1, 1])
        totalEntries = np.sum(data['brainMask'], axis=(0,1,2))
        brainMask = np.reshape(np.moveaxis(data['brainMask'], source=[0,1,2], destination=[2,1,0]), [int(x*y*z), 1]).astype(bool)
        dt = data['t'][1] - data['t'][0]
        bw = 1 / dt
        ppm = torch.from_numpy(np.linspace(-0.5*bw, 0.5*bw, Ns) / data['hzpppm'] + data['ppmoff']).unsqueeze(0)#.unsqueeze(0)
        print('bandwidth: {}, dwell time: {}, 0.5*bw: {}, 0.5*bw/centerFreq: {}'.format(bw, dt, 0.5*bw, 0.5*bw/data['hzpppm']))
        print('[ppm.min() {}, ppm.max() {}]'.format(ppm.min(), ppm.max()))

        print('Generating baselines')
        outputs = simulate(config_file=args.config_file, 
                        totalEntries=totalEntries,
                        xfAll=xfAll,
                        xfNuisance=xfNuisance,
                        xfMeta=xfMeta,
                        brainMask=brainMask,
                        ppm=ppm,
                        args=args)
    #     baseline = outputs
    #     for xx in range(x):
    #         for yy in range(y):
    #             for zz in range(z):
    #                 data1['xtAll_b']

    
    
    #     print('Finished')
    #     print('Unpacking and preparing to save')
    #     data1 = {}
    #     xfAll, xfNuisance, xfMeta, baseline = outputs
    #     data1['xtAll_b'] = xtAll_b[...,0,:] + 1j*xtAll_b[...,1,:]
    #     data1['xtAll_b'] = np.reshape(np.fft.ifft(data1['xtAll_b']), [x,y,z,Ns])
    #     del xtAll_b
        
    #     data1['xtNuisance_b'] = xtNuisance_b[...,0,:] + 1j*xtNuisance_b[...,1,:]
    #     data1['xtNuisance_b'] = np.reshape(np.fft.ifft(data1['xtNuisance_b']), [x,y,z,Ns])
    #     del xtNuisance_b
        
    #     data1['xtMeta_b'] = xtMeta_b[...,0,:] + 1j*xtMeta_b[...,1,:]
    #     data1['xtMeta_b'] = np.reshape(np.fft.ifft(data1['xtMeta_b']), [x,y,z,Ns])
    #     del xtMeta_b
        
    #     data1['baseline'] = baseline[...,0,:] + 1j*baseline[...,1,:]
    #     data1['baseline'] = np.reshape(np.fft.ifft(data1['baseline']), [x,y,z,Ns])
    #     del baseline
        
    #     data1['brainMask'] = np.reshape(brainMask, [x,y,z,1])
    #     data1['Iref'] = np.reshape(Iref, [x,y,z])
        
        
    #     print('Finished. Ready to save: ',time.time() - start)
    #     start = time.time()
    #     base, ext = os.path.splitext(d)
    #     new_name = base + '_baselines' + ext
    #     io.savemat(new_name, mdict=data1, do_compression=False)
    #     print(time.time() - start)
    # #     with h5py.File(new_name, 'w') as file:
    # #         for k, v in data.items():
    # #             file.create_dataset(k, data=v)
    # # # hdf5storage.savemat(d, data, format='7.3', oned_as='column', matlab_compatible=True)