import argparse
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
                  ind: torch.Tensor=None,
                  ) -> dict:        
    '''
    Used for adding residual water and baselines. config dictionaries are 
    needed for each one.
    '''
    max_val = torch.amax(spec, dim=(0,1), keepdims=True) # 10**(OrderOfMagnitude(fid) - OrderOfMagnitude(out))
    # out, ind = rand_omit(out, 0.0, drop_prob)
    # print('baselines.shape {}, max_val.shape {}, scale.shape {}'.format(baselines.shape, max_val.shape, scale.shape))

    # if scale.ndim==2: scale = scale.unsqueeze(-1)
    print('spec.shape: ',spec.shape)
    print('baselines.shape: ',baselines.shape)
    print('max_val.shape: ',max_val.shape)

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
    params = torch.ones_like(brainMask).unsqueeze(0).unsqueeze(0).float()#.uniform_(0.2,1)
    
    first = True
    step = 10000
    for i in range(0,totalEntries,step):
        n = 1+step if 1+step<=totalEntries else totalEntries-i
        print('step: ',n)
        outputs = baselines(sample_baselines(n, **baseline_cfg), ppm)

        if first:
            # _, _, baseline, _, _, _ = outputs
            baseline = outputs
            first = False
        else:
            baseline = torch.cat([baseline, outputs], dim=0)# if baseline else None

    '''After baselines have been generated,...'''
    Ns, c, z, y, x = xfAll.shape
    baseline = baseline[0:totalEntries,...]
    ind = torch.squeeze(brainMask)
    temp = torch.zeros_like(xfAll)
    print(type(baseline))
    baseline = torch.movedim(torch.movedim(baseline, source=0, destination=2), source=0, destination=1)
    print(type(baseline))
    print('temp.shape: {}, baseline.shape: {}'.format(temp.shape,baseline.shape))
    
    Ns, ch, z, y, x = xfMeta.shape
    cnt = -1
    for xx in range(x):
        for yy in range(y):
            for zz in range(z):
                if brainMask[zz,yy,xx]:
                    cnt += 1
                    temp[:,:,zz,yy,xx] = baseline[:,:,cnt]
    print('temp.shape: ',temp.shape)
    print('xfMeta.shape: ',xfMeta.shape)
    print('params.shape: ',params.shape)
    print('brainMask.shape: ',brainMask.shape)
    print('ind.shape: ',ind.shape)
    baseline = scale_offsets(spec=xfMeta, baselines=temp, scale=params, drop_prob=0.0, ind=ind)#.numpy()
    
    
    
    # temp[:,:,brainMask] = torch.from_numpy(np.moveaxis(baseline, source=0, destination=1))
    # #baseline = baseline[0:totalEntries,...]
    # # baseline = normalize(baseline[0:totalEntries,...])
    # # print('xfMeta.shape: ', xfMeta.shape)
    # # print('ind.shape: ',ind.shape)
    # # baseline = scale_offsets(spec=xfMeta[ind,...], baselines=baseline, scale=params, drop_prob=0.0).numpy()
    # # baseline = scale_offsets(spec=xfMeta[ind,...], baselines=baseline, scale=params, drop_prob=0.0).numpy()
    # baseline = scale_offsets(spec=xfMeta[...,brainMask], baselines=temp, scale=params, drop_prob=0.0).numpy()
    
    
    # xfAll[ind,...] += baseline
    # xfNuisance[ind,...] += baseline
    # xfMeta[ind,...] += baseline
    # temp[ind,...] += baseline
    
    # xfAll[...,ind] += baseline
    # xfNuisance[...,ind] += baseline
    # xfMeta[...,ind] += baseline
    
    xfAll += baseline
    xfNuisance += baseline
    xfMeta += baseline
    
    return xfAll, xfNuisance, xfMeta, baseline
    
    
    
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
    parser.add_argument('--parentdir', action='store_true', default=False, help='Flag a parent directory containing the matfiles instead of individual file paths')
    
    args = parser.parse_args()
    
    # os.makedirs(args.savedir, exist_ok=True)
    
    if args.parentdir:
        import pathlib
        desktop = pathlib.Path(args.subjectdir)
        lst = ''
        for item in desktop.rglob('*_3T_all.mat'):
            item = str(item)
            # if item.endswith('_all.mat'):
            if lst=='': lst = item
            else: lst += ','+item
        args.subjectdir = lst
    
    folders = [d for d in args.subjectdir.split(',')]
    for d in folders:
        start = time.time()
        # data = io.loadmat(d)
        with h5py.File(d, 'r') as file:
            # data = convert(file)
            data = h5py_to_dict(file)

        Ns, z, y, x = data['xtAll'].shape
        print(data['xtAll'].dtype, data['xtAll'].shape)
        print('xtAll FFT')
        xfAll = np.fft.fftshift(np.fft.fft(data['xtAll']['real'] + 1j*data['xtAll']['imag'], axis=0), axes=0)
        # xfAll = np.fft.fft(np.moveaxis(data['xtAll']['real'] + 1j*data['xtAll']['imag'], source=[0,1,2,3], destination=[3,2,1,0])))
        # print('xfAll: ',xfAll.shape)
        # xfAll = np.reshape(xfAll, [int(x*y*z), 1, Ns])
        # print('xfAll: ',xfAll.shape)
        
        xfAll = torch.from_numpy(np.stack([np.real(xfAll), np.imag(xfAll)], axis=1))
        print('xtNuisance FFT')
        xfNuisance = np.fft.fftshift(np.fft.fft(data['xtNuisance']['real'] + 1j*data['xtNuisance']['imag'], axis=0), axes=0)
        xfNuisance = torch.from_numpy(np.stack([np.real(xfNuisance), np.imag(xfNuisance)], axis=1))
        print('xtMeta FFT')
        xfMeta = np.fft.fftshift(np.fft.fft(data['xtMeta']['real'] + 1j*data['xtMeta']['imag'], axis=0), axes=0)
        xfMeta = torch.from_numpy(np.stack([np.real(xfMeta), np.imag(xfMeta)], axis=1))
        print('T1w')
        Iref = torch.from_numpy(data['Iref'])
        # Iref = np.reshape(np.moveaxis(data['Iref'], source=[0,1,2], destination=[2,1,0]), [int(x*y*z), 1, 1])
        totalEntries = np.sum(data['brainMask'], axis=(0,1,2))
        print('data["brainMask"].shape: ',data["brainMask"].shape)
        # brainMask = np.reshape(np.moveaxis(data['brainMask'], source=[0,1,2], destination=[2,1,0]), [int(x*y*z), 1]).astype(bool)
        brainMask = torch.from_numpy(data['brainMask'])
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
        
        # del xfAll, xfMeta, xfNuisance
    #     print('Finished')
    #     print('Unpacking and preparing to save')
    #     xfAll, xfNuisance, xfMeta, baseline = outputs
    #     xtAll_b = np.reshape(np.fft.ifft(xfAll), [x,y,z,2,Ns])
    #     xtAll_b = xtAll_b[...,0,:] + 1j*xtAll_b[...,1,:]
    #     data.create_dataset(name='xtAll_b', data=xtAll_b, maxshape=(xtAll_b.shape))
    #     del xtAll_b
        
    #     xtNuisance_b = np.reshape(np.fft.ifft(xfNuisance), [x,y,z,2,Ns])
    #     xtNuisance_b = xtNuisance_b[...,0,:] + 1j*xtNuisance_b[...,1,:]
    #     data.create_dataset(name='xtNuisance_b', data=xtNuisance_b, maxshape=(xtNuisance_b.shape))
    #     del xtNuisance_b
        
    #     xtMeta_b = np.reshape(np.fft.ifft(xfMeta), [x,y,z,2,Ns])
    #     xtMeta_b = xtMeta_b[...,0,:] + 1j*xtMeta_b[...,1,:]
    #     data.create_dataset(name='xtMeta_b', data=xtMeta_b, maxshape=(xtMeta_b.shape))
    #     del xtMeta_b
        
    #     baseline = np.reshape(np.fft.ifft(baseline), [x,y,z,2,Ns])
    #     baseline = baseline[...,0,:] + 1j*baseline[...,1,:]
    #     data.create_dataset(name='baseline', data=baseline, maxshape=(baseline.shape))
    #     del baseline
    # # hdf5storage.savemat(d, data, format='7.3', oned_as='column', matlab_compatible=True)
    
    
        print('Finished')
        print('Unpacking and preparing to save')
        data1 = {}
        # data1['xtMeta'] = data['xtMeta']['real'] + 1j*data['xtMeta']['imag']
        xfAll_b, xfNuisance_b, xfMeta_b, baseline = outputs
        # data1['xtAll_b'] = torch.view_as_complex(torch.movedim(xfAll_b,1,-1))# xfAll_b[:,0,...] + 1j*xfAll_b[:,1,...]
        del xfAll_b
        # data1['xtAll_b'] = np.fft.ifft(np.fft.ifftshift(data1['xtAll_b'].numpy(), axes=0), axis=0)
        
        # data1['xtNuisance_b'] = torch.view_as_complex(torch.movedim(xfNuisance_b,1,-1))# xfNuisance_b[:,0,...] + 1j*xfNuisance_b[:,1,...]
        del xfNuisance_b
        # data1['xtNuisance_b'] = np.fft.ifft(np.fft.ifftshift(data1['xtNuisance_b'].numpy(), axes=0), axis=0)
        
        # data1['xtMeta_b'] = torch.view_as_complex(torch.movedim(xfMeta_b,1,-1))# xfMeta_b[:,0,...] + 1j*xfMeta_b[:,1,...]
        del xfMeta_b
        # data1['xtMeta_b'] = np.fft.ifft(np.fft.ifftshift(data1['xtMeta_b'].numpy(), axes=0), axis=0)
        
        data1['baseline'] = torch.view_as_complex(torch.movedim(baseline,1,-1))# baseline[:,0,...] + 1j*baseline[:,1,...]
        del baseline
        data1['baseline'] = np.fft.ifft(np.fft.ifftshift(data1['baseline'].numpy(), axes=0), axis=0)
        
        # data1['brainMask'] = brainMask.numpy()
        # data1['Iref'] = Iref.numpy()
        
        
        print('Finished. Ready to save: ',time.time() - start)
        start = time.time()
        base, ext = os.path.splitext(d)
        new_name = base + '_baselines' + ext
        io.savemat(new_name, mdict=data1, do_compression=False)
        print(time.time() - start)
    #     with h5py.File(new_name, 'w') as file:
    #         for k, v in data.items():
    #             file.create_dataset(k, data=v)
    # # hdf5storage.savemat(d, data, format='7.3', oned_as='column', matlab_compatible=True)
    
'''    
python ./src/baselines_for_fitting_challenge_2024_01.py --config_file './src/config/predefined/MRS_Fitting_Challenge_2024_baselines.json' --subjectdir '/home/john/Downloads/Data/TestSub4/671855_3T_all.mat'

python ./src/baselines_for_fitting_challenge_2024_01.py --config_file './src/config/predefined/MRS_Fitting_Challenge_2024_baselines.json' --subjectdir '/home/john/Downloads/testing_data_2' --parentdir


'''