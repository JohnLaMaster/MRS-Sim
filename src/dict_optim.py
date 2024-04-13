import torch
import torch.nn as nn
from torch.nn import init

from aux import *
from baselines import bounded_random_walk
from interpolate import CubicHermiteMAkima as CubicHermiteInterp

from typing import Callable, List




class NLayerDiscriminator(nn.Module):
    def __init__(self,
                 layers: List=[8, 16, 32, 64],
                 input_nc: int=1,
                #  output_nc: int=8,
                 num_class: int=1,
                 data_length: int=512,
                ) -> None:
        stem = [nn.Conv1d(in_channels=input_nc, out_channels=layers[0], kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm1d(num_features=layers[0]),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        self.stem = nn.Sequential(*stem)
        data_length //= 2
        
        critic = []
        for i in range(len(layers)-1):
            critic.append(nn.Conv1d(in_channels=layers[i], out_channels=layers[i+1], kernel_size=4, 
                                    stride=2, padding=1, bias=True))
            critic.append(nn.InstanceNorm1d(num_features=layers[i+1]))
            critic.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            data_length //= 2
        
        critic.append(nn.Conv1d(in_channels=layers[-1], out_channels=1, kernel_size=4, 
                                stride=2, padding=1, bias=True))
        critic.append(nn.InstanceNorm1d(num_features=1))
        critic.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        data_length //= 2
        
        # critic.append(nn.AdaptiveAvgPool1d(output_size=output_nc))    
        critic.append(nn.Flatten())
        critic.append(nn.Linear(in_features=data_length,out_features=num_class))
        self.critic = nn.Sequential(*critic)
        
        init_weights(self, init_type='normal', activation='leaky_relu')
        
    def forward(self,
                data: torch.Tensor,
               ) -> torch.Tensor:
        return self.critic(self.stem(data))     


class Generator(nn.Module):
    def __init__(self, 
                 function: Callable([dict], torch.Tensor), #baselines or residual_water
                 cfg: dict
                ) -> None:
        super().__init__()
        self.function = self.baselines if function=="baselines" else self.residual_water
        self.register_parameter('start', nn.Parameter(torch.as_tensor([cfg['start'][0], cfg['start'][1]]), requires_grad=True))
        self.register_parameter('end',   nn.Parameter(torch.as_tensor([cfg['end'][0],   cfg['end'][1]]),   requires_grad=True))
        self.register_parameter('std',   nn.Parameter(torch.as_tensor([cfg['std'][0],   cfg['std'][1]]),   requires_grad=True))
        self.register_parameter('upper_bnd', nn.Parameter(torch.as_tensor([cfg['upper_bnd'][0], cfg['upper_bnd'][1]]), requires_grad=True))
        self.register_parameter('lower_bnd', nn.Parameter(torch.as_tensor([cfg['lower_bnd'][0], cfg['lower_bnd'][1]]), requires_grad=True))
        self.register_parameter('windows', nn.Parameter(torch.as_tensor([cfg['windows'][0], cfg['windows'][1]]), requires_grad=True))
        self.register_parameter('scale', nn.Parameter(torch.as_tensor([cfg['scale'][0], cfg['scale'][1]]), requires_grad=False))
        self.register_parameter('prime', nn.Parameter(torch.as_tensor(cfg['prime']), requires_grad=True))
        self.register_parameter('pt_density', nn.Parameter(torch.as_tensor(cfg['pt_density']), requires_grad=True))
        self.register_buffer('ppm_range', torch.as_tensor([cfg['ppm_range'][0], cfg['ppm_range'][1]]))
        self.register_buffer('rand_omit', torch.as_tensor(cfg['drop_prob']))
        self.register_buffer('zero',      torch.zero(1))
        
    def sample(self, 
               N: int=1
              ) -> dict:
        primes = torch.distributions.uniform(self.zero, self.prime).rsample([N,2])
        primes[:,0] = -1 * primes[:,0]
        ppm_range = self.ppm_range + primes
        length = ((ppm_range[1] - ppm_range[0]) * self.pt_density).round()
        if length % 2 != 0: length += 1
        return {
            'start':     torch.distributions.uniform(self.start[0],  self.start[1]).rsample(N),
            'end':       torch.distributions.uniform(self.end[0],    self.end[1]).rsample(N),
            'std':       torch.distributions.uniform(self.std[0],    self.std[1]).rsample(N),
            'upper_bnd': torch.distributions.uniform(self.upper[0],  self.upper[1]).rsample(N),
            'lower_bnd': torch.distributions.uniform(self.lower[0],  self.lower[1]).rsample(N),
            'windows':   torch.distributions.uniform(self.window[0], self.window[1]).rsample(N),
            'length':    length,
            'ppm_range': ppm_range,
            'scale':     torch.distributions.uniform(self.scale[0],  self.scale[1]).rsample(N),
            'rand_omit': self.rand_omit,
        }
        
    def export_params(self):
        return {
            'start':      [self.start[0],    self.start[1]],
            'end':        [self.end[0],      self.end[1]],
            'prime':      [self.prime],
            'std':        [self.std[0],      self.std[1]],
            'upper_bnd':  [self.upper[0],    self.upper[1]],
            'lower_bnd':  [self.lower[0],    self.lower[1]],
            'windows':    [self.window[0],   self.window[1]],
            'pt_density': [self.pt_density],
            'ppm_range':  [self.ppm_range],
            'scale':      [self.scale[0],    self.scale[1]],
            'rand_omit':  [self.rand_omit],
        }
        
    def forward(self, 
                N: int=1
               ) -> torch.Tensor:
        return self.function(self.sample(N))
    
    
    def baselines(self,
                cfg: dict,
                ) -> tuple():
        '''
        Simulate baseline offsets
        '''
        baselines = batch_smooth(
                        bounded_random_walk(cfg['start'], 
                                            cfg['end'], 
                                            cfg['std'], 
                                            cfg['lower_bnd'], 
                                            cfg['upper_bnd'], 
                                            cfg['length']), 
                                    cfg['windows'], 'constant')

        # Subtract the trend lines 
        trend = batch_linspace(baselines[...,0].unsqueeze(-1),
                            baselines[...,-1].unsqueeze(-1), 
                            cfg['length'])
        baselines = baselines - trend

        baselines, _ = self.normalize(signal=baselines, fid=False, denom=None, noisy=-3)

        if cfg['rand_omit']>0: 
            baselines, _ = rand_omit(baselines, 0.0, cfg['rand_omit'])


        # Convert simulated residual water from local to clinical range before 
        # Hilbert transform makes the imaginary component. Then resample 
        # acquired range to cropped range.
        # ppm_range =  [torch.as_tensor(val) for val in cfg.ppm_range]
        raw_baseline = HilbertTransform(
                        sim2acquired(baselines * cfg['scale'], 
                                    [cfg['ppm_range'][0], cfg['ppm_range'][1]],
                                    self.ppm)
                        )
        
        ch_interp = CubicHermiteInterp(xaxis=self.ppm, signal=raw_baseline)
        baselines = ch_interp.interp(xs=self.ppm_cropped)
        
        return raw_baseline.fliplr(), raw_baseline


    def residual_water(self,
                       cfg: dict,
                      ) -> tuple():
        start_prime = cfg['ppm_range'][0]# + cfg.start_prime
        end_prime   = cfg['ppm_range'][1]# -   cfg.end_prime
        ppm = batch_linspace(start=start_prime, stop=end_prime, steps=cfg['length'].int())
        res_water = batch_smooth(bounded_random_walk(cfg['start'], 
                                                    cfg['end'], 
                                                    cfg['std'], 
                                                    cfg['lower_bnd'], 
                                                    cfg['upper_bnd'], 
                                                    cfg['length']),
                                cfg['windows'], 'constant') 
        trend = batch_linspace(res_water[...,0].unsqueeze(-1),
                            res_water[...,-1].unsqueeze(-1), 
                            cfg['length'])
        res_water -= trend
        res_water, _ = self.normalize(signal=res_water, fid=False, denom=None, noisy=-3)

        if cfg['rand_omit']>0: 
            res_water, ind = rand_omit(res_water, 0.0, cfg['rand_omit'])

        # Convert simulated residual water from local to clinical range before 
        # Hilbert transform makes the imaginary component. Then resample 
        # acquired range to cropped range.
        raw_res_water = HilbertTransform(
                            sim2acquired(res_water * cfg['scale'], 
                                        [start_prime, 
                                            end_prime], self.ppm)
                        )
        # ch_interp = CubicHermiteInterp(xaxis=self.ppm, signal=raw_res_water)
        # res_water = ch_interp.interp(xs=self.ppm_cropped)

        return raw_res_water, raw_res_water   # raw_res_water is flat on the tails


def init_weights(net, init_type='normal', init_gain=0.02, activation='leaky_relu'):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity=activation)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    
def enforce_divisibility(input, divisor):
    if not 1 / (input % divisor) == 0:
        rem = 1 / (input % divisor) / 10
        if rem >= 0.5:
            input += (1 - rem) * divisor
        else:
            input -= rem * divisor
    return int(input)   

# def prepareConfig(N: int, cfg: dict) -> dict:#, pt_density: int):
#     # try: cfg['pt_density']
#     # except: cfg['pt_density'] = pt_density
#     # print('N = ',N)
#     length = (cfg['ppm_range'][1]-cfg['ppm_range'][0]) * cfg['pt_density']
#     length = length.round
#     if length % 2 != 0: length += 1
#     if len(cfg['start'])==1: cfg['start'] = [cfg['start'][0], cfg['start'][0]]
#     if len(cfg['end'])==1: cfg['end'] = [cfg['end'][0], cfg['end'][0]]
#     if len(cfg['std'])==1: cfg['std'] = [cfg['std'][0], cfg['std'][0]]
#     if len(cfg['upper'])==1: cfg['upper'] = [cfg['upper'][0], cfg['upper'][0]]
#     if len(cfg['lower'])==1: cfg['lower'] = [cfg['lower'][0], cfg['lower'][0]]
#     if len(cfg['window'])==1: cfg['window'] = [cfg['window'][0], cfg['window'][0]]
#     if len(cfg['scale'])==1: cfg['scale'] = [cfg['scale'][0], cfg['scale'][0]]
#     cfg['ppm_range'] = [torch.as_tensor([val]) for val in cfg['ppm_range']]

#     return {
#         'start': torch.zeros(N,1,1).uniform_(cfg['start'][0],
#                                              cfg['start'][1]),
#         'end': torch.zeros(N,1,1).uniform_(cfg['end'][0],
#                                            cfg['end'][1]),
#         'std': torch.zeros(N,1,1).uniform_(cfg['std'][0],
#                                            cfg['std'][1]),
#         'upper_bnd': torch.ones(N,1,1).uniform_(cfg['upper'][0],
#                                                 cfg['upper'][1]),
#         'lower_bnd': torch.ones(N,1,1).uniform_(cfg['lower'][0],
#                                                 cfg['lower'][1]),
#         'windows': torch.ones(N,1,1).uniform_(cfg['window'][0],
#                                               cfg['window'][1]),
#         'length': length,
#         'ppm_range': cfg['ppm_range'],
#         'scale': torch.ones(N,1,1).uniform_(cfg['scale'][0],
#                                             cfg['scale'][1]),
#         'rand_omit': cfg['drop_prob'],
#     }


# def sample_baselines(N: int, **cfg): 
#     if not isinstance(cfg, type(None)):
#         try: cfg['pt_density']
#         except: cfg['pt_density'] = 128
#         dct = prepareConfig(N=N, cfg=cfg)#, pt_density=cfg['pt_density'])
#         return dct
#     return None


# def sample_resWater(N: int, **cfg):
#     if not isinstance(cfg, type(None)):
#         try: cfg['pt_density']
#         except: cfg['pt_density'] = 1204
#         dct = prepareConfig(N=N, cfg=cfg)#, pt_density=cfg['pt_density'])
#         # start, _ = rand_omit(torch.zeros(N,1,1).uniform_(0,cfg['prime']), 
#         #                      0.0, cfg['drop_prob'])
#         # end, _   = rand_omit(torch.zeros(N,1,1).uniform_(0,cfg['prime']), 
#         #                      0.0, cfg['drop_prob'])
#         start, _ = rand_omit(torch.zeros(N,1,1).uniform_(-1*cfg['prime'],
#                                                          cfg['prime']), 
#                              0.0, cfg['drop_prob'])
#         end, _   = rand_omit(torch.zeros(N,1,1).uniform_(-1*cfg['prime'],
#                                                          cfg['prime']), 
#                              0.0, cfg['drop_prob'])
#         dct.update({#'cropRange_resWater': cfg['cropRange_water'],
#                     'start_prime': start,
#                     'end_prime': end})

#         return dct
#     return None


# def baselines(self,
#               cfg: dict,
#              ) -> tuple():
#     '''
#     Simulate baseline offsets
#     '''
#     baselines = batch_smooth(
#                     bounded_random_walk(cfg['start'], 
#                                         cfg['end'], 
#                                         cfg['std'], 
#                                         cfg['lower_bnd'], 
#                                         cfg['upper_bnd'], 
#                                         cfg['length']), 
#                                 cfg['windows'], 'constant')

#     # Subtract the trend lines 
#     trend = batch_linspace(baselines[...,0].unsqueeze(-1),
#                            baselines[...,-1].unsqueeze(-1), 
#                            cfg['length'])
#     baselines = baselines - trend

#     baselines, _ = self.normalize(signal=baselines, fid=False, denom=None, noisy=-3)

#     if cfg.rand_omit>0: 
#         baselines, _ = rand_omit(baselines, 0.0, cfg['rand_omit'])


#     # Convert simulated residual water from local to clinical range before 
#     # Hilbert transform makes the imaginary component. Then resample 
#     # acquired range to cropped range.
#     # ppm_range =  [torch.as_tensor(val) for val in cfg.ppm_range]
#     raw_baseline = HilbertTransform(
#                     sim2acquired(baselines * cfg['scale'], 
#                                  [cfg['ppm_range'][0], cfg['ppm_range'][1]],
#                                  self.ppm)
#                     )
    
#     ch_interp = CubicHermiteInterp(xaxis=self.ppm, signal=raw_baseline)
#     baselines = ch_interp.interp(xs=self.ppm_cropped)
    
#     return raw_baseline.fliplr(), raw_baseline


# def residual_water(self,
#                    cfg: dict,
#                   ) -> tuple():
#     start_prime = cfg['ppm_range'][0]# + cfg.start_prime
#     end_prime   = cfg['ppm_range'][1]# -   cfg.end_prime
#     ppm = batch_linspace(start=start_prime, stop=end_prime, 
#                             steps=cfg['length'].int()))
#     res_water = batch_smooth(bounded_random_walk(cfg['start'], 
#                                                  cfg['end'], 
#                                                  cfg['std'], 
#                                                  cfg['lower_bnd'], 
#                                                  cfg['upper_bnd'], 
#                                                  cfg['length']),
#                              cfg['windows'], 'constant') 
#     trend = batch_linspace(res_water[...,0].unsqueeze(-1),
#                            res_water[...,-1].unsqueeze(-1), 
#                            cfg['length'])
#     res_water -= trend
#     res_water, _ = self.normalize(signal=res_water, fid=False, denom=None, noisy=-3)

#     if cfg.rand_omit>0: 
#         res_water, ind = rand_omit(res_water, 0.0, cfg['rand_omit'])

#     # Convert simulated residual water from local to clinical range before 
#     # Hilbert transform makes the imaginary component. Then resample 
#     # acquired range to cropped range.
#     raw_res_water = HilbertTransform(
#                         sim2acquired(res_water * cfg['scale'], 
#                                      [start_prime, 
#                                         end_prime], self.ppm)
#                     )
#     # ch_interp = CubicHermiteInterp(xaxis=self.ppm, signal=raw_res_water)
#     # res_water = ch_interp.interp(xs=self.ppm_cropped)

#     return raw_res_water, raw_res_water   # raw_res_water is flat on the tails

