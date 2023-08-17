####################################
# 2D An interpolator perturbations
####################################

import guitarsounds
from guitarsounds import Sound
import sys, os
import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import librosa.display
import DEV_utils as du
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, interp2d, InterpolatedUnivariateSpline, RectBivariateSpline
import scipy.signal


def subsig_divide(s, max_time=2):
    """ Divide a sound in sub signals """
    n_intervals = int(du.get_arrgen_interval(s) * max_time)
    time_intervals = np.linspace(0.13, max_time, n_intervals)
    center_times = [np.mean([time_intervals[i], time_intervals[i + 1]]) for i in range(len(time_intervals) - 1)]
    # Sound division in sub intervals
    sub_sigs = []
    for i, _ in enumerate(time_intervals[:-1]):
        # create a signal from subset indexes
        idx1 = du.time_index(s.signal, time_intervals[i])
        idx2 = du.time_index(s.signal, time_intervals[i + 1])
        new_sig = guitarsounds.Signal(s.signal.signal[idx1:idx2], s.signal.sr, s.SP)
        sub_sigs.append(new_sig)
    return sub_sigs, center_times

def get_An_data(s, max_time=2):
    """ An data from a list of sub signals with peak freqs """
    sub_sigs, center_times = subsig_divide(s, max_time=max_time)
    peak_freqs = peak_frequencies(s)
    all_amps = []
    for i, pf in enumerate(peak_freqs):
        # Time based amplitudes
        time_amps = []
        for sig in sub_sigs:
            fidx = du.frequency_index(sig, pf)
            time_amps.append(du.real_fft(sig)[fidx])
        all_amps.append(time_amps)
    return np.array(all_amps), np.array(center_times)

def signal_from_An(center_times, peak_freqs, all_amps,  max_time=2, onset_env=3):
    """ Construct a signal array from an interpolator """
    An_itrp_2D = interp2d(center_times, peak_freqs, all_amps, kind='cubic')
    t = signal_time(max_time)
    An_values = An_itrp_2D(t, peak_freqs)
    
    new_sig = 0
    for i, pf in enumerate(peak_freqs):
        new_sig += An_values[i]* np.sin(pf * t * 2 * np.pi)
        
    new_sig = apply_onset_ifft(new_sig, onset_env)
    new_sig = fadeout_sigarr(new_sig)
    return new_sig

def perturbation_1D():
    s = wood_sounds['A5']
    
    all_amps, center_times = get_An_data(s)
    
    # Add a perturbation
    pert = np.linspace(10*np.ones_like(center_times), 1*np.ones_like(center_times), all_amps.shape[0])
    all_amps = all_amps*pert
    
    new_sig = signal_from_An(center_times, peak_frequencies(s), all_amps, onset_env=0)
            
    
    print('Original')
    s.signal.normalize().listen()
    print('Generated')
    du.listen_sig_array(new_sig)
    
    
def perturbation_2D():
    s = wood_sounds['A5']

    all_amps, center_times = get_An_data(s)
    
    # Add a 2D perturbation
    pert = np.linspace(10*np.linspace(1, 0.2, len(center_times)), 1*np.linspace(1, 0.2, len(center_times)), all_amps.shape[0])
    all_amps = all_amps*pert
    
    new_sig = signal_from_An(center_times, peak_frequencies(s), all_amps, onset_env=0)
            
    
    print('Original')
    s.signal.normalize().listen()
    print('Generated')
    du.listen_sig_array(new_sig)