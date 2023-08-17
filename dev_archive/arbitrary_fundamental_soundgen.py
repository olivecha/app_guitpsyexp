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

wood_sounds = du.load_wood_sounds()
wood_sounds = du.arr2notedict(wood_sounds)
notes = du.get_notes_names()


# FUNCTIONS
def resample_An_time(An_lo, ct_lo, ct_hi):
    """ 
    Resample An data to a higher sample rate 
    An_lo : An data with low sample rate
    ct_lo : time values with low sample rate
    ct_hi : time values with high sample rate
    """
    resamp_An = []
    # For each fixed frequency values in An low sample rate
    for An_ti in An_lo:
        # create an interpolator 
        spl = interp1d(ct_A, An_ti, bounds_error=False, fill_value=(An_ti[0], An_ti[-1]))
        # evaluate it at the new center times values
        An_new = spl(ct_D)
        resamp_An.append(An_new)
    resamp_An = np.array(resamp_An)
    return resamp_An

def joint_timesample_resampling():
    s = wood_sounds['A5']
    all_amps, center_times = get_An_data(s)
    new_sig = signal_from_An(center_times, peak_frequencies(s), all_amps, onset_env=0)
    print('A string generated sound')
    du.listen_sig_array(new_sig)
    
    An_A, ct_A = get_An_data(s)
    _, ct_D = get_An_data(wood_sounds['D4'])
    # interpolate A to have the same number of center times as D
    resamp_An_A = []
    for An_ti in An_A:
        spl = interp1d(ct_A, An_ti, bounds_error=False, fill_value=(An_ti[0], An_ti[-1]))
        An_new = spl(ct_D)
        resamp_An_A.append(An_new)
    resamp_An_A = np.array(resamp_An_A)
    new_sig = signal_from_An(ct_D, peak_frequencies(s), resamp_An_A, onset_env=0)
    print('A with resampled time (same center times as D sound)')
    du.listen_sig_array(new_sig)
    
    s = wood_sounds['D4']
    all_amps, center_times = get_An_data(s)
    new_sig = signal_from_An(center_times, peak_frequencies(s), all_amps, onset_env=0)
    print('D string generated sound')
    du.listen_sig_array(new_sig)
    

def frequency_vector_align():
    pf_A, pf_D = peak_frequencies(wood_sounds['A5']), peak_frequencies(wood_sounds['D4'])
    plt.scatter(pf_A, range(pf_A.size), label='A')
    plt.scatter(pf_D, range(pf_D.size), label='D')
    plt.title('Peak frequency values for the two interpolation signals')
    du.bigfig()
    plt.legend()
        

def align_peak_frequencies(pf_1, pf_2, max_iter=1, sanitize=True, max_err=500):
    """
    Align two peak frequency vectors until their maximum error is at the end of the array
    pf_1 : peak frequencies with the lowest fundamental
    pf_2 : peak frequencies with the highest fundamental
    """
    # Check if the input is clean
    if pf_1[0] > pf_2[0]:
        raise ValueError('first peak frequency vector must have lower fundamental')
    # just to be sure
    pf_1 = np.array(pf_1)
    pf_2 = np.array(pf_2)
    pf_arrs = {'pf1':pf_1, 'pf2':pf_2}
    min_size = np.min([pf_1.size, pf_2.size])
    # Theoretical difference
    pf_1_th = np.arange(pf_1[0], pf_1[-1], pf_1[0])
    pf_2_th = np.arange(pf_2[0], pf_2[-1], pf_2[0])
    diff_th = pf_2_th[:min_size] - pf_1_th[:min_size]
    # Actual difference
    diff = pf_2[:min_size] - pf_1[:min_size]
    # Error
    error = np.abs(diff_th - diff)
    if np.max(error) > max_err:
        # find where the max error gradient is
        idx_max_err = np.argmax(np.gradient(error))
        # find which array to remove values (lowest value at max error gradient)
        which_trans = np.argmin([pf_1[idx_max_err], pf_2[idx_max_err]])
        which_trans_key = ['pf1', 'pf2'][which_trans]
        which_notrans_key = [ky for ky in ['pf1', 'pf2'] if ky != which_trans_key][0]
        arr_trans = pf_arrs[which_trans_key]
        arr_notrans = pf_arrs[which_notrans_key]
        # Goal value for the reindexing
        value_target = arr_notrans[idx_max_err]
        # Find the index of the closest value
        which_close = np.argmin(np.abs(arr_trans-value_target))
        # Alterate the array so that the closest value is at idx_max_err
        arr_trans = np.hstack([arr_trans[:idx_max_err], arr_trans[(which_close):]])
        # reassing the new array
        pf_arrs[which_trans_key] = arr_trans
        # find the pf_1 and pf_2 arrays
        pf_1 = pf_arrs['pf1']
        pf_2 = pf_arrs['pf2']
        min_size = np.min([pf_1.size, pf_2.size])
        # recompute the error
        diff = pf_2[:min_size] - pf_1[:min_size]
        error = np.abs(diff_th[:diff.size] - diff)

    pf_1, pf_2 = pf_1[:min_size], pf_2[:min_size]
        
    if sanitize == True:
        # sanitize according to difference
        diff = pf_2 - pf_1

        return pf_1[diff>10], pf_2[diff>10]
    else:
        return pf_1, pf_2
    
def frequency_alignment_visualization():
    pf_Aog, pf_Dog = peak_frequencies(wood_sounds['A5']), peak_frequencies(wood_sounds['D4'])
    plt.scatter(np.arange(pf_Aog.size), pf_Aog,  label='pf_A init')
    plt.scatter(np.arange(pf_Dog.size), pf_Dog,  label='pf_D init')
    pf_A1, pf_D1 = align_peak_frequencies(pf_Aog, pf_Dog, sanitize=False, max_err=0)
    plt.scatter(np.arange(pf_A1.size), pf_A1,  label='pf_A align')
    plt.scatter(np.arange(pf_D1.size), pf_D1,  label='pf_D align')
    
    plt.legend()
    du.bigfig()
    

def pair_value_sanitizing():
    plt.plot(pf_D1, label = 'D')
    plt.plot(pf_A1, label = 'A')
    diff = pf_D1 - pf_A1
    plt.scatter(np.arange(pf_D1.size)[diff>0], np.mean([pf_D1, pf_A1], axis=0)[diff>0], label='nonzero diff values')
    plt.legend()
    plt.title('Values having acceptable difference')
    du.bigfig()
    
    
def arg_indexed_array(pf_new, pf_old):
    """ Find the logical array so that pf_old[logical_arr] = pf_new """
    logical_arr = [False for _ in range(pf_old.size)]
    for pfn in pf_new:
        log_i = [np.isclose(pfn, pfo) for pfo in pf_old]
        logical_arr = np.logical_or(logical_arr, log_i)
    return logical_arr


def interpolated_peak_frequency_visualization():
    s_A = wood_sounds['A5']
    s_D = wood_sounds['D4']
    pf_A = peak_frequencies(s_A)
    pf_D = peak_frequencies(s_D)
    pf_A_rs, pf_D_rs = align_peak_frequencies(pf_A, pf_D, max_err=100)
    
    plt.plot(pf_A_rs, label='new A pf values')
    plt.plot(pf_A[arg_indexed_array(pf_A_rs, pf_A)] + 200, label='indexed D old (offset)')
    
    plt.plot(pf_D_rs, label='new D pf values')
    plt.plot(pf_D[arg_indexed_array(pf_D_rs, pf_D)] + 200, label='indexed D old (offset)')
    plt.legend()
    du.bigfig()
    
    
def interpolated_An_construction():
    # Note somewhat in between the A and S string
    C3_freq = 130.81 # Hz
    s_A = wood_sounds['A5']
    s_D = wood_sounds['D4']
    
    An_A, ct_A = get_An_data(s_A)
    An_D, ct_D = get_An_data(s_D)
    pf_A, pf_D = peak_frequencies(s_A), peak_frequencies(s_D)
    
    # interpolate A to have the same number of center times as D
    An_A = resample_An_time(An_A, ct_A, ct_D)
    
    # align the two frequency vectors
    pf_A_rs, pf_D_rs = align_peak_frequencies(pf_A, pf_D, max_err=10000)
    
    # construct the pf_C array
    # theoretical pf_C values
    pf_C_th = np.array([C3_freq*i for i in range(1, int(pf_D_rs[-1]//C3_freq + 2))])
    pf_C = []
    pf_A_corr = []
    pf_D_corr = []
    for pf_Ai, pf_Di in zip(pf_A_rs, pf_D_rs):
        try:
            possible_values = pf_C_th[np.logical_and(pf_C_th > pf_Ai, pf_C_th < pf_Di)]
            pf_C.append(possible_values[possible_values.size//2])
            pf_A_corr.append(pf_Ai)
            pf_D_corr.append(pf_Di)
        except IndexError:
            continue
    pf_C, idxs_C_unique = np.unique(pf_C, return_index=True)
    pf_A_corr = np.array(pf_A_corr)[idxs_C_unique]
    pf_D_corr = np.array(pf_D_corr)[idxs_C_unique]
        
    # find the mapping arrays
    pf_A_mp = arg_indexed_array(pf_A_corr, pf_A)
    pf_D_mp = arg_indexed_array(pf_D_corr, pf_D)
    
    # resample the An values
    An_A = An_A[pf_A_mp, :]
    An_D = An_D[pf_D_mp, :]
    
    # Interpolate the An_C array
    An_C = []
    for i in range(ct_D.size):
        An_Ai = An_A[:, i]
        An_Di = An_D[:, i]
        An_Ci = []
        for pfai, pfdi, pfci, Ani, Dni in zip(pf_A_corr, pf_D_corr, pf_C, An_Ai, An_Di):
            An_val = np.interp(pfci, [pfai, pfdi], [Ani, Dni])
            An_Ci.append(An_val)
        An_C.append(An_Ci)
    An_C = np.transpose(An_C)
    
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    
    ax = axs[0]
    ax.set_title('A string An data')
    ax.pcolormesh(ct_D, pf_A_corr,  np.log(An_A))
    
    ax = axs[1]
    ax.set_title('C string An data')
    ax.pcolormesh(ct_D, pf_C,  np.log(An_C))
    
    ax = axs[2]
    ax.set_title('D string An data')
    ax.pcolormesh(ct_D, pf_D_corr, np.log(An_D))
    
    for ax in axs:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        

def interpolated_frequency_soundgen():
    print('resampled A5 sound : ')
    new_sig = signal_from_An(ct_D, pf_A_corr, An_A)
    du.listen_sig_array(new_sig)
    
    print('resampled D4 sound : ')
    new_sig = signal_from_An(ct_D, pf_D_corr, An_D)
    du.listen_sig_array(new_sig)
    
    print('Interpolated C sound : ')
    new_sig = signal_from_An(ct_D, pf_C, An_C)
    du.listen_sig_array(new_sig)