# imports
import numpy as np
from soundfile import write
from scipy.interpolate import interp1d


def random_envelop_parameter(other=None, bounds=(-1, 1)):
    """
    Generate a random number to be used to generate the onset envelop
    """
    interval = bounds[1] - bounds[0]
    param = np.random.rand()*interval + bounds[0]
    return param


def generate_chord(env_param, strum_interval=0.3, sr=22050, which='wood'):
    """
    Generate a chord from sound arrays using an envelop parameter
    """
    notes = ['E6', 'A5', 'D4', 'G3', 'B2', 'E1']
    arrays = [np.load(f'assets/sound_arrays/{which}/{note}.npy') for note in notes]
    array_size = arrays[0].shape[0]
    strum_sample_interval = int(strum_interval * sr)
    chord = np.zeros(array_size + 6 * strum_sample_interval)
    i = 0
    correction = [1, 0.95, 0.94, 0.90, 0.88, 0.80]
    for j, arr in enumerate(arrays):
        arr = apply_onset_to_array(arr, env_param)
        arr = fadeout_sigarr(arr)
        chord[i:i + array_size] += arr * correction[j]
        i += strum_sample_interval

    chord *= 0.95 / np.max(np.abs(chord))
    return chord


def apply_onset_to_array(arr, env_param=0, sr=22050):
    """
    Apply a onset envelop with env_param to a sound array
    """
    env = get_expenv(env_param)
    # Apply it to time = 0.0 - 0.1 s
    t_idx = int(sr * 0.1)
    time = np.arange(0, 0.1, 1/sr)
    arr[:t_idx] = env(time[:t_idx]) * arr[:t_idx]
    return arr


def fadeout_sigarr(arr,fadeout_time=0.05, sr=22050):
    """
    Fades out the end of a signal array
    fadeout_time : time length being faded out
    """
    n_samples = int(fadeout_time * sr)
    fadeout = np.linspace(1, 0.1, n_samples)
    arr[-n_samples:] *= fadeout
    return arr


def save_wav(name, arr, sr=22050):
    """
    Create a soundfile from a signal
    :param name: the name of the saved file
    :param arr: array to save as a wavefile
    """
    write(name + ".wav", arr, sr)


def get_expenv(p):
    """ 
    Generate an exponential onset based on curves fitted on real signals
    :param p: a float, if p is between -1 and 1 the fitted envelop will be 
    within the experimental range
    for p > 1 or p < -1 the shape of the onset envelop is extrapolated.
    """
    # Hard coded bounds on envelop onset parameters
    a_min = 1.216110966595552e-21
    a_max = 2.956038122585772e-11
    b_min = 241.88870914170292
    b_max = 476.8504184761331
    
    if p > 1:
        p = np.sqrt(p)
        # a_max is the highest curve
        a = a_min + (p / 2 + 0.5) * (a_max - a_min)
        # b_min is the highest curve
        b = b_min + (1 - (p / 2 + 0.5)) * (b_max - b_min)
        
    elif p < -1:
        b = b_min + (1 - (p / 2 + 0.5)) * (b_max - b_min)
        p = 1 / np.abs(p)
        a = a_min + (p / 2 + 0.5) * (a_max - a_min)
        
    else:
        # a_max is the highest curve
        a = a_min + (p / 2 + 0.5) * (a_max - a_min)
        # b_min is the highest curve
        b = b_min + (1 - (p / 2 + 0.5)) * (b_max - b_min)
            
    # Correct the value at t = 0.1 at run time
    t1 = np.linspace(0, 0.6, 1000)
    env_exp_draft = a * np.exp(t1 * b)
    itrp = interp1d(env_exp_draft, t1)
    offset = itrp(1) - 0.1
    
    # Create the callable for the current envelop value
    def expenv(t):
        return a * np.exp((t + offset) * b)
    
    return expenv

