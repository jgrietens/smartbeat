import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.signal import resample as scipy_resample
import numpy as np
import math


def high_pass(timestamps, data, fs, cutoff_fs):
    if len(timestamps) <= 1 or fs <= 0:
        return data

    wp = cutoff_fs / (fs / 2.0)
    b1, a1 = butter(2, wp, 'high')

    return filtfilt(b1, a1, data, padtype='constant', axis=0)


def low_pass(timestamps, data, fs, cutoff_fs):
    if len(timestamps) <= 1 or fs <= 0:
        return data
    print('Wn = ' + str(
    cutoff_fs = min(fs / 2.0, cutoff_fs)
    wp = cutoff_fs / (fs / 2.0)
    
    import pdb; pdb.set_trace() #delete this line
    print(wp) # delete this line
  
   
        
    b1, a1 = butter(2, wp, 'low')

    return filtfilt(b1, a1, data, padtype='constant', axis=0)


def _interpolate_linear(x, y, x_new):
    """
    copied from scipy
    Interpolate a function linearly given a new x.
    :param x: the original x of the function
    :param y: the original y of the function
    :param x_new: the new x to interpolate to
    :return: interpolated y-values on x_new
    """

    # 1. convert to numpy arrays if necessary.
    x = np.asarray(x)
    y = np.asarray(y)
    if len(y) == 0 or len(x) == 0:
        return y

    y_shape = y.shape
    y = y.reshape((y.shape[0], -1))

    # 2. Find where in the original data, the values to interpolate
    #    would be inserted.
    #    Note: If x_new[n] == x[m], then m is returned by searchsorted.
    x_new_indices = np.searchsorted(x, x_new)

    # 3. Clip x_new_indices so that they are within the range of
    #    x indices and at least 1.  Removes mis-interpolation
    #    of x_new[n] = x[0]
    x_new_indices = x_new_indices.clip(1, len(x) - 1).astype(int)

    # 4. Calculate the slope of regions that each x_new value falls in.
    lo = x_new_indices - 1
    hi = x_new_indices

    x_lo = x[lo]
    x_hi = x[hi]
    y_lo = y[lo]
    y_hi = y[hi]

    # Note that the following two expressions rely on the specifics of the
    # broadcasting semantics.
    slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]

    # 5. Calculate the actual value for each entry in x_new.
    y_new = slope * (x_new - x_lo)[:, None] + y_lo

    return y_new.reshape(x_new.shape + y_shape[1:])


SECOND_TO_TIMESTAMP = 1000000000.0


def interpolate(timestamps, data, fs):
    if len(timestamps) <= 1 or fs <= 0:
        return timestamps, data

    first_stamp = timestamps[0]

    srate = SECOND_TO_TIMESTAMP / fs
    xnew = np.arange(first_stamp, timestamps[-1], srate)

    data = _interpolate_linear(timestamps, data, xnew)

    return xnew, data


def _nearest_pow2(x):
    if x == 0:
        return 0

    _ceil = math.ceil(math.log(x) / math.log(2))
    return int(math.pow(2, _ceil))


def _resample(timestamps, data, orig_fs, target_fs):
    len_ts = len(timestamps)

    if len_ts <= 1 or orig_fs <= 0 or target_fs <= 0:
        return timestamps, data

    first_stamp = timestamps[0]

    ratio = float(target_fs) / orig_fs
    len_new = int(np.ceil(len_ts * ratio))
    len_padding = _nearest_pow2(len(timestamps)) - len_ts
    len_interp = int(np.ceil((len_ts + len_padding) * ratio))

    data_padded = np.pad(data, ((0, len_padding), (0, 0)), mode='constant')
    data_resampled = scipy_resample(data_padded, len_interp, axis=0)[:len_new]

    step_size = long(SECOND_TO_TIMESTAMP / target_fs)
    timestamps = long(first_stamp) + np.arange(start=0,
                                               stop=len(data_resampled ) * step_size,
                                               step=step_size,
                                               dtype=long)

    return timestamps, data_resampled


def _sort(timestamps, data):
    idx = np.argsort(timestamps)
    return timestamps[idx], data[idx]


def _deduplicate(timestamps, data):
    timestamps, idx = np.unique(timestamps, return_index=True)
    return timestamps, data[idx]


def resample(timestamps, data, target_fs):
    orig_fs = np.round(1.0 / (np.diff(timestamps).mean() / SECOND_TO_TIMESTAMP), 1)

    # Sort the values by their timestamp
    timestamps, data = _sort(timestamps, data)

    # Remove duplicates
    timestamps, data = _deduplicate(timestamps, data)

    # Calculate the most efficient interpolation sampling rate
    interpolation_fs = _nearest_pow2(int(round(orig_fs / target_fs))) * target_fs

    # Interpolate teh data
    timestamps, data = interpolate(timestamps, data, interpolation_fs)

    # Apply low pass filter for half the target sampling rate
    data = low_pass(timestamps, data, interpolation_fs, target_fs / 2.0)

    # Resample to the target sampling rate
    timestamps, data = _resample(timestamps, data, interpolation_fs, target_fs)

    return timestamps, data


def resample_df(df, target_fs):
    timestamps = df.timestamp.values
    data = df.loc[:, ['x', 'y', 'z']].values

    timestamps, data = resample(timestamps, data, target_fs)

    values = np.hstack((timestamps.reshape((-1, 1)), data))

    return pd.DataFrame(values, columns=['timestamp', 'x', 'y', 'z'])


def high_pass_df(df, cutoff_fs):
    timestamps = df.timestamp.values
    data = df.loc[:, ['x', 'y', 'z']].values

    orig_fs = np.round(1.0 / (np.diff(timestamps).mean() / SECOND_TO_TIMESTAMP), 1)

    data = high_pass(timestamps, data, orig_fs, cutoff_fs)

    values = np.hstack((timestamps.reshape((-1, 1)), data))

    return pd.DataFrame(values, columns=['timestamp', 'x', 'y', 'z'])