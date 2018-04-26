import pandas as pd
from numpy.lib.stride_tricks import as_strided
import numpy as np


def norm_shape(shape):
    """
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    """
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        pass  # shape was not a number
    try:
        t = tuple(shape)
        return t
    except TypeError:
        pass  # shape was not iterable
    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a, ws, ss=None, flatten=True):
    """
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a

    Code is modified from:
    http://www.johnvinyard.com/blog/?p=268
    """

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    assert np.all(np.zeros(a.ndim) < np.asarray(ss)) and np.all(np.asarray(ss) <= np.asarray(ws)), \
        "Step size must be greater than zero and less than or equal to window size."
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError('a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError('ws cannot be larger than a in any dimension.a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(np.ceil((shape - (ws - ss)) / ss.astype(np.float)))
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # pad with step size (ss) zeros on each dimension in order to not get garbage from the last indices
    na = np.zeros(np.array(shape) + ss)
    na[tuple(map(lambda s: slice(0, -s), ss))] = a
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(na.strides) * ss) + na.strides

    strided = as_strided(na, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i: i != 1, dim)
    return strided.reshape(dim)


def split_user_df(user_df): #discriminate recording sessions by assuming that when time difference between consec. values in wdw > 1 sec, it was a different recording. Returns subframes from user_df. 
    
    #Calculate difference and fill any missing values with 0.
    diffs = user_df.timestamp.diff().fillna(0) 
    
    #generate list of positions where the difference is larger than 1 second = 1e9 nanoseconds.
    gaps = np.where(diffs > 1e9)[0].tolist()

    #grab subframes from user 
    gaps = [0] + gaps + [len(diffs)]
    splits = []
    for i in xrange(1, len(gaps)):
        splits.append(user_df.iloc[gaps[i - 1]: gaps[i]])

    return splits


def _print_confusion_matrix(cm, labels):
    """
    Pretty-prints a confusion matrix.

    Parameters
    ----------
    classes : list
        List class labels corresponding to the confusion matrix rows and columns.
    """
    cm = pd.DataFrame(cm)
    cm.columns = labels
    cm.index = labels

    print cm
