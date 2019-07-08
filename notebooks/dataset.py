import os

import numpy as np
from sklearn.utils import shuffle

import tensorflow.compat.v1 as tf

GATE2 = 'gate2.sscc.uos.ac.kr'
DGX = 'dgx'
CMS05 = 'cms05.sscc.uos.ac.kr'

def load_dataset(tag, min_pt=1000, with_feature_names=False, onehot=True):
    assert tag in ['training', 'validation', 'test']
    assert min_pt in [100, 200, 500, 1000]

    host = os.environ['HOSTNAME']
    if host == GATE2:
        data_dir = '/scratch/seyang/SelfAttention/1-QGJets/'
    elif host == CMS05:
        data_dir = '/store/slowmoyang/SelfAttention/1-QGJets'
    else:
        raise NotImplementedError(host)

    max_pt = int(1.1 * min_pt)
    name_fmt = f'{{}}_pt_{min_pt}_{max_pt}_{tag}_set.npz'
    path_fmt = os.path.join(data_dir, name_fmt)
    paths = [path_fmt.format(each) for each in ['qq', 'gg']]

    npz_files = [np.load(each) for each in paths]
    
    feature_names = [
        'ptd', 'major_axis', 'minor_axis',
        'chad_mult', 'nhad_mult',
        'photon_mult', 'electron_mult', 'muon_mult'
    ]
    
    def format_example(npz):
        x = [npz[each] for each in feature_names]
        return np.stack(x, axis=-1).astype(np.float32)

    x = [format_example(npz) for npz in npz_files]
    x = np.concatenate(x)

    y = np.concatenate([npz['label'] for npz in npz_files])
    y = y.astype(np.int64)
    if onehot:
        y = tf.keras.utils.to_categorical(y)

    x, y = shuffle(x, y)
    if with_feature_names:
        return x, y, feature_names
    return x, y 
