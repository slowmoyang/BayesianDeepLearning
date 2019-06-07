import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf

def load_dataset(paths):
    npz_files = [np.load(each) for each in paths]
    
    feature_names = [
        'ptd', 'major_axis', 'minor_axis',
        'chad_mult', 'nhad_mult',
        'photon_mult', 'electron_mult', 'muon_mult'
    ]
    
    def load(npz):
        x = [npz[each] for each in feature_names]
        return np.stack(x, axis=-1).astype(np.float32)

    x = [load(npz) for npz in npz_files]
    x = np.concatenate(x)

    y = np.concatenate([npz['label'] for npz in npz_files])
    # y = tf.keras.utils.to_categorical(y)

    x, y = shuffle(x, y)
    return x, y

def build_input_pipeline(train_set,
                         valid_set,
                         test_set,
                         batch_size=128):
    """Build an Iterator switching between train, valid and test data."""
    # Build an iterator over training batches.
    train_set = tf.data.Dataset.from_tensor_slices(train_set)
    train_set = train_set.shuffle(
        buffer_size=50000,
        reshuffle_each_iteration=True)
    # train_set = train_set.repeat()
    train_set = train_set.batch(batch_size)
    # tarin_iter = tf.compat.v1.data.make_one_shot_iterator(train_set)
    tarin_iter = train_set.make_initializable_iterator()

    def get_one_shot_iter(dset):
        size = len(dset[0])
        dset = tf.data.Dataset.from_tensor_slices(dset)
        dset = dset.take(size)
        dset = dset.repeat()
        dset = dset.batch(size)
        return tf.compat.v1.data.make_one_shot_iterator(dset)

    valid_iter = get_one_shot_iter(valid_set)
    test_iter = get_one_shot_iter(test_set)

    # Combine these into a feedable iterator that can switch between training
    # and validation inputs.
    handle = tf.compat.v1.placeholder(tf.string, shape=[])

    feedable_iter = tf.compat.v1.data.Iterator.from_string_handle(
        string_handle=handle,
        output_types=train_set.output_types,
        output_shapes=train_set.output_shapes)

    batch = feedable_iter.get_next()

    return batch, handle, (tarin_iter, valid_iter, test_iter)
