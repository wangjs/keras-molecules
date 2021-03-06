from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
from keras.utils import plot_model
import cPickle
import tensorflow as tf
from keras import backend as K


NUM_EPOCHS = 1
BATCH_SIZE = 600
LATENT_DIM = 292
RANDOM_SEED = 1337
NUM_CORES = -1

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=RANDOM_SEED,
                        help='Seed to use to start randomizer for shuffling.')
    parser.add_argument('--simple', dest='simple', action='store_true', help='Use simple model.')
    parser.add_argument('--num_cores', type=int, metavar='N', default=NUM_CORES,
                        help='Number of CPU cores for TensorFlow. If -1 then uses all.')
    return parser.parse_args()

def main():
    args = get_arguments()
    np.random.seed(args.random_seed)

    from molecules.model import MoleculeVAE, SimpleMoleculeVAE
    from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
        decode_smiles_from_indexes, load_dataset
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

    if args.num_cores != -1:
        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
                                allow_soft_placement=True, device_count = {'CPU': args.num_cores})
        session = tf.Session(config=config)
        K.set_session(session)

    data_train, data_test, charset = load_dataset(args.data)

    if args.simple:
        model = SimpleMoleculeVAE()
    else:
        model = MoleculeVAE()

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = args.latent_dim)
    else:
        model.create(charset, latent_rep_size = args.latent_dim)

    checkpointer = ModelCheckpoint(filepath = args.model,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)


    # plot_model(model, to_file='model.png')
    history = model.autoencoder.fit(
        data_train,
        data_train,
        shuffle = True,
        epochs = args.epochs,
        batch_size = args.batch_size,
        callbacks = [checkpointer, reduce_lr],
        validation_data = (data_test, data_test)
    )
    with open('history.p', 'wb') as f:
        cPickle.dump(history.history, f)

if __name__ == '__main__':
    main()
