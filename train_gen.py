from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K

from molecules.vectorizer import SmilesDataGenerator, CanonicalSmilesDataGenerator

NUM_EPOCHS = 1
EPOCH_SIZE = 500000
BATCH_SIZE = 500
LATENT_DIM = 292
MAX_LEN = 120
TEST_SPLIT = 0.20
RANDOM_SEED = 1337
NUM_CORES = -1

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='The HDF5 file containing structures.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--epoch_size', type=int, metavar='N', default=EPOCH_SIZE,
                        help='Number of samples to process per epoch during training.')
    parser.add_argument('--test_split', type=float, metavar='N', default=TEST_SPLIT,
                        help='Fraction of dataset to use as test data, rest is training data.')
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
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

    if args.num_cores != -1:
        print('num_cores = ' + str(args.num_cores))
        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
                                allow_soft_placement=True, device_count = {'CPU': args.num_cores})
        session = tf.Session(config=config)
        K.set_session(session)

    data = pd.read_hdf(args.data, 'table')
    structures = data['structure']

    # import gzip
    # filepath = args.data
    # structures = [line.split()[0].strip() for line in gzip.open(filepath) if line]

    # can also use CanonicalSmilesDataGenerator
    datobj = CanonicalSmilesDataGenerator(structures, MAX_LEN,
                                 test_split=args.test_split,
                                 random_seed=args.random_seed)
    test_divisor = int((1 - datobj.test_split) / (datobj.test_split))
    train_gen = datobj.train_generator(args.batch_size)
    test_gen = datobj.test_generator(args.batch_size)

    # reformulate generators to not use weights
    train_gen = ((tens, tens) for (tens, _, weights) in train_gen)
    test_gen = ((tens, tens) for (tens, _, weights) in test_gen)

    if args.simple:
        model = SimpleMoleculeVAE()
    else:
        model = MoleculeVAE()

    if os.path.isfile(args.model):
        model.load(datobj.chars, args.model, latent_rep_size = args.latent_dim)
    else:
        model.create(datobj.chars, latent_rep_size = args.latent_dim)

    checkpointer = ModelCheckpoint(filepath = args.model,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

    history = model.autoencoder.fit_generator(
        train_gen,
        args.epoch_size,
        nb_epoch = args.epochs,
        callbacks = [checkpointer, reduce_lr],
        validation_data = test_gen,
        nb_val_samples = args.epoch_size / test_divisor,
        pickle_safe = True
    )
    with open('history.p', 'wb') as f:
        cPickle.dump(history.history, f)

if __name__ == '__main__':
    main()
