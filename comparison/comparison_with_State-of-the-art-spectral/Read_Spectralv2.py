import tensorflow as tf
import os
import numpy as np
import keras
import pandas as pd
import h5py
from scipy.io import loadmat


def Input_image(image):
    with h5py.File(image, 'r') as f:
        images = f.get('Data_24b')[:]
    images = np.array(images).transpose(2, 1, 0)

    return images


def image_mat(image):
    im = loadmat(image).get('Hyperimg')
    return im


class DataGenerator(keras.utils.Sequence):
    def __init__(self, samples, PATH, batch_size=1,
                 dim_input=(512, 512, 31),
                 shuffle=True, dim_oput=(512, 512, 31),memory_req = 1):
        'Initialization'
        self.dim_input = dim_input
        self.dim_oput = dim_oput
        self.batch_size = batch_size
        self.list_images = samples
        self.shuffle = shuffle
        self.PATH = PATH
        self.len_images = len(self.list_images)
        self.true_epoch = 0
        # Tamaño de cada super batch

        self.sup_bath_size = int(self.len_images/(int(self.len_images/int(memory_req / (dim_input [0] * dim_input [1] * dim_input [2] * 5.5e-09)))+1))
        self.num_sup_batch = int(self.len_images/self.sup_bath_size) # Numeros de super batch
        self.cont_sup_bath = 0
        self.cont_repeat = 0
        self.num_repeat = 5  # Cantidad de veces que se repetira cada super batch
        self.indexes = np.arange(self.len_images)

    # falta __data_generation

    def __len__(self):
        'Denotes the number of batches per super batch'
        return int(self.sup_bath_size / self.batch_size)

    def __getitem__(self, index):

        self.sub_indexs = np.arange(
            index * self.batch_size, (index + 1) * self.batch_size, 1)
        if index == 0:
            if self.cont_repeat == 0:
                self.on_epoch_end()
                self.cont_repeat += 1
            elif self.cont_repeat != self.num_repeat - 1:
                self.cont_repeat += 1
            else:
                self.cont_repeat = 0
            np.random.shuffle(self.sup_bath)
            tf.print('Global Epoch ' + str(self.true_epoch * 5) + ' Batch ' + str(self.cont_sup_bath) + '/' + str(
                self.num_sup_batch) + ' rep ' + str(self.cont_repeat) + '/' + str(self.num_repeat))
        X = self.sup_bath[self.sub_indexs]
        return X, X

    def on_epoch_end(self):

        if self.cont_sup_bath != self.num_sup_batch:
            self.sup_bath_indxs = self.indexes[self.cont_sup_bath *
                                               self.sup_bath_size:(
                                                   self.cont_sup_bath + 1)
                                               * self.sup_bath_size]
            self.images_name = [self.list_images[k]
                                for k in self.sup_bath_indxs]

            self.cont_sup_bath += 1
            self.sup_bath = self.__data_generation(self.images_name)
        else:
            if self.shuffle:
                np.random.shuffle(self.indexes)
                self.true_epoch+=1

            self.cont_sup_bath = 0
            self.sup_bath_indxs = self.indexes[self.cont_sup_bath *
                                               self.sup_bath_size:(
                                                   self.cont_sup_bath + 1)
                                               * self.sup_bath_size]

            self.images_name = [self.list_images[k]
                                for k in self.sup_bath_indxs]

            self.cont_sup_bath += 1
            self.sup_bath = self.__data_generation(self.images_name)

    def __data_generation(self, images_names):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization

        # Array de numpy con zeros de tamaño
        X = np.empty((self.sup_bath_size, *self.dim_input))
        # Generate data
        for i, file_name in enumerate(images_names):
            # Store sample
            X[i, ] = image_mat(self.PATH + '/' + file_name)
        return X


def Build_data_set(IMG_WIDTH, IMG_HEIGHT,
                   L_bands, L_imput, BATCH_SIZE, PATH,memory_req=1):

    data_dir_list = os.listdir(PATH)

    params_df = {'dim_input': (IMG_WIDTH, IMG_HEIGHT, L_imput),
                 'dim_oput': (IMG_WIDTH, IMG_HEIGHT, L_bands),
                 'batch_size': BATCH_SIZE,
                 'PATH': PATH,
                 'shuffle': True,
                 'memory_req':memory_req}

    df_generator = DataGenerator(data_dir_list, **params_df)

    df_dataset = tf.data.Dataset.from_generator(
        lambda: df_generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([None, IMG_WIDTH, IMG_HEIGHT, L_imput]),
         tf.TensorShape([None, IMG_WIDTH, IMG_HEIGHT, L_bands]))
    )

    return df_dataset
