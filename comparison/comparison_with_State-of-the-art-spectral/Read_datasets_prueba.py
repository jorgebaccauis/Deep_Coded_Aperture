
#--------------------------------------------
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from IPython import display
from IPython.display import clear_output
from PIL import Image

from os import listdir
from os.path import isfile, join

import numpy as np
import scipy.io
import pandas as pd
from scipy.io import loadmat


from tensorflow.keras.preprocessing.image import ImageDataGenerator


def image_mat(image):
    im = loadmat(image).get('Hyperimg')
    return im


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, samples,PATH,batch_size=1,dim_input=(512, 512, 3), shuffle=True, dim_oput=(512, 512, 3),num_super_batch=20):
        'Initialization'
        self.dim_input = dim_input
        self.dim_oput = dim_oput
        self.batch_size = batch_size
        self.list_images = samples
        self.shuffle = shuffle
        self.PATH = PATH
        self.num_super_batch = num_super_batch
        self.super_batch = np.empty((self.num_super_batch, *self.dim_input))
        self.contador_mega_bach = 0
        self.contador_mega_bach_global = 0
        self.epoch_per_super_batch = 5
        self.p = 0
        self.on_epoch_end()


    # falta __data_generation
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.num_super_batch / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        #images_name = [self.list_images[k] for k in indexes]

        # Generate data
        X =  self.super_batch[index * self.batch_size:(index + 1) * self.batch_size]

        return X, X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.list_images))

        if self.contador_mega_bach_global < self.epoch_per_super_batch*(int(len(self.list_images) / self.num_super_batch)):
            self.contador_mega_bach_global += 1
        else:
            self.contador_mega_bach_global=0
            if self.shuffle == True:
                np.random.shuffle(self.indexes)
                self.p = 0

        if self.contador_mega_bach < self.epoch_per_super_batch:
            self.contador_mega_bach_global += 1
        else:
            self.contador_mega_bach_global=0
            index = self.indexes [self.p*self.num_super_batch:(self.p+1)*self.num_super_batch]
            self.p +=1
            images_name = [self.list_images [k] for k in index]
            self.super_batch = self.__data_generation(images_name)



    def __data_generation(self, images_names):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, *self.dim_input))  # Array de numpy con zeros de tamaño
        #y = np.empty((self.batch_size, *self.dim_oput))
        # Generate data
        for i, file_name in enumerate(images_names):

            #temp = image_mat(self.PATH + '/' + file_name)
            X[i,] = image_mat(self.PATH + '/' + file_name)
            #y[i,] = temp

        return X



def Build_data_set(IMG_WIDTH,IMG_HEIGHT,L_bands,L_imput,BATCH_SIZE,PATH):

  # Random split
  data_dir_list = os.listdir(PATH)
  params = {'dim_input': (IMG_WIDTH, IMG_HEIGHT, L_imput),
            'dim_oput': (IMG_WIDTH, IMG_HEIGHT, L_bands),
            'batch_size': BATCH_SIZE,
            'PATH': PATH,
            'shuffle': True}

  train_generator = DataGenerator(data_dir_list, **params)


  train_dataset = tf.data.Dataset.from_generator(
      lambda: train_generator,
      (tf.float32, tf.float32))

  return train_dataset













