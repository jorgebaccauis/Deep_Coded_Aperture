
#%% md

# Libraries



#%%


import tensorflow as tf
import os
from matplotlib import pyplot as plt
from os import listdir

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


from os.path import isfile, join
import numpy as np
import keras
import scipy.io
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from keras import backend as K
from os import listdir
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import MinMaxNorm,NonNeg
import scipy
from IPython.display import clear_output
from scipy import interpolate
#from google.colab import drive
#drive.mount('/content/drive', force_remount=False)
#%cd /content/drive/My Drive/DeepFusion
from Filter_pattern import *
import h5py
#%%

val_dir = '/home/hdsp/Documentos/Bases_datos/ICVL_dataset/val_1/'
data_dir = '/home/hdsp/Documentos/Bases_datos/ICVL_dataset/train/'
images_names = [ "ICVL_" +str(i+1) + ".mat" for i in range(0,1530)]
val_names =  [ "ICVL_50"]


dataset_size = (len(images_names),512,512,31)
val_size = (len(val_names),512,512,31)

Y = np.zeros(dataset_size)
Y_val = np.zeros(val_size)
i = 0
for name in images_names:
    Hy = scipy.io.loadmat(data_dir+name)['Hyperimg']
    Y[i] = Hy
    print(i)
    clear_output(wait=True)
    i = i +1

print('training')
i = 0
for name in val_names:
    Hy = scipy.io.loadmat(val_dir+name)['Hyperimg']
    Y_val[i] = Hy
    i = i +1
print('val')






#%%

M = 512
L = 31
batch_size = 1
epochs_v = 150
p_aum = 1e1
p_step = 15
RGB = [22, 14 , 5]
ca_shape = (1 , 512 , 512 + 31 - 1, 1)

coded_aperture = np.random.random(ca_shape)

coded_aperture = np.asanyarray( (coded_aperture<0.5)*1, dtype=np.float32)
ca_total= np.prod(ca_shape)

T = np.linalg.norm( coded_aperture.flatten(),1)  / ca_total

print('Transmitancia: ', T)

#%%

from Hx import *


#%% md

# Unrolling Functions

#%%

class Mu_parameter(keras.layers.Layer):
    def __init__(self, units=1, input_dim=32):
        super(Mu_parameter, self).__init__()
        w_init = tf.keras.initializers.Constant(value=0)
        self.w = tf.Variable(
            initial_value=w_init(shape=(units,1), dtype="float32"),
            trainable=True, constraint = NonNeg()
        )

    def call(self, inputs):
        return tf.multiply(self.w, inputs)

class Lambda_parameter(keras.layers.Layer):
    def __init__(self, units=1, input_dim=32):
        super(Lambda_parameter, self).__init__()
        w_init = tf.keras.initializers.Constant(value=0)
        self.w = tf.Variable(
            initial_value=w_init(shape=(units,1), dtype="float32"),
            trainable=True, constraint = NonNeg()
        )

    def call(self, inputs):
        return tf.multiply(self.w, inputs)

def GradientCASSI(x):
  X = x[0]
  y = x[1]
  H = x[2]
  y_e = Forward_D_CASSI(X,H)
  rh = y_e-y
  Gh = Transpose_D_CASSI(rh,H)
  return Gh


#%% md

# Unrolling Network


#%%

from Custumer_Layers import  DD_CASSI_Layer

def HIDDSP(pretrained_weights=None, input_size=(512, 512, 31), depth=64, bands=31):

  inputs = Input(shape=input_size,name='image')
  X0, y,H = DD_CASSI_Layer(output_dim=input_size, input_dim=input_size,parm1=1e-9)(inputs)
  #y = Forward_D_CASSI(inputs,H)
  #X0 = Transpose_D_CASSI(y,H)
  #--------Stage 1---------------------
  # - h step--
  conv_r1 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X0)
  conv_r1 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r1)
  conv_r1 = Add()([X0,conv_r1])
  conv_r1 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r1)
  # - x step --
  # H^T(Hf-y)
  X1 = Lambda(GradientCASSI)([X0,y,H])
  X1_prior = Subtract()([X0,conv_r1])
  X1_prior = Mu_parameter()(X1_prior)

  X1 = Add()([X1_prior,X1])
  X1 = Lambda_parameter()(X1)
  X1 = Subtract(name='X1')([X0,X1])

    #--------Stage 2---------------------
  # - h step--
  conv_r2 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X1)
  conv_r2 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r2)
  conv_r2 = Add()([X1,conv_r2])
  conv_r2 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r2)
  # - x step --
  # H^T(Hf-y)
  X2 = Lambda(GradientCASSI)([X1,y,H])
  X2_prior = Subtract()([X1,conv_r2])
  X2_prior = Mu_parameter()(X2_prior)

  X2 = Add()([X2_prior,X2])
  X2 = Lambda_parameter()(X2)
  X2 = Subtract(name='X2')([X1,X2])

    #--------Stage 3---------------------
  # - h step--
  conv_r3 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X2)
  conv_r3 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r3)
  conv_r3 = Add()([X2,conv_r3])
  conv_r3 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r3)
  # - x step --
  # H^T(Hf-y)
  X3 = Lambda(GradientCASSI)([X2,y,H])
  X3_prior = Subtract()([X2,conv_r3])
  X3_prior = Mu_parameter()(X3_prior)

  X3 = Add()([X3_prior,X3])
  X3 = Lambda_parameter()(X3)
  X3 = Subtract(name='X3')([X2,X3])

      #--------Stage 4---------------------
  # - h step--
  conv_r4 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X3)
  conv_r4 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r4)
  conv_r4 = Add()([X3,conv_r4])
  conv_r4 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r4)
  # - x step --
  # H^T(Hf-y)
  X4 = Lambda(GradientCASSI)([X3,y,H])
  X4_prior = Subtract()([X3,conv_r4])
  X4_prior = Mu_parameter()(X4_prior)

  X4 = Add()([X4_prior,X4])
  X4 = Lambda_parameter()(X4)
  X4 = Subtract(name='X4')([X3,X4])

        #--------Stage 5---------------------
  # - h step--
  conv_r5 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X4)
  conv_r5 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r5)
  conv_r5 = Add()([X4,conv_r5])
  conv_r5 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r5)
  # - x step --
  # H^T(Hf-y)
  X5 = Lambda(GradientCASSI)([X4,y,H])
  X5_prior = Subtract()([X4,conv_r5])
  X5_prior = Mu_parameter()(X5_prior)

  X5 = Add()([X5_prior,X5])
  X5 = Lambda_parameter()(X5)
  X5 = Subtract(name='X5')([X4,X5])

          #--------Stage 6---------------------
  # - h step--
  conv_r6 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X5)
  conv_r6 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r6)
  conv_r6 = Add()([X5,conv_r6])
  conv_r6 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r6)
  # - x step --
  # H^T(Hf-y)
  X6 = Lambda(GradientCASSI)([X5,y,H])
  X6_prior = Subtract()([X5,conv_r6])
  X6_prior = Mu_parameter()(X6_prior)

  X6 = Add()([X6_prior,X6])
  X6 = Lambda_parameter()(X6)
  X6 = Subtract(name='X6')([X5,X6])

            #--------Stage 7---------------------
  # - h step--
  conv_r7 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X6)
  conv_r7 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r7)
  conv_r7 = Add()([X6,conv_r7])
  conv_r7 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r7)
  # - x step --
  # H^T(Hf-y)
  X7 = Lambda(GradientCASSI)([X6,y,H])
  X7_prior = Subtract()([X6,conv_r7])
  X7_prior = Mu_parameter()(X7_prior)

  X7 = Add()([X7_prior,X7])
  X7 = Lambda_parameter()(X7)
  X7 = Subtract(name='X7')([X6,X7])

              #--------Stage 8---------------------
  # - h step--
  conv_r8 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X7)
  conv_r8 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r8)
  conv_r8 = Add()([X7,conv_r8])
  conv_r8 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r8)
  # - x step --
  # H^T(Hf-y)
  X8 = Lambda(GradientCASSI)([X7,y,H])
  X8_prior = Subtract()([X7,conv_r8])
  X8_prior = Mu_parameter()(X8_prior)

  X8 = Add()([X8_prior,X8])
  X8 = Lambda_parameter()(X8)
  X8 = Subtract(name='X8')([X7,X8])

              #--------Stage 9---------------------
  # - h step--
  conv_r9 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(X8)
  conv_r9 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r9)
  conv_r9 = Add()([X8,conv_r9])
  conv_r9 = Conv2D(bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r9)
  # - x step --
  # H^T(Hf-y)
  X9 = Lambda(GradientCASSI)([X8,y,H])
  X9_prior = Subtract()([X8,conv_r9])
  X9_prior = Mu_parameter()(X9_prior)

  X9 = Add()([X9_prior,X9])
  X9 = Lambda_parameter()(X9)
  X9 = Subtract(name='X9')([X8,X9])

  model = Model(inputs,X9)


  if (pretrained_weights):
      model.load_weights(pretrained_weights)
      print('loading weights generator')

  return model

#%% md

# Custom Callbacks, Losses and Metrics

#%%

def PSNR_Metric(y_true, y_pred):
  return tf.reduce_mean(tf.image.psnr(y_true,y_pred,1))

def SSIM_Metric(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_pred,y_true,1))


class save_each_epoch(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir


    def on_epoch_end(self, epoch, logs=None):
        print('Model Saved at: ' + self.checkpoint_dir)

        self.model.save_weights(self.checkpoint_dir)


from keras.callbacks import LearningRateScheduler

# This is a sample of a scheduler I used in the past
def lr_scheduler(epoch, lr):
    decay_step = 40
    if epoch % decay_step == 0 and epoch:
        lr = lr/2
        tf.print(' Learning rate ='+ str(lr))
        return lr

    return lr

#%% md

# Train network

#%%

model = HIDDSP(pretrained_weights=None,input_size=(512,512,31), depth=31, bands=31 )

optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics = [PSNR_Metric])

from keras.callbacks import ModelCheckpoint

check_path = './weights/HIDDSP_1_batch.h5'
checkpoint = ModelCheckpoint(check_path, monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=True ,mode='auto', save_freq='epoch')

#%%

from Custumer_Callbacks import Aument_parameters
history = model.fit( x=Y,y=Y , epochs=epochs_v, batch_size=batch_size, validation_data=(Y_val,Y_val), callbacks=[checkpoint,Aument_parameters(p_aum=p_aum, p_step=p_step)])

#%%

temporal = model.get_weights()
    # obtain the CA
CA = temporal[0]
scipy.io.savemat("Results/CA1batch.mat", {'CA': CA})
result = model.predict(Y_val, batch_size=1)


scipy.io.savemat("Results/result_1batch.mat", {'result': result})