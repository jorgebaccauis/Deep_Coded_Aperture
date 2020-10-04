import os
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import pandas
import os.path as path
import matplotlib.pyplot as plt
import datetime
from Models import *
from Custumer_Callbacks import *
from Function_project import *
from Read_Dataset import *
from Metrics import *

#
# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)


# os.environ["CUDA_VISIBLE_DEVICES"]= '-1'

# Paramters of the paper


flags.DEFINE_string('architecture','CASSI', 'name of the sensing architecture used [Single_Pixel,CASSI,CASSI_Color]')
flags.DEFINE_float('compression',0.1, 'compression ratio') #0.035, 0.1, 0.2, 0.25, 0.3
flags.DEFINE_integer('noise',40,'SNR(dB) for noise measurements')

flags.DEFINE_string('Database','Data_store','Database used [MNIST,Data_store,Data_storeDepth]')
flags.DEFINE_string('Data_path','/home/hdsp/Documentos/Bases_datos/ARAD_DATASET_s','path directory of dataset')
flags.DEFINE_integer('M',256,'Spatial dimensions')
flags.DEFINE_integer('L',16,'Spectral dimensions')

flags.DEFINE_string('Net','UNetL_conditional','Net structure for reconstruction, [Net, DeepCubeNet, UNetL-Net]')
flags.DEFINE_string('CA','Binary_0', 'Type of the Coded Aperture[Binary_0,Binary_1,Gray_scale]') #

flags.DEFINE_float('reg_HtH',0.8,'regularization parameter for HTH')

flags.DEFINE_string('type_reg','Physical', 'Type of the Coded Aperture')
flags.DEFINE_float('param_reg',1e-8, 'Regularization parameter for first constraint (binary)')
flags.DEFINE_float('param_reg2',5e-2, 'Regularization parameter for the second regularizer')

flags.DEFINE_float('aument_reg',1e1,'Aumented parameter for regularzer param_reg *= aument_Red')
flags.DEFINE_integer('step_reg',15,'Regulator Increase Step')


flags.DEFINE_float('tram',1.0, 'Transmittance value')

flags.DEFINE_integer('kernel',256,'Kernel size of the repeat CA')

# Tranining paparameters
flags.DEFINE_integer('epochs',3, 'number of epochs')
flags.DEFINE_integer('batch_size',  12, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate') #5e-4


flags.DEFINE_string('results','../Results','Path for the results')

flags.DEFINE_boolean('Load_weights',0,'Boolean 0 not 1 yes')
flags.DEFINE_string('Load_path','../Results/Weights/Net_last.tf','Path for the Load weight')

def main(_argv):
    try:
        os.stat(FLAGS.results)
    except:
        os.mkdir(FLAGS.results)
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

    #####################################################
    # -------------Training Step ------------------------
    ####################################################

    # ------- Load data set-------------------------------

    if (FLAGS.Database == 'MNIST'):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    if (FLAGS.Database == 'Data_store'):
        split = 1
        Path = FLAGS.Data_path + '/train'
        Aux, train_dataset = Build_data_set2(FLAGS.M, FLAGS.M, FLAGS.L, FLAGS.L, FLAGS.batch_size, Path, split, 'mat')
        Path = FLAGS.Data_path + '/val'
        Aux, test_dataset = Build_data_set2(FLAGS.M, FLAGS.M, FLAGS.L, FLAGS.L, 6, Path, split, 'mat')

    # --------- Load Model ---------------------------------

    if (FLAGS.Net == 'DeepCubeNet'):
        model = DeepCubeNet(input_size=(FLAGS.M, FLAGS.M, FLAGS.L), FLAGS=FLAGS)
    if (FLAGS.Net == 'Net'):
        model = Net(input_size=(FLAGS.M, FLAGS.M, FLAGS.L), FLAGS=FLAGS)

    if (FLAGS.Net == 'UNetL_conditional'):
        model = UNetL(input_size=(FLAGS.M, FLAGS.M, FLAGS.L), L=FLAGS.L, FLAGS=FLAGS)

    if (FLAGS.Net == 'UNetL_conditional'):
        model = UNetL_conditional(input_size=(FLAGS.M, FLAGS.M, FLAGS.L), L=FLAGS.L, FLAGS=FLAGS)

    last_epoch = 0
    if FLAGS.Load_weights:
        model.load_weights(FLAGS.Load_path)
        with open(FLAGS.results + '/Weights/last_epoch_info.txt', "r") as txt_log:
            lines = txt_log.readlines()
            last_epoch = int(lines[0].strip())
        print('Loaded pre-trained weights, epoch = ' + str(last_epoch))

    # ----------Calbacks---------------------------------

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'Unet_3'
    # my call backs
    if (FLAGS.Database == 'Data_store'):
        for inp, tar in test_dataset.take(-1):
            x_test = inp.numpy()

    # callbacks = [Aument_parameters(p_aum=FLAGS.aument_reg),SaveLastEpochWeights(FLAGS.results),
    #           Print_results(FLAGS.results,x_test,FLAGS.kernel,FLAGS.M),tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]

    # SaveLastEpochWeights_cond(FLAGS.results)
    callbacks = [SaveLastEpochWeights_cond(FLAGS.results),Aument_parameters(p_aum=FLAGS.aument_reg, p_step= FLAGS.step_reg)]

    # --------------- Optimization------------------------------------
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=FLAGS.learning_rate,
        decay_steps=10000,
        decay_rate=0.95)
    optimizad = keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizad = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, amsgrad=False)
    model.compile(optimizer=optimizad, loss='mean_squared_error', loss_weights=[1,FLAGS.reg_HtH],metrics=[psnr])

    if (FLAGS.Database == 'MNIST'):
        history = model.fit(x_train, x_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
                            validation_data=(x_test, x_test), initial_epoch=last_epoch, callbacks=callbacks)

    if (FLAGS.Database == 'Data_store'):
        history = model.fit(train_dataset, epochs=FLAGS.epochs, validation_data=test_dataset, initial_epoch=last_epoch,
                            callbacks=callbacks)

    #####################################################
    # -------------Testing Step ------------------------
    #####################################################

    # ------------- Results --------------------------------------------
    model.summary()
    temporal = model.get_weights()
    # obtain the CA
    CA = temporal[0]

    if (FLAGS.architecture == 'CASSI_Color'):
        wt = temporal[0] + temporal[1] + temporal[2] + temporal[3]
        wr = tf.math.divide(temporal[0], wt)
        wg = tf.math.divide(temporal[1], wt)
        wb = tf.math.divide(temporal[2], wt)
        wc = tf.math.divide(temporal[3], wt)
        fr, fg, fc, fb = get_color_bases(np.linspace(400, 700, FLAGS.L) * 1e-9)
        fr = tf.expand_dims(tf.expand_dims(fr, -1), 0)
        fg = tf.expand_dims(tf.expand_dims(fg, -1), 0)
        fc = tf.expand_dims(tf.expand_dims(fc, -1), 0)
        fb = tf.expand_dims(tf.expand_dims(fb, -1), 0)

        CA = tf.multiply(wr, fr) + tf.multiply(wg, fg) + tf.multiply(wb, fb) + tf.multiply(wc, fc)
        CA = kronecker_product_color(tf.ones((int(FLAGS.M / FLAGS.kernel), int(FLAGS.M / FLAGS.kernel))), CA).numpy()
    # CA = kronecker_product_np(np.ones((int(FLAGS.M / FLAGS.kernel), int(FLAGS.M / FLAGS.kernel))),CA)

    if (FLAGS.Database == 'MNIST'):
        # Make the measurements
        if (FLAGS.architecture == 'Single_Pixel'):

            # Task Network
            if (FLAGS.Net == 'Net'):
                modelTest = Net_test(input_size=(FLAGS.M, FLAGS.M, FLAGS.L), FLAGS=FLAGS)
                it = 0
                for layer in model.layers[2:]:
                    it = it + 1
                    modelTest.layers[it].set_weights(model.get_layer(layer.name).get_weights())
        # reconstrution
        psnr_v = [];
        mse_v = [];
        mae_v = [];
        ssim_v = []
        for i in range(0, x_test.shape[0], 100):
            x_est = measurements_single_pixel(x_test[i:i + 100, :, :, :], CA, FLAGS.noise, FLAGS.CA)
            result = modelTest.predict(x_est, batch_size=1)
            psnr_a, mse_a, mae_a, ssim_a = Metrics_batch(x_test[i:i + 100, :, :, :], result)
            psnr_v.append(psnr_a.numpy());
            mse_v.append(mse_a.numpy());
            mae_v.append(mae_a.numpy());
            ssim_v.append(ssim_a.numpy())

    if (FLAGS.Database == 'Data_store'):

        if (FLAGS.Net == 'UNetL_conditional'):
            modelTest = UNetL_test(input_size=(FLAGS.M, FLAGS.M, FLAGS.L), L=FLAGS.L, FLAGS=FLAGS)
            it = 0
            for layer in model.layers[2:]:
                it = it + 1
                modelTest.layers[it].set_weights(model.get_layer(layer.name).get_weights())

        if (FLAGS.Net == 'DeepCubeNet'):
            modelTest = DeepCubeNet_test(input_size=(FLAGS.M, FLAGS.M, FLAGS.L), FLAGS=FLAGS)
            it = 0
            for layer in model.layers[2:]:
                it = it + 1
                modelTest.layers[it].set_weights(model.get_layer(layer.name).get_weights())

        psnr_v = [];
        mse_v = [];
        mae_v = [];
        ssim_v = []
        for inp, tar in test_dataset.take(-1):
            x_test = inp.numpy()
            if (FLAGS.architecture == 'CASSI'):
                x_est = measurements_cassi(x_test, CA, FLAGS.noise, FLAGS.CA)
                result = modelTest.predict(x_est, batch_size=1)
            if (FLAGS.architecture == 'CASSI_Color'):
                result = model.predict(x_test, batch_size=1)

            psnr_a, mse_a, mae_a, ssim_a = Metrics_batch(x_test, result)
            psnr_v.append(psnr_a.numpy());
            mse_v.append(mse_a.numpy());
            mae_v.append(mae_a.numpy());
            ssim_v.append(ssim_a.numpy())

    # -----------------Visual Results---------------------
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.savefig(FLAGS.results + '/Loss')
    plt.show()

    if (len(CA.shape) < 5):
        CA = np.expand_dims(CA, axis=3)

    # --------- Coded Aperture---------------
    scipy.io.savemat(FLAGS.results + "/CA.mat", {'CA': CA})

    for iter in range(np.min((CA.shape[4], 4))):
        plt.subplot(1, np.min((CA.shape[4], 4)), iter + 1)
        plt.imshow(CA[0, :, :, 0, iter], 'gray'), plt.title('cod_' + str(iter + 1))

    plt.colorbar()
    plt.savefig(FLAGS.results + '/CA')
    plt.show()

    for iter in range(np.min((result.shape[0], 2))):
        plt.subplot(1, np.min((result.shape[0], 2)) * 2, 2 * iter + 1)
        plt.imshow(result[iter, :, :, 0], 'gray'), plt.title('reconstruction')
        plt.subplot(1, np.min((result.shape[0], 2)) * 2, 2 * iter + 2)
        plt.imshow(x_test[iter, :, :, 0], 'gray'), plt.title('original')

    plt.savefig(FLAGS.results + '/Reconstruction')
    plt.show()

    transmitance = np.sum(CA > 0.5) / (CA.shape[0] * CA.shape[1] * CA.shape[2] * CA.shape[3] * CA.shape[4])

    val_m = (np.max(CA) + np.min(CA)) / 2

    variance_1 = np.var(CA[CA > val_m])
    variance_2 = np.var(CA[CA < val_m])

    if path.exists(FLAGS.results + '/Results_summary.txt'):
        f = open(FLAGS.results + '/Results_summary.txt', "a+")
    else:
        f = open(FLAGS.results + '/Results_summary.txt', "w+")
    # f.write('Transmittance = ' + (str(transmitance)[0:6]) + "\n")
    f.write('Transmittance = ' + (str(transmitance)[0:6]))
    f.write(', Pho = ' + (str(FLAGS.param_reg)))
    f.write(', Var_1= ' + str(variance_1)[0:5] + str(variance_1)[len(str(variance_1)) - 4:] +
            ', Var_2= ' + str(variance_2)[0:5] + str(variance_2)[len(str(variance_2)) - 4:] +
            ', PSNR = ' + str(np.mean(psnr_v))[0:5] + ', MSE = ' + str(np.mean(mse_v))[0:6] +
            ', MAE = ' + str(np.mean(mae_v))[0:7] + ', SSIM = ' + str(np.mean(ssim_v))[0:5] + "\n")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass