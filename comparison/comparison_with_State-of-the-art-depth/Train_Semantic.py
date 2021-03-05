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
from Read_Dataset2 import *
from Metrics import *

#
# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

#os.environ["CUDA_VISIBLE_DEVICES"]= '-1'

# Paramters of the paper


flags.DEFINE_string('architecture', 'Depth_CA', 'name of the sensing architecture used [Single_Pixel,CASSI,CASSI_Color]')
flags.DEFINE_float('compression', 0.035, 'compression ratio')  # 0.035
flags.DEFINE_integer('noise', 60, 'SNR(dB) for noise measurements')

flags.DEFINE_string('Database', 'Data_storeDepth', 'Database used [MNIST,Data_store,Data_storeDepth]')
flags.DEFINE_string('Data_path', '/home/hdsp/Documentos/Bases_datos/nyu_dataset', 'path directory of dataset')
flags.DEFINE_integer('M', 512, 'Spatial dimensions')
flags.DEFINE_integer('L', 3, 'Spectral dimensions')

flags.DEFINE_string('Net', 'UNetL_class__p', 'Net structure for reconstruction, [Net, DeepCubeNet, UNetL-Net]')
flags.DEFINE_float('ker_l2', 0,'L_2 regularizer for kernel')

flags.DEFINE_string('CA', 'Binary_0', 'Type of the Coded Aperture[Binary_0,Binary_1,Real_value]')  #

flags.DEFINE_string('type_reg', 'Real_value', 'Type of the Coded Aperture')

flags.DEFINE_float('param_reg',10, 'Regularization parameter for first constraint (binary)')

flags.DEFINE_float('aument_reg', 1e1, 'Aumented parameter for regularzer param_reg *= aument_Red')
flags.DEFINE_integer('step_reg', 20, 'Regulator Increase Step')

flags.DEFINE_float('param_reg2', 5e-2, 'Regularization parameter for the second regularizer')
flags.DEFINE_integer('kernel', 256, 'Kernel size of the repeat CA')

flags.DEFINE_float('tram', 1.0, 'Transmittance value')

# Tranining paparameters
flags.DEFINE_integer('epochs', 0, 'number of epochs')
flags.DEFINE_integer('batch_size', 3, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')  # 5e-4

flags.DEFINE_string('results','../Results','Path for the results')

flags.DEFINE_boolean('Load_weights',1,'Boolean 0 not 1 yes')
flags.DEFINE_string('Load_path','../Results/Weights/Net_last.tf','Path for the Load weight')

def main(_argv):
    class_n =41

    try:
        os.stat(FLAGS.results)
    except:
        os.mkdir(FLAGS.results)

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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

    if (FLAGS.Database == 'Data_storeDepth'):
        split = 1
        Path = FLAGS.Data_path + '/Train'
        Aux, train_dataset = Build_data_set2(FLAGS.M, FLAGS.M, FLAGS.L, 15, class_n, FLAGS.batch_size, Path, split,
                                             'jpg')
        Path = FLAGS.Data_path + '/Test'
        Aux, test_dataset = Build_data_set2(FLAGS.M, FLAGS.M, FLAGS.L, 15, class_n, FLAGS.batch_size, Path, split,
                                            'jpg')

    def REDFNet(pretrained_weights=None, input_size1=(512, 512, 3), input_size2=(512, 512, 1), class_n=41,batch=2,FLAGS=None):

        inputs = Input(input_size1)
        inputs2 = Input(input_size2)

        rgbs, depths = Depth_CA_v2(output_dim=input_size2, input_dim=input_size1,
                          type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                          parm2=FLAGS.param_reg2, trans=FLAGS.tram, batch=batch)([inputs, inputs2])
        conv2 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(depths)

        RGB_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
        x5 = RGB_model.get_layer('conv5_block3_out').output
        x4 = RGB_model.get_layer('conv4_block6_out').output
        x3 = RGB_model.get_layer('conv3_block4_out').output
        x2 = RGB_model.get_layer('conv2_block3_out').output
        rgb_model = tf.keras.models.Model(inputs=RGB_model.input, outputs=[x2, x3, x4, x5], name='rgb_model')
        for layer in rgb_model.layers [:175]:
            layer.trainable = False

        Depth_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
        x5 = Depth_model.get_layer('conv5_block3_out').output
        x4 = Depth_model.get_layer('conv4_block6_out').output
        x3 = Depth_model.get_layer('conv3_block4_out').output
        x2 = Depth_model.get_layer('conv2_block3_out').output
        depth_model = tf.keras.models.Model(inputs=Depth_model.input, outputs=[x2, x3, x4, x5], name='depth_model')
        for layer in depth_model.layers [:175]:
            layer.trainable = False

        rest2_rgb,rest3_rgb,rest4_rgb,rest5_rgb, = rgb_model(rgbs)
        rest2_d,rest3_d,rest4_d,rest5_d, = depth_model(conv2)

        dept_1 = MMF(rest2_rgb,rest2_d)
        dept_2 = MMF(rest3_rgb, rest3_d)
        dept_3 = MMF(rest4_rgb, rest4_d)
        dept_4 = MMF(rest5_rgb, rest5_d,feature=512)

        Ref_4 = Refine4(dept_4,feature=512)

        Ref_3 = Refine(dept_3,Ref_4,feature1=256,feature2=512)
        Ref_2 = Refine(dept_2, Ref_3)
        Ref_1 = Refine(dept_1, Ref_2)
        Ref_1 = UpSampling2D(size=(4,4))(Ref_1)
        final = RCU(Ref_1)
        final = RCU(final)
        final = Dropout(0.5)(final)
        final = Conv2D(class_n, 1, activation='softmax', padding='same')(final)

        model = Model(inputs=[inputs,inputs2], outputs=final)

        if (pretrained_weights):
            model.load_weights(pretrained_weights)
        return model

    model = REDFNet(input_size1=(FLAGS.M, FLAGS.M, FLAGS.L), input_size2=(FLAGS.M, FLAGS.M, 15),batch =FLAGS.batch_size, FLAGS=FLAGS)
    input_size1 = (512, 512, 3)
    input_size2 = (512, 512, 1)


    # --------- Load Model ---------------------------------

    if (FLAGS.Net == 'Net'):
        model = Net(input_size=(FLAGS.M, FLAGS.M, FLAGS.L), FLAGS=FLAGS)

    if (FLAGS.Net == 'UNetL'):
        model = UNetL(input_size=(FLAGS.M, FLAGS.M, FLAGS.L), L=FLAGS.L, FLAGS=FLAGS)
    if (FLAGS.Net == 'UNetL_class'):
        model = UNetL_class(input_size1=(FLAGS.M, FLAGS.M, FLAGS.L), input_size2=(FLAGS.M, FLAGS.M, 15),
                            class_n=class_n, L=32, FLAGS=FLAGS, batch =FLAGS.batch_size )

    if (FLAGS.Net == 'UNetL_class_bn'):
        model = UNetL_class_bn(input_size1=(FLAGS.M, FLAGS.M, FLAGS.L), input_size2=(FLAGS.M, FLAGS.M, 15),
                               class_n=class_n, L=32, FLAGS=FLAGS, batch=FLAGS.batch_size)

    if (FLAGS.Net == 'UNetL_class_bnd'):
        model = UNetL_class_bnd(input_size1=(FLAGS.M, FLAGS.M, FLAGS.L), input_size2=(FLAGS.M, FLAGS.M, 15),
                               class_n=class_n, L=32, FLAGS=FLAGS, batch=FLAGS.batch_size)

    model.summary()

    last_epoch = 0
    if FLAGS.Load_weights:
        model.load_weights(FLAGS.Load_path)
        with open(FLAGS.results + '/Weights/last_epoch_info.txt', "r") as txt_log:
            lines = txt_log.readlines()
            last_epoch = int(lines[0].strip())
        print('Loaded pre-trained weights, epoch = '+str(last_epoch))

    # ----------Calbacks---------------------------------

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'Unet_3'
    # my call backs

    # callbacks = [Aument_parameters(p_aum=FLAGS.aument_reg),SaveLastEpochWeights(FLAGS.results),
    #           Print_results(FLAGS.results,x_test,FLAGS.kernel,FLAGS.M),tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]

    callbacks = [Depth_Aument_parameters(p_aum=FLAGS.aument_reg, p_step=FLAGS.step_reg), SaveLastEpochWeights2(FLAGS.results),
                 tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]

    # --------------- Optimization------------------------------------
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=FLAGS.learning_rate,
        decay_steps=10000,
        decay_rate=0.95)
    optimizad = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizad, loss='categorical_crossentropy',metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=FLAGS.epochs, validation_data=test_dataset, initial_epoch=last_epoch,
                        callbacks=callbacks)

    #####################################################
    # -------------Testing Step ------------------------
    #####################################################

    # ------------- Results --------------------------------------------
    temporal = model.get_weights()
    # obtain the CA
    CA = temporal[0]
    # CA = kronecker_product_np(np.ones((int(FLAGS.M / FLAGS.kernel), int(FLAGS.M / FLAGS.kernel))),CA)


    if (FLAGS.Database == 'Data_storeDepth'):
        if (FLAGS.Net == 'UNetL_class'):
            modelTest = UNetL_class_test(input_size1=(FLAGS.M, FLAGS.M, 45),
                                class_n=class_n, L=32, FLAGS=FLAGS)
            it = 0
            for layer in model.layers[3:]:
                it = it + 1
                modelTest.layers[it].set_weights(model.get_layer(layer.name).get_weights())

        psnr_v = [];
        mse_v = [];
        mae_v = [];
        ssim_v = []
        cont = 0
        for inp, tar in test_dataset.take(-1):
            cont = cont + 1
            x_test = inp
            x_est, y_measure, comCA, pp = measurments_Depth_CA(x_test, tf.convert_to_tensor(CA, dtype=tf.float32),
                                                               FLAGS.noise, x_test[0].shape[0], parameter=temporal[1])
            result = model.predict(x_test, batch_size=3)
            result = tf.math.argmax(result, axis=3)
            gt = tf.math.argmax(tar, axis=3)
            print(cont)
            try:
                os.stat(FLAGS.results+ '/Files')
            except:
                os.mkdir(FLAGS.results+ '/Files')
            scipy.io.savemat(FLAGS.results + "/Files/Num" + str(cont) + ".mat",
                             {'gt': np.squeeze(gt.numpy()), 'result': np.squeeze(result.numpy()),
                              'y_measure': np.squeeze(y_measure.numpy()),
                              'Image': np.squeeze(x_test[0].numpy()), 'Map': np.squeeze(x_test[1].numpy()),
                              'pp': np.squeeze(pp.numpy())})
        scipy.io.savemat(FLAGS.results + "/Files/CA1.mat", {'CA': np.squeeze(comCA.numpy())})


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
    tar = tf.math.argmax(tar,axis=3)
    result = tf.math.argmax(result, axis=3)
    for iter in range(np.min((result.shape[0], 2))):
        plt.subplot(1, np.min((result.shape[0], 2)) * 2, 2 * iter + 1)
        plt.imshow(result[iter, :, :], 'gray'), plt.title('reconstruction')
        plt.subplot(1, np.min((result.shape[0], 2)) * 2, 2 * iter + 2)
        plt.imshow(tar[iter, :, :], 'gray'), plt.title('original')


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
