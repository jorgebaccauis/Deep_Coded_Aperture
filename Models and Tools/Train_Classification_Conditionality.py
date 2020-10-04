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
from keras.utils import to_categorical

#os.environ["CUDA_VISIBLE_DEVICES"]= '-1'



flags.DEFINE_string('architecture','Single_Pixel', 'name of the sensing architecture used [Single_Pixel,CASSI,CASSI_Color]')
flags.DEFINE_float('compression',0.4, 'compression ratio') #0.035
flags.DEFINE_integer('noise',50,'SNR(dB) for noise measurements')

flags.DEFINE_string('Database','MNIST','Database used [MNIST,Data_store]')
flags.DEFINE_string('Data_path','/home/hdsp/Documentos/Bases_datos/ARAD_DATASET_AUGM/','path directory of dataset')
flags.DEFINE_integer('M',28,'Spatial dimensions')
flags.DEFINE_integer('L',1,'Spectral dimensions')

flags.DEFINE_string('Net','LeNet5_condition','Net structure for reconstruction, [Net, DeepCubeNet, UNetL-Net,LeNet5]')
flags.DEFINE_string('CA','Binary_1', 'Type of the Coded Aperture[Binary_0,Binary_1,Gray_scale]') #


flags.DEFINE_string('type_reg','Physical', 'Type of the Coded Aperture [Physical]')
flags.DEFINE_float('p',1,'family value')
flags.DEFINE_float('q',1,'family value')

flags.DEFINE_float('reg_HtH',0.5,'regularization parameter for HTH')


flags.DEFINE_float('param_reg',1e-8, 'Regularization parameter for first constraint (binary)') #1e-9
flags.DEFINE_float('aument_reg',15,'Aumented parameter for regularzer param_reg *= aument_Red') #3e1
flags.DEFINE_integer('step_reg',10,'Regulator Increase Step') #10

flags.DEFINE_float('param_reg2',5, 'Regularization parameter for the second regularizer')
flags.DEFINE_integer('kernel',28,'Kernel size of the repeat CA')

flags.DEFINE_float('tram',0.5, 'Transmittance value')

# Tranining paparameters
flags.DEFINE_integer('epochs',50, 'number of epochs')
flags.DEFINE_integer('batch_size',  64, 'batch size')
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

    #------- Load data set-------------------------------

    if(FLAGS.Database=='MNIST'):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = np.expand_dims(x_train,-1)
        x_test = np.expand_dims(x_test,-1)
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    if(FLAGS.Database=='Data_store'):
        split = 1
        Path = FLAGS.Data_path + '/train'
        Aux, train_dataset = Build_data_set(FLAGS.M, FLAGS.M, FLAGS.L, FLAGS.L, FLAGS.batch_size, Path, split, 'mat')
        Path = FLAGS.Data_path + '/val'
        Aux, test_dataset = Build_data_set(FLAGS.M, FLAGS.M, FLAGS.L, FLAGS.L, 6, Path, split, 'mat')


    # --------- Load Model ---------------------------------

    if (FLAGS.Net == 'DeepCubeNet'):
        model = DeepCubeNet(input_size=(FLAGS.M, FLAGS.M, FLAGS.L),FLAGS=FLAGS)
    if(FLAGS.Net == 'Net'):
        model = Net(input_size=(FLAGS.M, FLAGS.M, FLAGS.L),FLAGS=FLAGS)

    if(FLAGS.Net =='UNetL'):
        model=UNetL(input_size=(FLAGS.M,FLAGS.M,FLAGS.L),L=FLAGS.L,FLAGS=FLAGS)

    if(FLAGS.Net =='LeNet5'):
        model = LeNet5(input_size=((FLAGS.M,FLAGS.M,FLAGS.L)),FLAGS=FLAGS)

    if(FLAGS.Net =='LeNet5_condition'):
        model = LeNet5_cond(input_size=((FLAGS.M,FLAGS.M,FLAGS.L)),FLAGS=FLAGS)

    last_epoch = 0
    if FLAGS.Load_weights:
        model.load_weights(FLAGS.Load_path)
        with open(FLAGS.results + '/Weights/last_epoch_info.txt', "r") as txt_log:
            lines = txt_log.readlines()
            last_epoch = int(lines[0].strip())
        print('Loaded pre-trained weights, epoch = '+str(last_epoch))

    #----------Calbacks---------------------------------

    log_dir = "logs/fit/" + str(FLAGS.architecture[0:4]) + '_' + str(FLAGS.compression)[0:4] + '_' + str(
        FLAGS.Net[0:5]) + '_' + str(FLAGS.type_reg[0:6]) + '_' + str(FLAGS.tram) + '_Reg_' + str(
        FLAGS.param_reg) + '_Reg2_' + str(FLAGS.param_reg2) + '_Aum_' + str(FLAGS.aument_reg)[0:2] + \
              '_Step_' + str(FLAGS.step_reg) + '_Ep_' + str(FLAGS.epochs) + '_lr_' + str(
        FLAGS.learning_rate) + '_time_' + datetime.datetime.now().strftime("%m-%d-%H:%M")
    # my call backs
    if(FLAGS.Database=='Data_store'):
        for inp,tar in test_dataset.take(-1):
            x_test=inp.numpy()

    callbacks = [Aument_parameters(p_aum=FLAGS.aument_reg, p_step= FLAGS.step_reg),
                 tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]

    # --------------- Optimization------------------------------------
    optimizad = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    model.compile(optimizer=optimizad, loss=['categorical_crossentropy','mean_squared_error'],loss_weights=[1,FLAGS.reg_HtH],metrics=['accuracy'])

    if (FLAGS.Database == 'MNIST'):
      
        history = model.fit(x_train, [y_train,x_train], epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
                            validation_data=(x_test, [y_test,x_test]), initial_epoch=last_epoch, callbacks=callbacks)

    if (FLAGS.Database == 'Data_store'):
        history = model.fit(train_dataset, epochs=FLAGS.epochs, validation_data=test_dataset, initial_epoch=last_epoch,
        callbacks=callbacks)


    #####################################################
    # -------------Testing Step ------------------------
    #####################################################

    # ------------- Results --------------------------------------------

    try:
      os.stat(FLAGS.results+ '/Weights')
    except:
      os.mkdir(FLAGS.results+'/Weights')
    model.save_weights(FLAGS.results + '/Weights/Last_weights.tf')


    model.summary()
    temporal = model.get_weights()
    # obtain the CA
    CA = temporal[0]
    #CA = kronecker_product_np(np.ones((int(FLAGS.M / FLAGS.kernel), int(FLAGS.M / FLAGS.kernel))),CA)


    if(FLAGS.Database=='MNIST'):
        # Make the measurements
        if(FLAGS.architecture=='Single_Pixel'):

            if(FLAGS.Net == 'LeNet5'):
                modelTest = LeNet5_test(input_size=((FLAGS.M, FLAGS.M, FLAGS.L)), FLAGS=FLAGS)
                it = 0
                for layer in model.layers[2:]:
                    it = it + 1
                    modelTest.layers[it].set_weights(model.get_layer(layer.name).get_weights())


        results = model.evaluate(x_test, [y_test,x_test], batch_size=64)
        Acc = results[3]
        print('Test loss: %.4f accuracy: %.4f' % (results[0], Acc))


    #-----------------Visual Results---------------------
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.savefig(FLAGS.results +'/Loss')
    #plt.show()

    if(len(CA.shape)<5):
        CA = np.expand_dims(CA, axis=3)

    # --------- Coded Aperture---------------
    scipy.io.savemat(FLAGS.results+"/CA.mat", {'CA': CA})


    for iter in range(np.min((CA.shape[4],4))):
      plt.subplot(1,np.min((CA.shape[4],4)),iter+1)
      plt.imshow(CA[0, :, :, 0, iter], 'gray'), plt.title('cod_'+str(iter+1))

    plt.colorbar()
    plt.savefig(FLAGS.results + '/CA')
    #plt.show()

    transmitance = np.sum(CA > 0.5) / (CA.shape[0] * CA.shape[1] * CA.shape[2] * CA.shape[3] * CA.shape[4])
    print('transmitance'+str(transmitance))

    val_m = (np.max(CA)+np.min(CA))/2


    variance_1 = np.var(CA[CA>val_m])
    variance_2 = np.var(CA[CA<val_m])



    if path.exists(FLAGS.results + '/Results_summary.txt'):
        f = open(FLAGS.results + '/Results_summary.txt', "a+")
    else:
        f = open(FLAGS.results + '/Results_summary.txt', "w+")
    # f.write('Transmittance = ' + (str(transmitance)[0:6]) + "\n")
    f.write('Compresion = '+str(FLAGS.compression))
    f.write(', p = ' + str(FLAGS.p))
    f.write(', q = ' + str(FLAGS.q))
    f.write(', Transmittance = ' + (str(transmitance)[0:8]))
    f.write(', Pho = ' + (str(FLAGS.param_reg)))
    f.write(', Pho2 = ' + (str(FLAGS.param_reg2)))
    f.write(', Aug = ' + (str(FLAGS.aument_reg)))
    f.write(', Step_Aug = ' + (str(FLAGS.step_reg)))
    f.write(', Var_1= '+ str(variance_1)[0:5]+str(variance_1)[len(str(variance_1))-4:] +
            ', Var_2= '+ str(variance_2)[0:5]+str(variance_2)[len(str(variance_2))-4:]+
            ', Acc = '+ str(Acc)[0:5] + "\n")


    

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
