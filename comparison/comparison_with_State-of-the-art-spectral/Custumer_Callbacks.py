
import tensorflow as tf
import os
from Function_project import *
from keras import backend as K
import matplotlib.pyplot as plt

class Aument_parameters(tf.keras.callbacks.Callback):
    def __init__(self, p_aum,p_step):
        super().__init__()
        self.p_aum = p_aum
        self.p_step = p_step


    def on_epoch_end(self, epoch, logs=None):

        if(tf.math.floormod(epoch,self.p_step)==(self.p_step-1)):
            param=self.model.layers[1].my_regularizer.parameter
            param = tf.keras.backend.get_value(param)
            self.model.layers[1].my_regularizer.parameter.assign(param * self.p_aum)
            tf.print(' regularizator ='+ str(param* self.p_aum))


class Depth_Aument_parameters(tf.keras.callbacks.Callback):
    def __init__(self, p_aum,p_step):
        super().__init__()
        self.p_aum = p_aum
        self.p_step = p_step


    def on_epoch_end(self, epoch, logs=None):

        if(tf.math.floormod(epoch,self.p_step)==(self.p_step-1)):
            param=self.model.layers[2].my_regularizer.parameter
            param = tf.keras.backend.get_value(param)
            self.model.layers[2].my_regularizer.parameter.assign(param * self.p_aum)
            tf.print(' regularizator ='+ str(param* self.p_aum))

class Aument_parameters2(tf.keras.callbacks.Callback):
    def __init__(self, p_aum,p_step):
        super().__init__()
        self.p_aum = p_aum
        self.p_step = p_step


    def on_epoch_end(self, epoch, logs=None):

        if(tf.math.floormod(epoch,self.p_step)==(self.p_step-1)):
            param=self.model.layers[1].my_regularizer.parameter
            param2 = self.model.layers[1].my_regularizer.parameter2
            param = tf.keras.backend.get_value(param)
            param2 = tf.keras.backend.get_value(param2)

            #tf.print(tf.keras.backend.get_value(self.model.layers[1].my_regularizer.T))

            self.model.layers[1].my_regularizer.parameter.assign(param * self.p_aum)
            #self.model.layers[1].my_regularizer.parameter2.assign(param2 * self.p_aum)

            tf.print(' - reg: '+ str(param* self.p_aum)+ ' -reg_2: '+ str(param2* self.p_aum))


class save_each_epoch(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.best = 0


    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(self.checkpoint_dir)









class SaveLastEpochWeights_cond(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        super(SaveLastEpochWeights_cond, self).__init__()
        self.checkpoint_dir = checkpoint_dir+'/Weights'
        try:
            os.stat(self.checkpoint_dir)
            if(os.path.isfile(self.checkpoint_dir + '/best_epoch_info.txt')):
                with open(self.checkpoint_dir + '/best_epoch_info.txt', "r") as txt_log:
                    lines = txt_log.readlines()
                    val_best = float(lines[1].strip())
                    self.best_c = val_best
            else:
                self.best_c = 0
        except:
            os.mkdir(self.checkpoint_dir)
        self.best_c = 0


    def on_epoch_end(self, epoch, logs=None):
        path_name = self.checkpoint_dir + '/Net_last.tf'
        with open(self.checkpoint_dir + '/last_epoch_info.txt', "w") as txt_log:
            txt_log.write(str(epoch + 1))
            txt_log.write("\n")
            txt_log.write("Validation Loss for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_loss']))
            txt_log.write("\n")
            txt_log.write("VAL_PSNR for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_recov_psnr']))
            txt_log.write("\n")
            txt_log.write("VAl_ PSNR HTH for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_cassi_layer_1_psnr']))
            txt_log.write("\n")
            txt_log.write("Train Loss for epoch {} is {:7.7f} ".format(epoch + 1, logs['loss']))
            txt_log.write("\n")
            txt_log.write("Train PSNR for epoch {} is {:7.7f} ".format(epoch + 1, logs['recov_psnr']))
            txt_log.write("\n")
            txt_log.write("Train PSNR_HTH for epoch {} is {:7.7f} ".format(epoch + 1, logs['cassi_layer_1_psnr']))
        self.model.save_weights(path_name)
        if logs['val_recov_psnr'] > self.best_c:
            self.best_c = logs['val_recov_psnr']
            with open(self.checkpoint_dir + '/best_epoch_info.txt', "w") as txt_log:
                txt_log.write(str(epoch + 1))
                txt_log.write("\n")
                txt_log.write(str(logs['val_recov_psnr']))
                txt_log.write("\n")
                txt_log.write("Validation Loss for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_loss']))
                txt_log.write("\n")
                txt_log.write("VAL_PSNR for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_recov_psnr']))
                txt_log.write("\n")
                txt_log.write(
                    "VAl_ PSNR HTH for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_cassi_layer_1_psnr']))
                txt_log.write("\n")
                txt_log.write("Train Loss for epoch {} is {:7.7f} ".format(epoch + 1, logs['loss']))
                txt_log.write("\n")
                txt_log.write("Train PSNR for epoch {} is {:7.7f} ".format(epoch + 1, logs['recov_psnr']))
                txt_log.write("\n")
                txt_log.write("Train PSNR_HTH for epoch {} is {:7.7f} ".format(epoch + 1, logs['cassi_layer_1_psnr']))
            self.model.save_weights(self.checkpoint_dir + '/Net_best.tf')


class SaveLastEpochWeights(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        super(SaveLastEpochWeights, self).__init__()
        self.checkpoint_dir = checkpoint_dir+'/Weights'
        try:
            os.stat(self.checkpoint_dir)
            if(os.path.isfile(self.checkpoint_dir + '/best_epoch_info.txt')):
                with open(self.checkpoint_dir + '/best_epoch_info.txt', "r") as txt_log:
                    lines = txt_log.readlines()
                    val_best = float(lines[1].strip())
                    self.best_c = val_best
            else:
                self.best_c = 0
        except:
            os.mkdir(self.checkpoint_dir)
        self.best_c = 0


    def on_epoch_end(self, epoch, logs=None):
        path_name = self.checkpoint_dir + '/Net_last.tf'
        with open(self.checkpoint_dir + '/last_epoch_info.txt', "w") as txt_log:
            txt_log.write(str(epoch + 1))
            txt_log.write("\n")
            txt_log.write("Validation Loss for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_loss']))
            txt_log.write("\n")
            txt_log.write("VAL_PSNR for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_psnr']))
            txt_log.write("\n")
            txt_log.write("VAl_MSE for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_mse']))
            txt_log.write("\n")
            txt_log.write("Val_MAE for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_mae']))
            txt_log.write("\n")
            txt_log.write("Val_SSIM for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_SSIM']))
            txt_log.write("\n")
            txt_log.write("Train Loss for epoch {} is {:7.7f} ".format(epoch + 1, logs['loss']))
            txt_log.write("\n")
            txt_log.write("Train_PSNR for epoch {} is {:7.7f} ".format(epoch + 1, logs['psnr']))
            txt_log.write("\n")
            txt_log.write("Train_MSE for epoch {} is {:7.7f} ".format(epoch + 1, logs['mse']))
            txt_log.write("\n")
            txt_log.write("Train_MAE for epoch {} is {:7.7f} ".format(epoch + 1, logs['mae']))
            txt_log.write("\n")
            txt_log.write("Train_SSIM for epoch {} is {:7.7f} ".format(epoch + 1, logs['SSIM']))
        self.model.save_weights(path_name)
        if logs['val_psnr'] > self.best_c:
            self.best_c = logs['val_psnr']
            with open(self.checkpoint_dir + '/best_epoch_info.txt', "w") as txt_log:
                txt_log.write(str(epoch + 1))
                txt_log.write("\n")
                txt_log.write(str(logs['val_psnr']))
                txt_log.write("\n")
                txt_log.write("Validation Loss for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_loss']))
                txt_log.write("\n")
                txt_log.write("VAL_PSNR for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_psnr']))
                txt_log.write("\n")
                txt_log.write("VAl_MSE for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_mse']))
                txt_log.write("\n")
                txt_log.write("Val_MAE for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_mae']))
                txt_log.write("\n")
                txt_log.write("Val_SSIM for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_SSIM']))
                txt_log.write("\n")
                txt_log.write("Train Loss for epoch {} is {:7.7f} ".format(epoch + 1, logs['loss']))
                txt_log.write("\n")
                txt_log.write("Train_PSNR for epoch {} is {:7.7f} ".format(epoch + 1, logs['psnr']))
                txt_log.write("\n")
                txt_log.write("Train_MSE for epoch {} is {:7.7f} ".format(epoch + 1, logs['mse']))
                txt_log.write("\n")
                txt_log.write("Train_MAE for epoch {} is {:7.7f} ".format(epoch + 1, logs['mae']))
                txt_log.write("\n")
                txt_log.write("Train_SSIM for epoch {} is {:7.7f} ".format(epoch + 1, logs['SSIM']))
            self.model.save_weights(self.checkpoint_dir + '/Net_best.tf')

class SaveLastEpochWeights2(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        super(SaveLastEpochWeights2, self).__init__()
        self.checkpoint_dir = checkpoint_dir+'/Weights'
        try:
            os.stat(self.checkpoint_dir)
            with open(self.checkpoint_dir + '/best_epoch_info.txt', "r") as txt_log:
                lines = txt_log.readlines()
                val_best = float(lines[1].strip())
            self.best_c = val_best

        except:
            os.mkdir(self.checkpoint_dir)
            self.best_c = 0



    def on_epoch_end(self, epoch, logs=None):
        path_name = self.checkpoint_dir + '/Net_last.tf'
        with open(self.checkpoint_dir + '/last_epoch_info.txt', "w") as txt_log:
            txt_log.write(str(epoch + 1))
            txt_log.write("\n")
            txt_log.write("Validation Loss for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_loss']))
            txt_log.write("\n")
            txt_log.write("Validation Acc for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_accuracy']))
            txt_log.write("\n")
            txt_log.write("Train Loss for epoch {} is {:7.7f} ".format(epoch + 1, logs['loss']))
            txt_log.write("\n")
            txt_log.write("Train Acc for epoch {} is {:7.7f} ".format(epoch + 1, logs['accuracy']))
        self.model.save_weights(path_name)
        if logs['val_accuracy'] > self.best_c:
            self.best_c = logs['val_accuracy']
            with open(self.checkpoint_dir + '/best_epoch_info.txt', "w") as txt_log:
                txt_log.write(str(epoch + 1))
                txt_log.write("\n")
                txt_log.write(str(logs['val_accuracy']))
                txt_log.write("\n")
                txt_log.write("Validation Loss for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_loss']))
                txt_log.write("\n")
                txt_log.write("Validation Acc for epoch {} is {:7.7f} ".format(epoch + 1, logs['val_accuracy']))
                txt_log.write("\n")
                txt_log.write("Train Loss for epoch {} is {:7.7f} ".format(epoch + 1, logs['loss']))
                txt_log.write("\n")
                txt_log.write("Train Acc for epoch {} is {:7.7f} ".format(epoch + 1, logs['accuracy']))
            self.model.save_weights(self.checkpoint_dir + '/Net_best.tf')


class Print_results(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir,images,kernel,M):
        super(Print_results, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.images=images
        self.kernel=kernel
        self.M = M

    def on_epoch_end(self, epoch, logs=None):
        CA=self.model.get_weights()[0]
        CA=kronecker_product_np(np.ones((int(self.M / self.kernel),int(self.M / self.kernel))),CA)

        if (len(CA.shape) < 5):
            CA = np.expand_dims(CA, axis=3)

        plt.subplot(141)
        plt.imshow(CA[0, :, :, 0, 0], 'gray'), plt.title('cod_1')
        plt.subplot(142)
        plt.imshow(CA[0, :, :, 0, 1], 'gray'), plt.title('cod_2')
        plt.subplot(143)
        plt.imshow(CA[0, :, :, 0, 2], 'gray'), plt.title('cod_3')
        plt.subplot(144)
        plt.imshow(CA[0, :, :, 0, 3], 'gray'), plt.title('cod_4'), plt.colorbar()
        plt.savefig(self.checkpoint_dir+'/CA')
        plt.close()

        func=K.function([self.model.layers[0].input],[self.model.layers[len(self.model.layers)-1].output])
        result=func(self.images)
        result=np.squeeze(np.asarray(result),0)
        plt.subplot(141)
        plt.imshow(result[0,:,:,0],'gray'),plt.title('reconstruction')
        plt.subplot(142)
        plt.imshow(self.images[0,:,:,0],'gray'),plt.title('original')
        plt.subplot(143)
        plt.imshow(result[2,:,:,0],'gray'),plt.title('reconstruction')
        plt.subplot(144)
        plt.imshow(self.images[2,:,:,0],'gray'),plt.title('original')
        plt.savefig(self.checkpoint_dir+'/Reconstruction')
        plt.close()

