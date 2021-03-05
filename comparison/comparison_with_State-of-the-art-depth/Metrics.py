import keras
import tensorflow as tf
import keras.backend as K

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=K.max(y_true))
def cos_distance(y_true, y_pred):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())

    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return K.mean(K.sum(y_true * y_pred, axis=-1))
def relRMSE(y_true,y_pred):
    true_norm = K.sqrt(K.sum(K.square(y_true), axis=-1))
    return K.mean(K.sqrt(keras.losses.mean_squared_error(y_true, y_pred))/true_norm)
def SSIM(y_true,y_pred):
    return tf.image.ssim(y_pred,y_true,K.max(y_true))

def Metrics_batch(y_true,y_pred):
    y_true = tf.convert_to_tensor(y_true,dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred,dtype=tf.float32)

    psnr_v = psnr(y_true,y_pred)
    mse_v   = tf.losses.mse(y_true,y_pred)
    mae_v   = tf.losses.mae(y_true,y_pred)
    ssim_v  = SSIM(y_true,y_pred)

    return psnr_v,mse_v,mae_v,ssim_v
