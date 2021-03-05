from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.constraints import NonNeg
import numpy as np
import tensorflow as tf
from Custumer_Layers import *  # import the SIngle Pixel Layer
from tensorflow.keras.regularizers import l2


def RCU(inputs, feature=256):
    conv1 = ReLU()(inputs)
    conv1 = Conv2D(feature, 3, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(feature, 3, activation=None, padding='same')(conv1)
    final = Add()([inputs, conv1])
    return final


def MRF(input1, input2, feature=256, scale=2):
    conv1 = Conv2D(feature, 3, activation=None, padding='same')(input1)
    conv2 = Conv2D(feature, 3, activation=None, padding='same')(input2)
    conv2 = UpSampling2D(size=(scale, scale))(conv2)
    final = Add()([conv2, conv1])
    return final


def CRP(input1, feature=256):
    conv1 = ReLU()(input1)
    up_1 = MaxPool2D(pool_size=(5, 5), strides=1, padding='same')(conv1)
    up_1 = Conv2D(feature, 3, activation=None, padding='same')(up_1)
    up_2 = MaxPool2D(pool_size=(5, 5), strides=1, padding='same')(up_1)
    up_2 = Conv2D(feature, 3, activation=None, padding='same')(up_2)

    sum_1 = Add()([up_1, conv1])
    final = Add()([sum_1, up_2])
    return final


def MMF(input1, input2, feature=256):
    # first
    input1 = Dropout(0.5)(input1)
    conv1 = Conv2D(feature, 1, activation=None, padding='same')(input1)
    conv1 = RCU(conv1, feature=feature)
    conv1 = RCU(conv1, feature=feature)
    conv1 = Conv2D(feature, 3, activation=None, padding='same')(conv1)

    # second
    input2 = Dropout(0.5)(input2)
    conv2 = Conv2D(feature, 1, activation=None, padding='same')(input2)
    conv2 = RCU(conv2, feature=feature)
    conv2 = RCU(conv2, feature=feature)
    conv2 = Conv2D(feature, 3, activation=None, padding='same')(conv2)
    sum_lay = Add()([conv2, conv1])

    sum_lay = ReLU()(sum_lay)
    fused_up = MaxPool2D(pool_size=(5, 5), strides=1, padding='same')(sum_lay)
    fused_up = Conv2D(feature, 3, activation=None, padding='same')(fused_up)
    final = Add()([sum_lay, fused_up])
    return final


def Refine(input1, input2, feature1=256,feature2=256):
    conv1 = RCU(input1, feature=feature1)
    conv1 = RCU(conv1, feature=feature1)

    conv2 = RCU(input2, feature=feature2)
    conv2 = RCU(conv2, feature=feature2)

    fused = MRF(conv1, conv2, feature=feature1)
    fused = CRP(fused, feature=feature1)
    final = RCU(fused, feature=feature1)

    return final


def Refine4(input1, feature=512):
    conv1 = RCU(input1, feature=feature)
    conv1 = RCU(conv1, feature=feature)
    conv1 = Conv2D(feature, 3, activation=None, padding='same')(conv1)
    fused = CRP(conv1, feature=feature)
    final = RCU(fused, feature=feature)

    return final





def Net(pretrained_weights=None, input_size=(28, 28, 1),FLAGS=None):

    inputs = Input(input_size)

    if(FLAGS.architecture=='CASSI'):
        lay_si = CASSI_layer(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                             type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                             parm2=FLAGS.param_reg2,trans=FLAGS.tram,kern=FLAGS.kernel)(inputs)

    if(FLAGS.architecture=='CASSI_Color'):
        lay_si = CASSI_layer_Colored(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                             type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                             parm2=FLAGS.param_reg2,trans=FLAGS.tram,kern=FLAGS.kernel)(inputs)

    if(FLAGS.architecture=='Single_Pixel'):
        lay_si = Single_Pixel_Layer_trunc(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                                    type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                                    parm2=FLAGS.param_reg2,trans=FLAGS.tram,kern=FLAGS.kernel)(inputs)

    conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(lay_si)
    conv2 = Conv2D(16, 3, activation='relu',padding='same', kernel_initializer='he_normal')(conv1)
    conv3 = Conv2D(32, 3, activation='relu',padding='same', kernel_initializer='he_normal')(conv2)
    conv4 = Conv2D(16, 3, activation='relu',padding='same', kernel_initializer='he_normal')(conv3)
    conv5 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    final = Conv2D(input_size[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def Net_test(pretrained_weights=None, input_size=(28, 28, 1),FLAGS=None):

    inputs = Input(input_size)
    conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv2 = Conv2D(16, 3, activation='relu',padding='same', kernel_initializer='he_normal')(conv1)
    conv3 = Conv2D(32, 3, activation='relu',padding='same', kernel_initializer='he_normal')(conv2)
    conv4 = Conv2D(16, 3, activation='relu',padding='same', kernel_initializer='he_normal')(conv3)
    conv5 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    final = Conv2D(input_size[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def residualNet(pretrained_weights=None, input_size=(28, 28, 1),FLAGS=None):
    inputs = Input(input_size)

    if (FLAGS.architecture == 'CASSI'):
        lay_si = CASSI_layer(output_dim=input_size, input_dim=input_size, shots=1,
                             type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                             parm2=FLAGS.param_reg2, trans=FLAGS.tram, kern=FLAGS.kernel)(inputs)

    if (FLAGS.architecture == 'Single_Pixel'):
        lay_si = Single_Pixel_Layer(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                                    type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                                    parm2=FLAGS.param_reg2, trans=FLAGS.tram, kern=FLAGS.kernel)(inputs)

    conv1 = Conv2D(16, 3, activation='relu', use_bias=True, padding='same', kernel_initializer='he_normal')(lay_si)
    conv1 = Conv2D(32, 3, activation='relu', use_bias=True, padding='same', kernel_initializer='he_normal')(conv1)

    conv8 = Conv2D(16, 3, activation='relu', use_bias=True, padding='same', kernel_initializer='he_normal')(conv1)
    conv8 = Conv2D(1, 3, activation='relu', use_bias=True, padding='same', kernel_initializer='he_normal')(conv8)

    final = Add()([conv8, lay_si])
    final = Conv2D(8, 3, activation='relu', use_bias=True, padding='same', kernel_initializer='he_normal')(final)
    final = Conv2D(16, 3, activation='relu', use_bias=True, padding='same', kernel_initializer='he_normal')(final)
    final = Conv2D(4, 3, activation='relu', use_bias=True, padding='same', kernel_initializer='he_normal')(final)
    final = Conv2D(input_size[2], 3, activation='relu', use_bias=True, padding='same', kernel_initializer='he_normal')(final)
    model = Model(inputs, final)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def UNetL_conditional(pretrained_weights=None,input_size=(512,512,1),L=32,FLAGS=None):
    inputs=Input(input_size)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;

    if (FLAGS.architecture == 'CASSI'):
        lay_si = CASSI_layer(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                             type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                             parm2=FLAGS.param_reg2, trans=FLAGS.tram, kern=FLAGS.kernel)(inputs)

    if(FLAGS.architecture=='CASSI_Color'):
        lay_si = CASSI_layer_Colored(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                             type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                             parm2=FLAGS.param_reg2,trans=FLAGS.tram,kern=FLAGS.kernel)(inputs)

    if (FLAGS.architecture == 'Single_Pixel'):
        lay_si = Single_Pixel_Layer(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                                    type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                                    parm2=FLAGS.param_reg2, trans=FLAGS.tram, kern=FLAGS.kernel)(inputs)

    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(lay_si)
    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(conv3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    up5=Conv2D(L_3,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv4))
    merge5=concatenate([conv3,up5],axis=3)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge5)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv5)

    up6=Conv2D(L_2,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv5))
    merge6=concatenate([conv2,up6],axis=3)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv6)

    up7=Conv2D(L,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv6))
    merge7=concatenate([conv1,up7],axis=3)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge7)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv7)

    final=Conv2D(input_size[2],3,activation='relu',padding='same',kernel_initializer='he_normal',name='recov')(conv7)

    model = Model(inputs, [final,lay_si])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def UNetL(pretrained_weights=None,input_size=(512,512,1),L=32,FLAGS=None):
    inputs=Input(input_size)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;

    if (FLAGS.architecture == 'CASSI'):
        lay_si = CASSI_layer(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                             type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                             parm2=FLAGS.param_reg2, trans=FLAGS.tram, kern=FLAGS.kernel)(inputs)                                         

    if(FLAGS.architecture=='CASSI_Color'):
        lay_si = CASSI_layer_Colored(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                             type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                             parm2=FLAGS.param_reg2,trans=FLAGS.tram,kern=FLAGS.kernel)(inputs) 

    if (FLAGS.architecture == 'Single_Pixel'):
        lay_si = Single_Pixel_Layer(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                                    type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                                    parm2=FLAGS.param_reg2, trans=FLAGS.tram, kern=FLAGS.kernel)(inputs)

    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(lay_si)
    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(conv3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    up5=Conv2D(L_3,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv4))
    merge5=concatenate([conv3,up5],axis=3)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge5)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv5)

    up6=Conv2D(L_2,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv5))
    merge6=concatenate([conv2,up6],axis=3)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv6)

    up7=Conv2D(L,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv6))
    merge7=concatenate([conv1,up7],axis=3)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge7)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv7)

    final=Conv2D(input_size[2],3,activation='relu',padding='same',kernel_initializer='he_normal')(conv7)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def UNetL_class(pretrained_weights=None,input_size1=(512,512,3),input_size2=(512, 512,15),class_n = 10,L=32,batch=2,FLAGS=None):
    inputs=Input(input_size1)
    inputs2 = Input(input_size2)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;

    lay_si = Depth_CA(output_dim=input_size2, input_dim=input_size1,
                         type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                         parm2=FLAGS.param_reg2, trans=FLAGS.tram,batch=batch)([inputs,inputs2])

    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(lay_si)
    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(conv3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    up5=Conv2D(L_3,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv4))
    merge5=concatenate([conv3,up5],axis=3)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge5)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv5)

    up6=Conv2D(L_2,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv5))
    merge6=concatenate([conv2,up6],axis=3)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv6)

    up7=Conv2D(L,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv6))
    merge7=concatenate([conv1,up7],axis=3)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge7)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv7)

    final = Conv2D(class_n, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(conv7)

    model = Model([inputs,inputs2], final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def conv_bn_relu(x, filters, kernel_size=3, l2_reg=1e-6): #5e-4
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def UNetL_class_bn(pretrained_weights=None,input_size1=(512,512,3),input_size2=(512, 512,15),class_n = 10,L=32,batch=2,FLAGS=None):
    inputs=Input(input_size1)
    inputs2 = Input(input_size2)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;

    lay_si = Depth_CA(output_dim=input_size2, input_dim=input_size1,
                         type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                         parm2=FLAGS.param_reg2, trans=FLAGS.tram,batch=batch)([inputs,inputs2])

    conv1 = conv_bn_relu(lay_si,filters=L,kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv1 = conv_bn_relu(conv1, filters=L, kernel_size=3,l2_reg=FLAGS.ker_l2)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = conv_bn_relu(pool1, filters=L_2, kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv2 = conv_bn_relu(conv2, filters=L_2, kernel_size=3,l2_reg=FLAGS.ker_l2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = conv_bn_relu(pool2, filters=L_3, kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv3 = conv_bn_relu(conv3, filters=L_3, kernel_size=3,l2_reg=FLAGS.ker_l2)
    pool3=MaxPooling2D(pool_size=(2,2))(conv3)
    conv4 = conv_bn_relu(pool3, filters=L_4, kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv4 = conv_bn_relu(conv4, filters=L_4, kernel_size=3,l2_reg=FLAGS.ker_l2)
    up5 = conv_bn_relu(UpSampling2D(size=(2, 2))(conv4), filters=L_3, kernel_size=2,l2_reg=FLAGS.ker_l2)
    merge5=concatenate([conv3,up5],axis=3)
    conv5 = conv_bn_relu(merge5, filters=L_3, kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv5 = conv_bn_relu(conv5, filters=L_3, kernel_size=3,l2_reg=FLAGS.ker_l2)

    up6 = conv_bn_relu(UpSampling2D(size=(2, 2))(conv5), filters=L_2, kernel_size=2,l2_reg=FLAGS.ker_l2)
    merge6=concatenate([conv2,up6],axis=3)
    conv6 = conv_bn_relu(merge6, filters=L_2, kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv6 = conv_bn_relu(conv6, filters=L_2, kernel_size=3,l2_reg=FLAGS.ker_l2)

    up7 = conv_bn_relu(UpSampling2D(size=(2,2))(conv6), filters=L, kernel_size=2,l2_reg=FLAGS.ker_l2)
    merge7=concatenate([conv1,up7],axis=3)
    conv7 = conv_bn_relu(merge7, filters=L, kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv7 = conv_bn_relu(conv7, filters=L, kernel_size=3,l2_reg=FLAGS.ker_l2)

    final = Conv2D(class_n, 1, activation='softmax', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(FLAGS.ker_l2))(conv7)

    model = Model([inputs,inputs2], final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


def UNetL_class_bnd(pretrained_weights=None,input_size1=(512,512,3),input_size2=(512, 512,15),class_n = 10,L=32,batch=2,FLAGS=None):
    inputs=Input(input_size1)
    inputs2 = Input(input_size2)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;

    lay_si = Depth_CA(output_dim=input_size2, input_dim=input_size1,
                         type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                         parm2=FLAGS.param_reg2, trans=FLAGS.tram,batch=batch)([inputs,inputs2])

    conv1 = conv_bn_relu(lay_si,filters=L,kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv1 = conv_bn_relu(conv1, filters=L, kernel_size=3,l2_reg=FLAGS.ker_l2)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = conv_bn_relu(pool1, filters=L_2, kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv2 = conv_bn_relu(conv2, filters=L_2, kernel_size=3,l2_reg=FLAGS.ker_l2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = conv_bn_relu(pool2, filters=L_3, kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv3 = conv_bn_relu(conv3, filters=L_3, kernel_size=3,l2_reg=FLAGS.ker_l2)
    drop3 = Dropout(0.5)(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(drop3)
    conv4 = conv_bn_relu(pool3, filters=L_4, kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv4 = conv_bn_relu(conv4, filters=L_4, kernel_size=3,l2_reg=FLAGS.ker_l2)
    drop3 = Dropout(0.5)(conv4)
    up5 = conv_bn_relu(UpSampling2D(size=(2, 2))(drop3), filters=L_3, kernel_size=2,l2_reg=FLAGS.ker_l2)
    merge5=concatenate([conv3,up5],axis=3)
    conv5 = conv_bn_relu(merge5, filters=L_3, kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv5 = conv_bn_relu(conv5, filters=L_3, kernel_size=3,l2_reg=FLAGS.ker_l2)

    up6 = conv_bn_relu(UpSampling2D(size=(2, 2))(conv5), filters=L_2, kernel_size=2,l2_reg=FLAGS.ker_l2)
    merge6=concatenate([conv2,up6],axis=3)
    conv6 = conv_bn_relu(merge6, filters=L_2, kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv6 = conv_bn_relu(conv6, filters=L_2, kernel_size=3,l2_reg=FLAGS.ker_l2)

    up7 = conv_bn_relu(UpSampling2D(size=(2,2))(conv6), filters=L, kernel_size=2,l2_reg=FLAGS.ker_l2)
    merge7=concatenate([conv1,up7],axis=3)
    conv7 = conv_bn_relu(merge7, filters=L, kernel_size=3,l2_reg=FLAGS.ker_l2)
    conv7 = conv_bn_relu(conv7, filters=L, kernel_size=3,l2_reg=FLAGS.ker_l2)

    final = Conv2D(class_n, 1, activation='softmax', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(FLAGS.ker_l2))(conv7)

    model = Model([inputs,inputs2], final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


def UNetL_class_bn_test(pretrained_weights=None,input_size1=(512,512,3),input_size2=(512, 512,15),class_n = 10,L=32,batch=2,FLAGS=None):
    inputs=Input(input_size1)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;

    conv1 = conv_bn_relu(inputs, filters=L, kernel_size=3, l2_reg=FLAGS.ker_l2)
    conv1 = conv_bn_relu(conv1, filters=L, kernel_size=3, l2_reg=FLAGS.ker_l2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_bn_relu(pool1, filters=L_2, kernel_size=3, l2_reg=FLAGS.ker_l2)
    conv2 = conv_bn_relu(conv2, filters=L_2, kernel_size=3, l2_reg=FLAGS.ker_l2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_bn_relu(pool2, filters=L_3, kernel_size=3, l2_reg=FLAGS.ker_l2)
    conv3 = conv_bn_relu(conv3, filters=L_3, kernel_size=3, l2_reg=FLAGS.ker_l2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_bn_relu(pool3, filters=L_4, kernel_size=3, l2_reg=FLAGS.ker_l2)
    conv4 = conv_bn_relu(conv4, filters=L_4, kernel_size=3, l2_reg=FLAGS.ker_l2)
    up5 = conv_bn_relu(UpSampling2D(size=(2, 2))(conv4), filters=L_3, kernel_size=2, l2_reg=FLAGS.ker_l2)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = conv_bn_relu(merge5, filters=L_3, kernel_size=3, l2_reg=FLAGS.ker_l2)
    conv5 = conv_bn_relu(conv5, filters=L_3, kernel_size=3, l2_reg=FLAGS.ker_l2)

    up6 = conv_bn_relu(UpSampling2D(size=(2, 2))(conv5), filters=L_2, kernel_size=2, l2_reg=FLAGS.ker_l2)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = conv_bn_relu(merge6, filters=L_2, kernel_size=3, l2_reg=FLAGS.ker_l2)
    conv6 = conv_bn_relu(conv6, filters=L_2, kernel_size=3, l2_reg=FLAGS.ker_l2)

    up7 = conv_bn_relu(UpSampling2D(size=(2, 2))(conv6), filters=L, kernel_size=2, l2_reg=FLAGS.ker_l2)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = conv_bn_relu(merge7, filters=L, kernel_size=3, l2_reg=FLAGS.ker_l2)
    conv7 = conv_bn_relu(conv7, filters=L, kernel_size=3, l2_reg=FLAGS.ker_l2)

    final = Conv2D(class_n, 1, activation='softmax', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(FLAGS.ker_l2))(conv7)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def UNetL_class_bnd_test(pretrained_weights=None,input_size1=(512,512,3),input_size2=(512, 512,15),class_n = 10,L=32,batch=2,FLAGS=None):
    inputs=Input(input_size1)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;

    conv1 = conv_bn_relu(inputs,filters=L,kernel_size=3)
    conv1 = conv_bn_relu(conv1, filters=L, kernel_size=3)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = conv_bn_relu(pool1, filters=L_2, kernel_size=3)
    conv2 = conv_bn_relu(conv2, filters=L_2, kernel_size=3)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = conv_bn_relu(pool2, filters=L_3, kernel_size=3)
    conv3 = conv_bn_relu(conv3, filters=L_3, kernel_size=3)
    drop3 = Dropout(0.5)(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(drop3)
    conv4 = conv_bn_relu(pool3, filters=L_4, kernel_size=3)
    conv4 = conv_bn_relu(conv4, filters=L_4, kernel_size=3)
    drop4 = Dropout(0.5)(conv4)
    up5 = conv_bn_relu(UpSampling2D(size=(2, 2))(drop4), filters=L_3, kernel_size=2)
    merge5=concatenate([conv3,up5],axis=3)
    conv5 = conv_bn_relu(merge5, filters=L_3, kernel_size=3)
    conv5 = conv_bn_relu(conv5, filters=L_3, kernel_size=3)

    up6 = conv_bn_relu(UpSampling2D(size=(2, 2))(conv5), filters=L_2, kernel_size=2)
    merge6=concatenate([conv2,up6],axis=3)
    conv6 = conv_bn_relu(merge6, filters=L_2, kernel_size=3)
    conv6 = conv_bn_relu(conv6, filters=L_2, kernel_size=3)

    up7 = conv_bn_relu(UpSampling2D(size=(2,2))(conv6), filters=L, kernel_size=2)
    merge7=concatenate([conv1,up7],axis=3)
    conv7 = conv_bn_relu(merge7, filters=L, kernel_size=3)
    conv7 = conv_bn_relu(conv7, filters=L, kernel_size=3)

    final = Conv2D(class_n, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(conv7)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def UNetL_class_test(pretrained_weights=None,input_size1=(512,512,3),input_size2=(512, 512,15),class_n = 10,L=32,batch=2,FLAGS=None):
    inputs=Input(input_size1)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;


    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(inputs)
    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(conv3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    up5=Conv2D(L_3,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv4))
    merge5=concatenate([conv3,up5],axis=3)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge5)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv5)

    up6=Conv2D(L_2,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv5))
    merge6=concatenate([conv2,up6],axis=3)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv6)

    up7=Conv2D(L,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv6))
    merge7=concatenate([conv1,up7],axis=3)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge7)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv7)

    final = Conv2D(class_n, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(conv7)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model



def UNetL_test(pretrained_weights=None,input_size=(512,512,1),L=32,FLAGS=None):
    inputs=Input(input_size)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;

    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(inputs)
    conv1=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(conv3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    conv4=Conv2D(L_4,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    up5=Conv2D(L_3,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv4))
    merge5=concatenate([conv3,up5],axis=3)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge5)
    conv5=Conv2D(L_3,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv5)

    up6=Conv2D(L_2,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv5))
    merge6=concatenate([conv2,up6],axis=3)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
    conv6=Conv2D(L_2,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv6)

    up7=Conv2D(L,2,activation='relu',padding='same',kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv6))
    merge7=concatenate([conv1,up7],axis=3)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge7)
    conv7=Conv2D(L,3,activation='relu',padding='same',kernel_initializer='he_normal',)(conv7)

    final=Conv2D(input_size[2],3,activation='relu',padding='same',kernel_initializer='he_normal',)(conv7)



    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model



def UNet3L(pretrained_weights=None,input_size=(512,512,1),L=32,FLAGS=None):
    inputs = Input(input_size)
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;

    if (FLAGS.architecture == 'CASSI'):
        lay_si = CASSI_layer(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                             type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                             parm2=FLAGS.param_reg2, trans=FLAGS.tram, batch_size=FLAGS.batch_size, noise=FLAGS.noise,
                             kern=FLAGS.kernel)(inputs)

    if (FLAGS.architecture == 'CASSI_Color'):
        lay_si = CASSI_layer_Colored(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                                     type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                                     parm2=FLAGS.param_reg2, trans=FLAGS.tram, kern=FLAGS.kernel)(inputs)

    if (FLAGS.architecture == 'Single_Pixel'):
        lay_si = Single_Pixel_Layer(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                                    type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                                    parm2=FLAGS.param_reg2, trans=FLAGS.tram, kern=FLAGS.kernel)(inputs)

    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(lay_si)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    up5 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv7)

    final = Conv2D(input_size[2], 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv7)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def UNet3L_test(pretrained_weights=None,input_size=(512,512,1),L=32,FLAGS=None):
    inputs=Input(input_size)
    L_2=2 * L;
    L_3=3 * L;
    L_4=4 * L;



    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)

    up5 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv7)

    final = Conv2D(input_size[2], 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv7)

    model = Model(inputs, final)


    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def LeNet5(pretrained_weights=None,input_size=(512,512,1),num_classes=10,FLAGS=None):
    inputs = Input(input_size)

    if (FLAGS.architecture == 'Single_Pixel'):
        lay_si = Single_Pixel_Layer(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                                    type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                                    parm2=FLAGS.param_reg2, trans=FLAGS.tram, kern=FLAGS.kernel, p=FLAGS.p, q=FLAGS.q)(inputs)

    conv1 = Conv2D(32, 3, activation='relu', padding='same')(lay_si)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    flat  = Flatten()(pool3)
    Den1  = Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001))(flat)
    final = Dense(num_classes,activation='softmax')(Den1)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def LeNet5_cond(pretrained_weights=None,input_size=(512,512,1),num_classes=10,FLAGS=None):
    inputs = Input(input_size)

    if (FLAGS.architecture == 'Single_Pixel'):
        lay_si = Single_Pixel_Layer(output_dim=input_size, input_dim=input_size, compression=FLAGS.compression,
                                    type_code=FLAGS.CA, type_reg=FLAGS.type_reg, parm1=FLAGS.param_reg,
                                    parm2=FLAGS.param_reg2, trans=FLAGS.tram, kern=FLAGS.kernel, p=FLAGS.p, q=FLAGS.q)(inputs)

    conv1 = Conv2D(32, 3, activation='relu', padding='same')(lay_si)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    flat  = Flatten()(pool3)
    Den1  = Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001))(flat)
    final = Dense(num_classes,activation='softmax')(Den1)

    model = Model(inputs, [final,lay_si])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def LeNet5_test(pretrained_weights=None,input_size=(512,512,1),num_classes=10,FLAGS=None):
    inputs = Input(input_size)

    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    flat = Flatten()(pool3)
    Den1 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(flat)
    final = Dense(num_classes, activation='softmax')(Den1)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

