from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
from Custumer_Regularizers import *
from tensorflow.keras.constraints import MinMaxNorm,NonNeg
from Function_project import *


#-------------------------------------------- CASSI LAYER-----------------------------------------
class CASSI_layer(Layer):
    def __init__(self, output_dim=(256, 256, 12), input_dim=(256, 256, 12), compression=1, parm1=0., parm2=0.,
                 type_code='Binary_1', trans=0.5, type_reg='Reg_Binary_1_1', kern=256, noise=40, batch_size=1, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.shots = int(np.round((compression*(input_dim[0]*input_dim[1]*input_dim[2]))/(input_dim[0]*(input_dim[1]+input_dim[2]-1))))
        tf.print(self.shots)
        self.parms1 = parm1
        self.parms2 = parm2
        self.type_code = type_code
        self.type_reg = type_reg
        self.trans = trans
        self.kern = kern
        self.noise = noise
        self.batch_size = batch_size

        if self.type_code == 'Binary_0':
            if self.type_reg =='Physical':
                self.my_regularizer = Reg_Binary_0_1(parm1)

        if self.type_code == 'Binary_1':
            if self.type_reg =='Physical':
                self.my_regularizer = Reg_Binary_1_1(parm1)

            if self.type_reg == 'Transmittance':
                self.my_regularizer = Reg_Binary_1_1_transm(parameter=self.parms1,
                                                            parameter2=self.parms2, tram=self.trans)



        super(CASSI_layer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'shots': self.shots,
            'parms1': self.parms1,
            'parms2': self.parms2,
            'type_code': self.type_code,
            'trans': self.trans,
            'kern': self.kern,
            'noise': self.noise,
            'batch_size': self.batch_size,
            'my_regularizer': self.my_regularizer,
            'type_reg': self.type_reg})
        return config

    def build(self, input_shape):


        if self.type_code == 'Binary_0':
            H_init = np.random.rand(1, self.kern, self.kern, self.shots)
            #H_init = np.random.normal(0, 1, (1, self.kern, self.kern, 1, self.shots)) / np.sqrt(
                #self.kern * self.kern)+0.5
            H_init = tf.constant_initializer(H_init)
            self.H = self.add_weight(name='H', shape=(1,self.kern, self.kern, 1,self.shots), initializer=H_init, trainable=True,
                                        regularizer=self.my_regularizer)

        # --------Binary -1 1 -------
        if self.type_code == 'Binary_1':
            H_init = np.random.normal(0, 1, (1, self.kern, self.kern,1,self.shots)) / np.sqrt(
                self.kern * self.kern)
            H_init = tf.constant_initializer(H_init)

            self.H = self.add_weight(name='H', shape=(1, self.kern, self.kern,1,self.shots),
                                     initializer=H_init, trainable=True,
                                     regularizer=self.my_regularizer)

            # --------Gray scale -------
        if self.type_code == 'Gray_scale':

            H_init = np.random.normal(0, 1, (1, self.kern, self.kern, self.shots)) / np.sqrt(
                self.kern * self.kern)
            H_init = tf.constant_initializer(H_init)

            if self.type_reg == 'Physical':
                self.H = self.add_weight(name='H', shape=(1, self.kern, self.kern,1,self.shots),
                                        initializer=H_init, trainable=True,
                                        constraint=MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0))


    def call(self, inputs, **kwargs):

        #H = kronecker_product(tf.ones((int(self.input_dim[0] / self.kern), int(self.input_dim[1] / self.kern))), self.H)
        #H = self.H + tf.random.uniform(shape=self.H.shape,minval=0, maxval=0.2, dtype=self.H.dtype)
        H = self.H
        L = self.input_dim[2]
        M = self.input_dim[0]
        # CASSI Sensing Model
        Aux1 = tf.multiply(H, tf.expand_dims(inputs,-1))
        Aux1 = tf.pad(Aux1, [[0, 0], [0, 0], [0, L - 1], [0, 0],[0, 0]])
        Y = None
        for i in range(L):
            Tempo = tf.roll(Aux1, shift=i, axis=2)
            if Y is not None:
                Y = tf.concat([Y, tf.expand_dims(Tempo[:, :, :, i], -1)], axis=4)
            else:
                Y = tf.expand_dims(Tempo[:, :, :, i], -1)
        Y = tf.reduce_sum(Y, 4)
        sigm = tf.reduce_sum(tf.math.pow(Y,2)) / ((M*(M+(L-1))*self.batch_size) * 10 ** (self.noise / 10))
        Y   = Y + tf.random.normal(shape =(self.batch_size,M,M+L-1,1), mean=0,stddev = tf.math.sqrt(sigm),dtype=Y.dtype)



        # CASSI Transpose model (x = H'*y)
        X = None
        for i in range(L):
            Tempo = tf.roll(Y, shift=-i, axis=2)
            if X is not None:
                X = tf.concat([X, tf.expand_dims(Tempo[:, :, 0:M], -1)], axis=4)
            else:
                X = tf.expand_dims(Tempo[:, :, 0:M], -1)

        X = tf.transpose(X,[0,1,2,4,3])
        X = tf.multiply(H, X)
        X = tf.reduce_sum(X,4)

        X = X / tf.math.reduce_max(X)

        return X

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


#------------------------------------ CASSI COLORED ------------------------------------------------

class CASSI_layer_Colored(Layer):
    def __init__(self, output_dim=(256, 256, 12), input_dim=(256, 256, 12), compression=1, parm1=0., parm2=0.,
                 type_code='Binary_1', trans=0.5, type_reg='Reg_Binary_1_1', kern=256, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.shots = int(np.round((compression*(input_dim[0]*input_dim[1]*input_dim[2]))/(input_dim[0]*(input_dim[1]+input_dim[2]-1))))
        self.parms1 = parm1
        self.parms2 = parm2
        self.type_code = type_code
        self.type_reg = type_reg
        self.trans = trans
        self.kern = kern

        fr,fg,fc,fb=get_color_bases(np.linspace(400, 700, input_dim[2])*1e-9)

        self.fr = tf.expand_dims(tf.expand_dims(fr, -1), 0)
        self.fg = tf.expand_dims(tf.expand_dims(fg, -1), 0)
        self.fc = tf.expand_dims(tf.expand_dims(fc, -1), 0)
        self.fb = tf.expand_dims(tf.expand_dims(fb, -1), 0)


        super(CASSI_layer_Colored, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'shots': self.shots,
            'parms1': self.parms1,
            'parms2': self.parms2,
            'type_code': self.type_code,
            'trans': self.trans,
            'kern': self.kern,
            'my_regularizer': self.my_regularizer,
            'type_reg': self.type_reg,
            'fr': self.fr,
            'fg': self.fg,
            'fc': self.fc,
            'fb': self.fb})
        return config

    def build(self, input_shape):

        wr = (np.random.rand(self.kern, self.kern, self.shots))
        wg = (np.random.rand(self.kern, self.kern, self.shots))
        wb = (np.random.rand(self.kern, self.kern, self.shots))
        wc = (np.random.rand(self.kern, self.kern, self.shots))
        wt = wr + wg + wb + wc
        wr = tf.constant_initializer(tf.math.divide(wr, wt))
        wg = tf.constant_initializer(tf.math.divide(wg, wt))
        wb = tf.constant_initializer(tf.math.divide(wb, wt))
        wc = tf.constant_initializer(tf.math.divide(wc, wt))

        self.wr = self.add_weight(name='wr', shape=(1, self.kern, self.kern, 1, self.shots),
                                  initializer=wr, trainable=True, constraint=NonNeg())
        self.wg = self.add_weight(name='wg', shape=(1, self.kern, self.kern, 1, self.shots),
                                  initializer=wg, trainable=True, constraint=NonNeg())
        self.wb = self.add_weight(name='wb', shape=(1, self.kern, self.kern, 1, self.shots),
                                  initializer=wb, trainable=True, constraint=NonNeg())
        self.wc = self.add_weight(name='wc', shape=(1, self.kern, self.kern, 1, self.shots),
                                  initializer=wc, trainable=True, constraint=NonNeg())


    def call(self, inputs, **kwargs):

        wt = self.wr + self.wg + self.wb + self.wc
        wr = tf.math.divide(self.wr, wt)
        wg = tf.math.divide(self.wg, wt)
        wb = tf.math.divide(self.wb, wt)
        wc = tf.math.divide(self.wc, wt)

        H = tf.multiply(wr, self.fr) + tf.multiply(wg, self.fg) + tf.multiply(wb, self.fb) + tf.multiply(wc, self.fc)

        H = kronecker_product_color(tf.ones((int(self.input_dim[0] / self.kern), int(self.input_dim[1] / self.kern))), H)

        L = self.input_dim[2]
        M = self.input_dim[0]
        # CASSI Sensing Model
        Aux1 = tf.multiply(H, tf.expand_dims(inputs,-1))
        Aux1 = tf.pad(Aux1, [[0, 0], [0, 0], [0, L - 1], [0, 0],[0, 0]])
        Y = None
        for i in range(L):
            Tempo = tf.roll(Aux1, shift=i, axis=2)
            if Y is not None:
                Y = tf.concat([Y, tf.expand_dims(Tempo[:, :, :, i], -1)], axis=4)
            else:
                Y = tf.expand_dims(Tempo[:, :, :, i], -1)
        Y = tf.reduce_sum(Y, 4)

        # CASSI Transpose model (x = H'*y)
        X = None
        for i in range(L):
            Tempo = tf.roll(Y, shift=-i, axis=2)
            if X is not None:
                X = tf.concat([X, tf.expand_dims(Tempo[:, :, 0:M], -1)], axis=4)
            else:
                X = tf.expand_dims(Tempo[:, :, 0:M], -1)

        X = tf.transpose(X,[0,1,2,4,3])
        X = tf.multiply(H, X)
        X = tf.reduce_sum(X,4)

        X = X / tf.math.reduce_max(X)

        return X

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)




class Depth_CA(Layer):
    def __init__(self, output_dim=(256, 256, 12), input_dim=(256, 256, 12), parm1=0., parm2=0.,
                trans=0.5, kern=32, z0 = 2.5, zi = 50e-3,radii = 2e-3, px = 6.22e-6, batch=2,
                 type_code='Binary_0', type_reg='Reg_Binary_1_1', **kwargs):

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.parms1 = parm1
        self.parms2 = parm2
        self.trans = trans
        self.kern = kern
        self.batch = batch
        self.type_code = type_code
        self.type_reg = type_reg
        self.N = input_dim[0]
        self.z0  = 0
        self.zi = zi
        self.f = 1/(1/zi + 1/z0)

        self.L_sen = px*self.N
        self.L_len = 2*radii*2
        self.lamb = np.asarray([460, 550,640.])*1e-9
        self.R = self.f * deta(550e-9 * 1e6)
        self.flmb = self.R / deta(self.lamb * 1e6)
        self.z = np.sort(-3*(np.log(np.linspace(0.9,11,15)))+8)
        self.du = self.L_len / self.N;
        u = np.arange(-self.L_len / 2, self.L_len / 2, self.du)
        [X, Y] = np.meshgrid(u, u)
        self.XY = (X * X + Y * Y)
        [r, thetha] = cart2pol(X, Y)
        rad = r <= radii
        self.rad = tf.cast(rad, dtype=tf.complex64)

        fx1 = np.arange(-1 / (2 * self.du), 1 / (2 * self.du), 1 / self.L_len)
        fx1 = np.fft.fftshift(fx1)
        [FX1, FY1] = np.meshgrid(fx1, fx1)
        self.FF = FX1 * FX1 + FY1 * FY1

        self.dx2 = self.L_sen / self.N
        x2 = np.arange(-self.L_sen / 2, self.L_sen / 2, self.dx2)
        [X2, Y2] = np.meshgrid(x2, x2)
        self.XY2 = X2 * X2 + Y2 * Y2

        if self.type_code == 'Binary_0':
            if self.type_reg == 'Physical':
                self.my_regularizer = Reg_Binary_0_1(parm1)

            if self.type_reg =='Real_value':
                self.my_regularizer = Reg_Real_value(parm1)


        if self.type_code == 'Binary_1':
            if self.type_reg == 'Physical':
                self.my_regularizer = Reg_Binary_1_1(parm1)

            if self.type_reg == 'Transmittance':
                self.my_regularizer = Reg_Binary_1_1_transm(parameter=self.parms1,
                                                            parameter2=self.parms2, tram=((self.trans)-(1-self.trans)))

        super(Depth_CA, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'parms1': self.parms1,
            'parms2': self.parms2,
            'trans': self.trans,
            'kern': self.kern,
            'batch': self.batch,
            'N': self.N,
            'z0': self.z0,
            'zi': self.zi,
            'f': self.f,
            'type_code': self.type_code,
            'my_regularizer': self.my_regularizer,
            'type_reg': self.type_reg,
            'L_sen': self.L_sen,
            'L_len': self.L_len,
            'lamb': self.lamb,
            'R': self.R,
            'flmb': self.flmb,
            'z': self.z,
            'du': self.du,
            'XY': self.XY,
            'rad': self.rad,
            'FF':self.FF,
            'XY2': self.XY2,
            'dx2': self.dx2})
        return config

    def build(self, input_shape):

        parameter = tf.constant_initializer(0.2)

        H_init = np.random.rand(1, self.kern, self.kern, 1)
        #H_init = np.random.normal(0, 1, (1, self.kern, self.kern, M)) / np.sqrt(
           # self.kern * self.kern)
        H_init = tf.constant_initializer(H_init)

        self.H = self.add_weight(name='H', shape=(1, self.kern, self.kern, 1),
                                 initializer=H_init, trainable=True,
                                 regularizer=self.my_regularizer)

        self.parameter = self.add_weight(name='parameter', shape=(1),
                                  initializer=parameter, trainable=True)




    def call(self, inputs, **kwargs):

        CA = kronecker_product_depth(self.H, tf.ones((16, 16)))
        CA = tf.cast(tf.squeeze(CA), dtype=tf.complex64)

        lt = 0
        psfs = None
        for dis in self.z:
            t_psf = None
            for band in  self.lamb:

                k = (2 * np.pi) / (band)

                # constant = tf.expand_dims(tf.expand_dims(-(k / (2 * flmb[lt])),0),0)
                # t= tf.multiply(tf.multiply(constant,XY),CA)

                t = tf.multiply(compl_exp_tf(-(k / (2 *  self.flmb[lt])) * ( self.XY)), CA)
                focus = compl_exp_tf((k / (2 * dis)) * ( self.XY))

                ph = tf.multiply(self.rad, tf.multiply(t, focus))

                vu = tf.multiply(ph, compl_exp_tf((np.pi / (band * self.zi * self.L_len) * (self.L_len - self.L_sen)) * (self.XY)))
                vu = tf.signal.fft2d(tf.signal.fftshift(vu))

                vu = tf.multiply(vu, compl_exp_tf(-(np.pi * band * self.zi * self.L_len / self.L_sen) * (self.FF)))

                vu = tf.signal.ifftshift(tf.signal.ifft2d(vu))

                vu = (self.L_sen / self.L_len) * tf.multiply(vu, compl_exp_tf(
                    -(np.pi / (band * self.zi * self.L_sen) * (self.L_len - self.L_sen)) * (self.XY2)))
                psf = tf.pow(tf.abs(vu * ((self.du * self.du) / (self.dx2 * self.dx2))), 2)
                psf = psf / tf.reduce_sum(psf)

                if t_psf is not None:
                    t_psf = tf.concat([t_psf, tf.expand_dims(psf, -1)], axis=2)
                else:
                    t_psf = tf.expand_dims(psf, -1)
            if psfs is not None:
                psfs = tf.concat([psfs, tf.expand_dims(t_psf, -1)], axis=3)
            else:
                psfs = tf.expand_dims(t_psf, -1)

        psfs = tf.transpose(psfs, [3, 0, 1, 2])

        img = inputs[0]
        Map = inputs[1]

        Map = tf.expand_dims(Map,3)
        img_ft = tf.signal.fftshift(transp_fft2d(tf.signal.ifftshift(img, axes=[1, 2])), axes=[1, 2])
        psf_fr = tf.signal.fftshift(transp_fft2d(tf.signal.ifftshift(psfs, axes=[1, 2])), axes=[1, 2])

        img_ft = tf.expand_dims(img_ft, -1)
        psf_fr = tf.expand_dims(tf.transpose(psf_fr, [1, 2, 3, 0]), 0)

        result = tf.abs(
            tf.signal.ifftshift(transp_ifft2d(tf.signal.ifftshift(tf.multiply(img_ft, psf_fr), axes=[1, 2])),
                                axes=[1, 2]))

        result = tf.reduce_sum(tf.multiply(Map, result), axis=4)
        result = result / tf.math.reduce_max(result)

        # ............. reconstruction step--------------

        psf_ifr = tf.signal.fftshift(
            transp_ifft2d(tf.signal.ifftshift(tf.expand_dims(tf.transpose(psfs, [1, 2, 3, 0]), 0), axes=[1, 2])),
            axes=[1, 2])
        abs_pfr = tf.cast(tf.pow(tf.abs(psf_fr), 2) + self.parameter, dtype=psf_ifr.dtype)

        result_fr = tf.expand_dims(
            tf.signal.fftshift(transp_fft2d(tf.signal.ifftshift(result, axes=[1, 2])), axes=[1, 2]), -1)

        recov = tf.abs(tf.signal.fftshift(
            transp_ifft2d(tf.signal.ifftshift(tf.math.divide(tf.multiply(psf_ifr, result_fr), abs_pfr), axes=[1, 2])),
            axes=[1, 2]))

        recov = tf.reshape(recov, [self.batch, self.N, self.N, self.input_dim[2]*self.output_dim[2]])

        return recov/tf.math.reduce_max(recov)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class Single_Pixel_Layer(Layer):

    def __init__(self, output_dim=(28, 28, 1), input_dim=(28, 28, 1), compression=0.2, parm1=0., parm2=0.,
                 type_code='Binary_1', trans=0.5, type_reg='Reg_Binary_1_1', kern=28, p=1, q=1, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.compression = compression
        self.parms1 = parm1
        self.parms2 = parm2
        self.type_code = type_code
        self.type_reg = type_reg
        self.trans = trans
        self.kern = kern
        self.p = p
        self.q = q

        if self.type_code == 'Binary_0':
            if self.type_reg == 'Physical':
                self.my_regularizer = Reg_Binary_0_1(parm1)
            if self.type_reg == 'L2_L1':
                self.my_regularizer = Reg_Binary_rank(parm1)


        if self.type_code == 'Binary_1':
            if self.type_reg == 'Physical':
                self.my_regularizer = Reg_Binary_1_1(parm1)

            if self.type_reg == 'Transmittance':
                self.my_regularizer = Reg_Binary_1_1_transm(parameter=self.parms1,
                                                            parameter2=self.parms2, tram=((self.trans)-(1-self.trans)))

            if self.type_reg == 'Family':
                self.my_regularizer = Reg_Binary_1_family(parameter=self.parms1,p=self.p,q=self.q)

            if self.type_reg =='Correlation':
                self.my_regularizer = Reg_Binary_1_1_correlation(parameter=self.parms1,parameter2=self.parms2)




        if self.type_reg == 'family_zeros':
            #self.my_regularizer = Reg_Binary_0_1(parm1)
            self.my_regularizer = Reg_Binary_family(parm1,2,2)

        super(Single_Pixel_Layer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'compression': self.compression,
            'parms1': self.parms1,
            'parms2': self.parms2,
            'type_code': self.type_code,
            'trans': self.trans,
            'kern': self.kern,
            'my_regularizer': self.my_regularizer,
            'type_reg': self.type_reg,
            'p': self.p,
            'q': self.q})
        return config

    def build(self, input_shape):

        M = round((self.input_dim[0] * self.input_dim[1] * self.input_dim[2]) * self.compression)
        # ---------------------------  Types of codes initial values ---------------------------------------

        # --------Binary -1 1 -------
        if self.type_code == 'Binary_1':
            H_init = np.random.normal(0, 1, (1, self.kern, self.kern, M)) / np.sqrt(
                self.kern * self.kern)
            H_init = tf.constant_initializer(H_init)

            self.H = self.add_weight(name='H', shape=(1, self.kern, self.kern, M),
                                     initializer=H_init, trainable=True,
                                     regularizer=self.my_regularizer)

        # --------Binary 0,1  -------
        if self.type_code == 'Binary_0':
            #H_init = np.random.rand(1, self.kern, self.kern, M)
            H_init = np.random.normal(0, 1, (1, self.kern, self.kern, M)) / np.sqrt(
                self.kern * self.kern)
            H_init = tf.constant_initializer(H_init)

            self.H = self.add_weight(name='H', shape=(1, self.kern, self.kern, M),
                                     initializer=H_init, trainable=True,
                                     regularizer=self.my_regularizer)

        # --------Gray scale -------
        if self.type_code == 'Gray_scale':

            H_init = np.random.normal(0, 1, (1, self.kern, self.kern, M)) / np.sqrt(
                self.kern * self.kern)
            H_init = tf.constant_initializer(H_init)

            if self.type_reg == 'Physical':
                self.H = self.add_weight(name='H', shape=(1, self.kern, self.kern, M),
                                         initializer=H_init, trainable=True,
                                         constraint=MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0))

        super(Single_Pixel_Layer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # Customizing the CA
        H = kronecker_product(tf.ones((int(self.input_dim[0] / self.kern), int(self.input_dim[1] / self.kern))), self.H)

        # direct sensing model in tensor verion
        y = tf.reduce_sum(tf.reduce_sum(tf.multiply(inputs, H), axis=2), axis=1)

        #######self.my_regularizer.parameter.assign(10)

        # transpose in tensor version
        y = tf.expand_dims(tf.expand_dims(y, -1), -1)
        H_t = tf.transpose(H, [0, 3, 1, 2])
        x = tf.reduce_sum(tf.multiply(y, H_t), axis=1)
        x = tf.expand_dims(x, -1)
        x = x/tf.math.reduce_max(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



class Single_Pixel_Layer_trunc(Layer):

    def __init__(self, output_dim=(28, 28, 1), input_dim=(28, 28, 1), compression=0.2, parm1=0., parm2=0.,
                 type_code='Binary_1', trans=0.5, type_reg='Reg_Binary_1_1', kern=28, p=1, q=1, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.compression = compression
        self.parms1 = parm1
        self.parms2 = parm2
        self.type_code = type_code
        self.type_reg = type_reg
        self.trans = trans
        self.kern = kern
        self.p = p
        self.q = q

        if self.type_code == 'Binary_0':
            if self.type_reg == 'Physical':
                self.my_regularizer = Reg_Binary_0_1(parm1)
            if self.type_reg == 'L2_L1':
                self.my_regularizer = Reg_Binary_rank(parm1)




        if self.type_code == 'Binary_1':
            if self.type_reg == 'Physical':
                self.my_regularizer = Reg_Binary_1_1(parm1)

            if self.type_reg == 'Transmittance':
                self.my_regularizer = Reg_Binary_1_1_transm(parameter=self.parms1,
                                                            parameter2=self.parms2, tram=((self.trans)-(1-self.trans)))

            if self.type_reg == 'Family':
                self.my_regularizer = Reg_Binary_1_family(parameter=self.parms1,p=self.p,q=self.q)




        if self.type_reg == 'family_zeros':
            #self.my_regularizer = Reg_Binary_0_1(parm1)
            self.my_regularizer = Reg_Binary_family(parm1,2,2)

        super(Single_Pixel_Layer_trunc, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'compression': self.compression,
            'parms1': self.parms1,
            'parms2': self.parms2,
            'type_code': self.type_code,
            'trans': self.trans,
            'kern': self.kern,
            'my_regularizer': self.my_regularizer,
            'type_reg': self.type_reg,
            'p': self.p,
            'q': self.q})
        return config

    def build(self, input_shape):

        M = round((self.input_dim[0] * self.input_dim[1] * self.input_dim[2]) * self.compression)
        # ---------------------------  Types of codes initial values ---------------------------------------

        # --------Binary -1 1 -------
        if self.type_code == 'Binary_1':
            H_init = np.random.normal(0, 1, (1, self.kern, self.kern, M)) / np.sqrt(
                self.kern * self.kern)
            H_init = tf.constant_initializer(H_init)

            self.H = self.add_weight(name='H', shape=(1, self.kern, self.kern, M),
                                     initializer=H_init, trainable=True,
                                     regularizer=self.my_regularizer)

        # --------Binary 0,1  -------
        if self.type_code == 'Binary_0':
            #H_init = np.random.rand(1, self.kern, self.kern, M)
            H_init = np.random.normal(0, 1, (1, self.kern, self.kern, M)) / np.sqrt(
                self.kern * self.kern)
            H_init = tf.constant_initializer(H_init)

            self.H = self.add_weight(name='H', shape=(1, self.kern, self.kern, M),
                                     initializer=H_init, trainable=True,
                                     regularizer=self.my_regularizer)

        # --------Gray scale -------
        if self.type_code == 'Gray_scale':

            H_init = np.random.normal(0, 1, (1, self.kern, self.kern, M)) / np.sqrt(
                self.kern * self.kern)
            H_init = tf.constant_initializer(H_init)

            if self.type_reg == 'Physical':
                self.H = self.add_weight(name='H', shape=(1, self.kern, self.kern, M),
                                         initializer=H_init, trainable=True,
                                         constraint=MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0))

        super(Single_Pixel_Layer_trunc, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # Customizing the CA
        H = kronecker_product(tf.ones((int(self.input_dim[0] / self.kern), int(self.input_dim[1] / self.kern))), self.H)

        #temp = tf.math.sqrt(tf.reduce_sum(tf.reduce_sum(tf.square(H), axis=2), axis=1))
        #ind = tf.abs(temp) > 0.01
        #ind = tf.expand_dims(tf.expand_dims(tf.cast(ind, dtype=tf.float32),axis=1),axis=1)
        #H = tf.math.multiply(H,ind)


        # direct sensing model in tensor verion
        y = tf.reduce_sum(tf.reduce_sum(tf.multiply(inputs, H), axis=2), axis=1)

        #######self.my_regularizer.parameter.assign(10)

        # transpose in tensor version
        y = tf.expand_dims(tf.expand_dims(y, -1), -1)
        H_t = tf.transpose(H, [0, 3, 1, 2])
        x = tf.reduce_sum(tf.multiply(y, H_t), axis=1)
        x = tf.expand_dims(x, -1)
        x = x/tf.math.reduce_max(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

