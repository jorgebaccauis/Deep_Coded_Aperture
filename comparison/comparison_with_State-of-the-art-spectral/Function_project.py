import tensorflow as tf
import numpy as np
import copy
from scipy import interpolate


def addGaussianNoise(y, SNR):
    sigma = np.sum(np.power(y, 2)) / (np.product(y.shape) * 10 ** (SNR / 10));
    w = np.random.normal(0, np.sqrt(sigma), size=y.shape);
    return y + w

def measurements_single_pixel(x_test, CA, noise,type):

    if(type=='Binary_0' or type=='Gray_scale'):
        y = np.sum(np.sum(np.multiply(x_test, CA), axis=2), axis=1)
        y = addGaussianNoise(y, noise)

    if (type == 'Binary_1'):
        CA_pos = copy.deepcopy(CA)
        CA_neg = copy.deepcopy(CA)
        CA_pos[CA_pos < 0] = 0
        CA_neg[CA_neg > 0] = 0

        y_pos = np.sum(np.sum(np.multiply(x_test, CA_pos), axis=2), axis=1)
        y_pos = addGaussianNoise(y_pos, noise)

        y_neg = np.sum(np.sum(np.multiply(x_test, CA_neg), axis=2), axis=1)
        y_neg = addGaussianNoise(y_neg, noise)

        y = y_pos+y_neg

    # transpose in tensor version
    y = np.expand_dims(np.expand_dims(y, -1), -1)
    H_t = np.transpose(CA, [0, 3, 1, 2])
    x = np.sum(np.multiply(y, H_t), axis=1)
    x = np.expand_dims(x, -1)
    return x / np.max(x)


def CASSI_transponse(Y,CA):
    X = None
    for i in range(16):
        Tempo = np.roll(Y, shift=-i, axis=2)
        if X is not None:
            X = np.concatenate([X, np.expand_dims(Tempo[:, :, 0:256], -1)], axis=4)
        else:
            X = np.expand_dims(Tempo[:, :, 0:256], -1)

    X = np.transpose(X, [0, 1, 2, 4, 3])
    X = np.multiply(CA, X)
    X = np.sum(X, 4)

    X = X / np.max(X)
    return  X

def measurements_cassi_mostrar(inputs, CA, noise,type):

    L = inputs.shape[3]
    M = inputs.shape[2]

    if(type=='Binary_0' or type=='Gray_scale'):
        Aux1 = np.multiply(CA, np.expand_dims(inputs, -1))
        Aux1 = np.pad(Aux1, [[0, 0], [0, 0], [0, L - 1], [0, 0], [0, 0]])
        Y = None
        for i in range(L):
            Tempo = np.roll(Aux1, shift=i, axis=2)
            if Y is not None:
                Y = np.concatenate([Y, np.expand_dims(Tempo[:, :, :, i], -1)], axis=4)
            else:
                Y = np.expand_dims(Tempo[:, :, :, i], -1)
        Y = np.sum(Y, 4)
        #Y = addGaussianNoise(Y, noise)


    if (type == 'Binary_1'):
        CA_pos = copy.deepcopy(CA)
        CA_neg = copy.deepcopy(CA)
        CA_pos[CA_pos < 0] = 0
        CA_neg[CA_neg > 0] = 0

        Aux_pos = np.multiply(CA_pos, np.expand_dims(inputs, -1))
        Aux_pos = np.pad(Aux_pos, [[0, 0], [0, 0], [0, L - 1], [0, 0], [0, 0]])
        Y_pos = None
        for i in range(L):
            Tempo = np.roll(Aux_pos, shift=i, axis=2)
            if Y_pos is not None:
                Y_pos = np.concatenate([Y_pos, np.expand_dims(Tempo[:, :, :, i], -1)], axis=4)
            else:
                Y_pos = np.expand_dims(Tempo[:, :, :, i], -1)
        Y_pos = np.sum(Y_pos, 4)
        #Y_pos = addGaussianNoise(Y_pos, noise)

        Aux_neg = np.multiply(CA_neg, np.expand_dims(inputs, -1))
        Aux_neg = np.pad(Aux_neg, [[0, 0], [0, 0], [0, L - 1], [0, 0], [0, 0]])
        Y_neg = None
        for i in range(L):
            Tempo = np.roll(Aux_neg, shift=i, axis=2)
            if Y_neg is not None:
                Y_neg = np.concatenate([Y_neg, np.expand_dims(Tempo[:, :, :, i], -1)], axis=4)
            else:
                Y_neg = np.expand_dims(Tempo[:, :, :, i], -1)
        Y_neg = np.sum(Y_neg, 4)
        Y_neg = addGaussianNoise(Y_neg, noise)

        Y = Y_pos + Y_neg

    # CASSI Sensing Model

    X = None
    for i in range(L):
        Tempo = np.roll(Y, shift=-i, axis=2)
        if X is not None:
            X = np.concatenate([X, np.expand_dims(Tempo[:, :, 0:M], -1)], axis=4)
        else:
            X = np.expand_dims(Tempo[:, :, 0:M], -1)

    X = np.transpose(X, [0, 1, 2, 4, 3])
    X = np.multiply(CA, X)
    X = np.sum(X, 4)

    X = X / np.max(X)
    return X,Y

def measurements_cassi(inputs, CA, noise,type):

    L = inputs.shape[3]
    M = inputs.shape[2]

    if(type=='Binary_0' or type=='Gray_scale'):
        Aux1 = np.multiply(CA, np.expand_dims(inputs, -1))
        Aux1 = np.pad(Aux1, [[0, 0], [0, 0], [0, L - 1], [0, 0], [0, 0]])
        Y = None
        for i in range(L):
            Tempo = np.roll(Aux1, shift=i, axis=2)
            if Y is not None:
                Y = np.concatenate([Y, np.expand_dims(Tempo[:, :, :, i], -1)], axis=4)
            else:
                Y = np.expand_dims(Tempo[:, :, :, i], -1)
        Y = np.sum(Y, 4)
        #Y = addGaussianNoise(Y, noise)


    if (type == 'Binary_1'):
        CA_pos = copy.deepcopy(CA)
        CA_neg = copy.deepcopy(CA)
        CA_pos[CA_pos < 0] = 0
        CA_neg[CA_neg > 0] = 0

        Aux_pos = np.multiply(CA_pos, np.expand_dims(inputs, -1))
        Aux_pos = np.pad(Aux_pos, [[0, 0], [0, 0], [0, L - 1], [0, 0], [0, 0]])
        Y_pos = None
        for i in range(L):
            Tempo = np.roll(Aux_pos, shift=i, axis=2)
            if Y_pos is not None:
                Y_pos = np.concatenate([Y_pos, np.expand_dims(Tempo[:, :, :, i], -1)], axis=4)
            else:
                Y_pos = np.expand_dims(Tempo[:, :, :, i], -1)
        Y_pos = np.sum(Y_pos, 4)
        #Y_pos = addGaussianNoise(Y_pos, noise)

        Aux_neg = np.multiply(CA_neg, np.expand_dims(inputs, -1))
        Aux_neg = np.pad(Aux_neg, [[0, 0], [0, 0], [0, L - 1], [0, 0], [0, 0]])
        Y_neg = None
        for i in range(L):
            Tempo = np.roll(Aux_neg, shift=i, axis=2)
            if Y_neg is not None:
                Y_neg = np.concatenate([Y_neg, np.expand_dims(Tempo[:, :, :, i], -1)], axis=4)
            else:
                Y_neg = np.expand_dims(Tempo[:, :, :, i], -1)
        Y_neg = np.sum(Y_neg, 4)
        Y_neg = addGaussianNoise(Y_neg, noise)

        Y = Y_pos + Y_neg

    # CASSI Sensing Model

    X = None
    for i in range(L):
        Tempo = np.roll(Y, shift=-i, axis=2)
        if X is not None:
            X = np.concatenate([X, np.expand_dims(Tempo[:, :, 0:M], -1)], axis=4)
        else:
            X = np.expand_dims(Tempo[:, :, 0:M], -1)

    X = np.transpose(X, [0, 1, 2, 4, 3])
    X = np.multiply(CA, X)
    X = np.sum(X, 4)

    X = X / np.max(X)
    return X


def measurments_Depth_CA (inputs,H,noise=50,batch=1,N = 512, z0 = 2.5, zi = 50e-3,radii = 2e-3, px = 6.22e-6,parameter=0.2):

    img = inputs[0]
    Map = inputs[1]


    f = 1 / (1 / zi + 1 / z0)
    R = f * deta(550e-9 * 1e6)

    L_len = 2 * radii * 2
    L_sen = px * N

    lamb = np.asarray([460, 550, 640.]) * 1e-9
    flmb = R / deta(lamb * 1e6)
    z = np.sort(-3 * (np.log(np.linspace(0.9, 11, 15))) + 8)

    du = L_len / N;
    u = np.arange(-L_len / 2, L_len / 2, du)
    [X, Y] = np.meshgrid(u, u)
    XY = (X * X + Y * Y)
    [r, thetha] = cart2pol(X, Y)
    rad = r <= radii

    fx1 = np.arange(-1 / (2 * du), 1 / (2 * du), 1 / L_len)
    fx1 = np.fft.fftshift(fx1)
    [FX1, FY1] = np.meshgrid(fx1, fx1)
    FF = FX1 * FX1 + FY1 * FY1

    dx2 = L_sen / N
    x2 = np.arange(-L_sen / 2, L_sen / 2, dx2)
    [X2, Y2] = np.meshgrid(x2, x2)
    XY2 = X2 * X2 + Y2 * Y2

    CA = kronecker_product_depth(H, tf.ones((16, 16)))
    CA = tf.cast(tf.squeeze(CA), dtype=tf.complex64)
    lt = 0
    psfs = None
    for dis in z:
        t_psf = None
        for band in lamb:

            k = (2 * np.pi) / (band)

            # constant = tf.expand_dims(tf.expand_dims(-(k / (2 * flmb[lt])),0),0)
            # t= tf.multiply(tf.multiply(constant,XY),CA)

            t = tf.multiply(compl_exp_tf(-(k / (2 * flmb[lt])) * (XY)), CA)
            focus = compl_exp_tf((k / (2 * dis)) * (XY))

            ph = tf.multiply(rad, tf.multiply(t, focus))

            vu = tf.multiply(ph, compl_exp_tf(
                (np.pi / (band * zi * L_len) * (L_len - L_sen)) * (XY)))
            vu = tf.signal.fft2d(tf.signal.fftshift(vu))

            vu = tf.multiply(vu, compl_exp_tf(-(np.pi * band * zi * L_len / L_sen) * (FF)))

            vu = tf.signal.ifftshift(tf.signal.ifft2d(vu))

            vu = (L_sen / L_len) * tf.multiply(vu, compl_exp_tf(
                -(np.pi / (band * zi * L_sen) * (L_len - L_sen)) * (XY2)))
            psf = tf.pow(tf.abs(vu * ((du * du) / (dx2 * dx2))), 2)
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


    Map = tf.expand_dims(Map, 3)
    img_ft = tf.signal.fftshift(transp_fft2d(tf.signal.ifftshift(img, axes=[1, 2])), axes=[1, 2])
    psf_fr = tf.signal.fftshift(transp_fft2d(tf.signal.ifftshift(psfs, axes=[1, 2])), axes=[1, 2])

    img_ft = tf.expand_dims(img_ft, -1)
    psf_fr = tf.expand_dims(tf.transpose(psf_fr, [1, 2, 3, 0]), 0)

    result = tf.abs(
        tf.signal.ifftshift(transp_ifft2d(tf.signal.ifftshift(tf.multiply(img_ft, psf_fr), axes=[1, 2])),
                            axes=[1, 2]))

    pp = tf.multiply(Map, result)
    result = tf.reduce_sum(tf.multiply(Map, result), axis=4)
    result = result / tf.math.reduce_max(result)

    result = addGaussianNoise(result, noise)

    # ............. reconstruction step--------------

    psf_ifr = tf.signal.fftshift(
        transp_ifft2d(tf.signal.ifftshift(tf.expand_dims(tf.transpose(psfs, [1, 2, 3, 0]), 0), axes=[1, 2])),
        axes=[1, 2])
    abs_pfr = tf.cast(tf.pow(tf.abs(psf_fr), 2) + parameter, dtype=psf_ifr.dtype)

    result_fr = tf.expand_dims(
        tf.signal.fftshift(transp_fft2d(tf.signal.ifftshift(result, axes=[1, 2])), axes=[1, 2]), -1)

    recov = tf.abs(tf.signal.fftshift(
        transp_ifft2d(tf.signal.ifftshift(tf.math.divide(tf.multiply(psf_ifr, result_fr), abs_pfr), axes=[1, 2])),
        axes=[1, 2]))

    recov = tf.reshape(recov, [batch, N, N, 3 * 15])

    return recov / tf.math.reduce_max(recov), result,ph,pp

def kronecker_product_color(mat1, mat2):
    """Computes the Kronecker product two matrices."""
    m1, n1 = mat1.get_shape().as_list()
    mat1_rsh = tf.reshape(mat1, [1, m1, 1, n1, 1, 1,1])
    ini, m2, n2, l1, sh = mat2.get_shape().as_list()
    mat2_rsh = tf.reshape(mat2, [1, 1, m2, 1, n2, l1,sh])
    return tf.reshape(tf.multiply(mat1_rsh, mat2_rsh), [1, m1 * m2, n1 * n2, l1,sh])

def kronecker_product(mat1, mat2):
    """Computes the Kronecker product two matrices."""
    m1, n1 = mat1.get_shape().as_list()
    mat1_rsh = tf.reshape(mat1, [1, m1, 1, n1, 1, 1])
    ini, m2, n2, l1 = mat2.get_shape().as_list()
    mat2_rsh = tf.reshape(mat2, [1, 1, m2, 1, n2, l1])
    return tf.reshape(tf.multiply(mat1_rsh, mat2_rsh), [1, m1 * m2, n1 * n2, l1])

def kronecker_product_np(mat1, mat2):
    """Computes the Kronecker product two matrices."""
    m1, n1 = mat1.shape
    mat1_rsh = np.reshape(mat1, [1, m1, 1, n1, 1, 1])
    ini, m2, n2, l1 = mat2.shape
    mat2_rsh = np.reshape(mat2, [1, 1, m2, 1, n2, l1])
    return np.reshape(np.multiply(mat1_rsh, mat2_rsh), [1, m1 * m2, n1 * n2, l1])

def get_color_bases(wls):
    SG = [0.0875,0.1098,0.1157,0.1245,0.1379,0.1561,0.1840,0.2458,0.3101,0.3384,0.3917\
        ,0.5000,0.5732,0.6547,0.6627,0.624,0.5719,0.5157,0.4310,0.3470,0.2670,0.1760\
        ,0.1170,0.0874,0.0754,0.0674,0.0667,0.0694,0.0567,0.0360,0.0213]   # green color

    SB = [0.2340,0.2885,0.4613,0.5091,0.5558,0.5740,0.6120,0.6066,0.5759,0.4997\
        ,0.4000,0.3000,0.2070,0.1360,0.0921,0.0637,0.0360,0.0205,0.0130,0.0110\
        ,0.0080,0.0060,0.0062,0.0084,0.0101,0.0121,0.0180,0.0215,0.0164,0.0085\
        ,0.0050]                                                           # blue color

    SR = [0.1020,0.1020,0.0790,0.0590,0.0460,0.0360,0.0297,0.0293,0.0310,0.03230\
          ,0.0317,0.0367,0.0483,0.0667,0.0580,0.0346,0.0263,0.0487,0.1716,0.4342\
          ,0.5736,0.5839,0.5679,0.5438,0.5318,0.5010,0.4810,0.4249,0.2979,0.1362\
          ,0.0651]                                                         # red color

    SC = [0.1895,0.2118,0.1947,0.1835,0.1839,0.1921,0.2137,0.2751,0.3411,0.3707\
        ,0.4234,0.5367,0.6215,0.7214,0.7207,0.6586,0.5982,0.5644,0.6026,0.7812\
        ,0.8406,0.7599,0.6849,0.6312,0.6072,0.5684,0.5477,0.4943,0.3546,0.1722\
        ,0.0864]                                                          # cyan color

    x_wvls = np.linspace(399e-9,701e-9,len(SG))

    fg = interpolate.interp1d(x_wvls, SG)
    fr = interpolate.interp1d(x_wvls, SR)
    fc = interpolate.interp1d(x_wvls, SC)
    fb = interpolate.interp1d(x_wvls, SB)

    fr = fr(wls)
    fg = fg(wls)
    fc = fc(wls)
    fb = fb(wls)

    fr = tf.convert_to_tensor(fr, dtype=tf.float32)
    fg = tf.convert_to_tensor(fg, dtype=tf.float32)
    fc = tf.convert_to_tensor(fc, dtype=tf.float32)
    fb = tf.convert_to_tensor(fb, dtype=tf.float32)

    fr = tf.expand_dims(tf.expand_dims(fr, 0), 0)
    fg = tf.expand_dims(tf.expand_dims(fg, 0), 0)
    fc = tf.expand_dims(tf.expand_dims(fc, 0), 0)
    fb = tf.expand_dims(tf.expand_dims(fb, 0), 0)


    return fr,fg,fc,fb




def deta(Lb):

    IdLens = np.sqrt(1 + (0.6961663 * (Lb ** 2) / ( (Lb ** 2)- 0.0684043 ** 2) + 0.4079426 * (Lb ** 2) / (
            (Lb ** 2) - 0.1162414 ** 2) + 0.8974794 *  (Lb ** 2) / ( (Lb ** 2)- 9.896161 ** 2)));

    #IdLens = 1.5375 + 0.00829045 * (Lb ** -2) - 0.000211046 * (Lb ** -4)
    IdAir = 1 + 0.05792105 / (238.0185 - Lb ** -2) + 0.00167917 / (57.362 - Lb ** -2)
    val = abs(IdLens - IdAir)
    return val

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def kronecker_product_depth(mat1, mat2):
    """Computes the Kronecker product two matrices."""
    ini,m1, n1,l1 = mat1.get_shape().as_list()
    mat1_rsh = tf.reshape(mat1, [1, m1, 1, n1, 1, l1])
    m2, n2 = mat2.get_shape().as_list()
    mat2_rsh = tf.reshape(mat2, [1, 1, m2, 1, n2, 1])
    return tf.reshape(tf.multiply(mat1_rsh, mat2_rsh), [1, m1 * m2, n1 * n2, l1])

def compl_exp_tf(phase, dtype=tf.complex64, name='complex_exp'):
    """Complex exponent via euler's formula, since Cuda doesn't have a GPU kernel for that.
    Casts to *dtype*.
    """
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype),
                  1.j * tf.cast(tf.sin(phase), dtype=dtype),
                  name=name)

def transp_fft2d(a_tensor, dtype=tf.complex64):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    # Tensorflow's fft only supports complex64 dtype
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    a_tensor_transp = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_fft2d = tf.signal.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a_fft2d, [0, 2, 3, 1])
    return a_fft2d

def transp_ifft2d(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a_tensor, [0, 3, 4, 1, 2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.signal.ifft2d(a_tensor)
    # Transpose back to [batch_size, x, y, channels]
    a_ifft2d = tf.transpose(a_ifft2d_transp, [0, 3, 4, 1,2])
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d