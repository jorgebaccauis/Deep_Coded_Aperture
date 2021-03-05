import tensorflow as tf
import numpy as np
from math import *
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import interpolate


def get_color_bases(wls):
    SG = np.array([0.0875, 0.1098, 0.1157, 0.1245, 0.1379, 0.1561, 0.1840, 0.2458, 0.3101, 0.3384, 0.3917,
                   0.5000, 0.5732, 0.6547, 0.6627, 0.624, 0.5719, 0.5157, 0.4310, 0.3470, 0.2670, 0.1760,
                   0.1170, 0.0874, 0.0754, 0.0674, 0.0667, 0.0694, 0.0567, 0.0360, 0.0213])

    SB = np.array([0.2340, 0.2885, 0.4613, 0.5091, 0.5558, 0.5740, 0.6120, 0.6066, 0.5759, 0.4997
                      , 0.4000, 0.3000, 0.2070, 0.1360, 0.0921, 0.0637, 0.0360, 0.0205, 0.0130, 0.0110
                      , 0.0080, 0.0060, 0.0062, 0.0084, 0.0101, 0.0121, 0.0180, 0.0215, 0.0164, 0.0085, 0.0050])

    SR = np.array([0.1020, 0.1020, 0.0790, 0.0590, 0.0460, 0.0360, 0.0297, 0.0293, 0.0310, 0.03230
                      , 0.0317, 0.0367, 0.0483, 0.0667, 0.0580, 0.0346, 0.0263, 0.0487, 0.1716, 0.4342
                      , 0.5736, 0.5839, 0.5679, 0.5438, 0.5318, 0.5010, 0.4810, 0.4249, 0.2979, 0.1362
                      , 0.0651])

    SC = np.array([0.1895, 0.2118, 0.1947, 0.1835, 0.1839, 0.1921, 0.2137, 0.2751, 0.3411, 0.3707
                      , 0.4234, 0.5367, 0.6215, 0.7214, 0.7207, 0.6586, 0.5982, 0.5644, 0.6026, 0.7812
                      , 0.8406, 0.7599, 0.6849, 0.6312, 0.6072, 0.5684, 0.5477, 0.4943, 0.3546, 0.1722
                      , 0.0864])

    x_wvls = np.linspace(400e-9, 2500e-9, SG.shape[0])
    fg = interpolate.interp1d(x_wvls, SG)
    fr = interpolate.interp1d(x_wvls, SB)
    fc = interpolate.interp1d(x_wvls, SR)
    fb = interpolate.interp1d(x_wvls, SC)

    fgv = tf.reshape(fg(wls), shape=(1, 1, wls.shape[0]))
    frv = tf.reshape(fr(wls), shape=(1, 1, wls.shape[0]))
    fcv = tf.reshape(fc(wls), shape=(1, 1, wls.shape[0]))
    fbv = tf.reshape(fb(wls), shape=(1, 1, wls.shape[0]))

    return fgv, frv, fcv, fbv

def Color_Mask(M = 400, N = 400, L= 24, w_in = 400, w_fin = 2500):

    wls = np.linspace(w_in, w_fin, L) * 1e-9
    [fr, fg, fc, fb] = get_color_bases(wls)

    wr = np.random.rand(M, N)
    wg = np.random.rand(M, N)
    wb = np.random.rand(M, N)

    wg = np.multiply(wg, (1 - wr))
    wb = np.multiply(wb, (1 - wr - wg))
    wc = 1 - wr - wg - wb

    wr = np.expand_dims(wr, -1)
    wg = np.expand_dims(wg, -1)
    wb = np.expand_dims(wb, -1)
    wc = np.expand_dims(wc, -1)

    filters = np.multiply(wr, fr) + np.multiply(wg, fg) + np.multiply(wb, fb) + np.multiply(wc, fc)
    return  filters