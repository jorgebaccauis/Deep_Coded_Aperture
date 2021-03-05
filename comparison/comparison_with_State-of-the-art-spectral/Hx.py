import tensorflow as tf
def Forward_D_CASSI(inputs, CA):
    M, L = inputs.shape[2:4]
    H = tf.concat([CA[:, :, i:M+i, :] for i in range(L)], -1)
    H = tf.expand_dims(H,0)
    # def multH(x): return tf.multiply(H, x)
    # y = tf.map_fn(multH, inputs)
    y = tf.multiply(H, inputs)
    y = tf.reduce_sum(y, -1)
    y = tf.transpose(y, perm=[1, 2, 3, 0])
    return y

def Transpose_D_CASSI(y, CA):
    M = CA.shape[1]
    C = CA.shape[2] - M + 1
    H = tf.concat([CA[:, :, i:M+i, :] for i in range(C)], -1)
    # Correction with L1 normalization
    # H = tf.divide(H, tf.add(tf.reduce_sum(H, -1, keepdims=True), 1e-8))
    H = tf.expand_dims(H,0)
    y = tf.tile(y, [1, 1, 1, C])
    # def multH(x): return tf.multiply(H, x)
    # output = tf.map_fn(multH, y)
    output = tf.multiply(H, y)
    output = tf.reduce_sum(output, 0)
    
    return output / tf.math.reduce_max(output)