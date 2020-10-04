import tensorflow as tf

class Reg_Binary_1_1(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter=10):
        self.parameter = tf.keras.backend.variable(parameter,name='parameter')
    def __call__(self, x):
        regularization = self.parameter*(tf.reduce_sum(tf.math.multiply(tf.math.pow(1+x,2),tf.math.pow(1-x,2))))
        return regularization

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter))}


class Reg_Binary_1_family(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter=10, p=1,q=1):
        self.parameter = tf.keras.backend.variable(parameter,name='parameter')
        self.p = p
        self.q = q
    def __call__(self, x):
        regularization = self.parameter * (tf.reduce_sum(
            tf.math.multiply(tf.math.pow(tf.math.pow(1 + x, 2), self.p), tf.math.pow(tf.math.pow(1 - x, 2), self.q))))
        return regularization

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter)),
                'p': float(tf.keras.backend.get_value(self.p)),
                'q': float(tf.keras.backend.get_value(self.q))}



class Reg_Binary_1_1_transm(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter=10, parameter2= 10, tram=0.8):
        self.parameter = tf.keras.backend.variable(parameter,name='parameter')
        self.parameter2 = tf.keras.backend.variable(parameter2, name='parameter2')
        self.T  = tram
    def __call__(self, x):
        binary = self.parameter*(tf.reduce_sum(tf.math.multiply(tf.math.pow(1+x,2),tf.math.pow(1-x,2))))
        #tramitance = self.parameter2*(tf.reduce_sum(tf.square(tf.math.divide(x,x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3])-self.T)))
        tramitance = self.parameter2 * (tf.square(tf.math.divide(tf.reduce_sum(x),x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]) - self.T))
        return binary+tramitance

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter)),
                'parameter2': float(tf.keras.backend.get_value(self.parameter2)),
                'T': float(self.T)}



class Reg_Binary_1_1_correlation(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter=10, parameter2= 10,eps=1e-8):
        self.parameter = tf.keras.backend.variable(parameter,name='parameter')
        self.parameter2 = tf.keras.backend.variable(parameter2, name='parameter2')
        self.eps  = eps

    def __call__(self, x):
        binary = self.parameter*(tf.reduce_sum(tf.math.multiply(tf.math.pow(1+x,2),tf.math.pow(1-x,2))))
        #numera = tf.reduce_sum(tf.math.reduce_prod(x,axis=3))
        #denom  = tf.reduce_prod(tf.sqrt(tf.reduce_sum(tf.math.pow(x,2),axis=[0,1,2])))

        nuev2 = self.parameter2*(tf.reduce_sum(tf.math.pow(tf.reduce_sum(x,axis=3),2)))
        tf.print(nuev2)
        nuev = self.parameter2*(tf.reduce_sum(tf.math.pow(tf.math.reduce_prod(x,axis=3)+x.shape[3],2))/(x.shape[1]*x.shape[2]*x.shape[3]))
        return binary+nuev2

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter)),
                'parameter2': float(tf.keras.backend.get_value(self.parameter2)),
                'T': float(self.T)}



class Reg_Binary_0_1(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter=10):
        self.parameter = tf.keras.backend.variable(parameter,name='parameter')
    def __call__(self, x):
        regularization = self.parameter*(tf.reduce_sum(tf.multiply(tf.square(x),tf.square(1-x))))
        return regularization

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter))}


class Reg_Real_value(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter=10):
        self.parameter = tf.keras.backend.variable(parameter,name='parameter')
    def __call__(self, x):

        tem1 = tf.square(x)
        tem2 = tf.square(0.25-x)
        tem3 = tf.square(0.5-x)
        tem4 = tf.square(0.75-x)
        tem5 = tf.square(1-x)

        regularization = self.parameter * (
            tf.reduce_sum(tf.multiply(tem1, tf.multiply(tem2, tf.multiply(tem3, tf.multiply(tem4, tem5))))))

        return regularization

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter))}



@tf.custom_gradient
def Rank(x):
    val = tf.reduce_sum(tf.math.sqrt(tf.reduce_sum(tf.reduce_sum(tf.square(x),axis=2),axis=1)))
    def grad(dy):
        temp = tf.math.sqrt(tf.reduce_sum(tf.reduce_sum(tf.square(x),axis=2),axis=1))

        ind = tf.abs(temp) > 0.0001
        ind = tf.cast(ind, dtype=tf.float32)
        temp = tf.math.multiply(temp, ind) + (1-ind)
        return tf.math.divide(x,tf.expand_dims(tf.expand_dims(temp,axis=1),axis=1))
    return val, grad

class Reg_Binary_rank(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter=10):
        self.parameter = tf.keras.backend.variable(parameter,name='parameter')
    def __call__(self, x):
        return  self.parameter*Rank(x)

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter))}


class Reg_Binary_family(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter=10,m =2, n = 2):
        self.parameter = tf.keras.backend.variable(parameter,name='parameter9 cisto')
        self.m = m
        self.n = n
    def __call__(self, x):
        rr = tf.convert_to_tensor(0,dtype=tf.float32)
        for i in range(1,self.m+1):
            for j in range(i+1,self.n+1):
                rr =  rr + (tf.reduce_sum(tf.square(tf.math.pow(x,j)-tf.math.pow(x,i))))
        return tf.cast(self.parameter *rr,dtype=tf.float32)

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter)),
                'm': float(tf.keras.backend.get_value(self.m)),
                'n': float(tf.keras.backend.get_value(self.m))}


def set_Transmitance(Tr,parm1,parm2):
    def Binary_0_1(W):
        T= tf.constant(Tr)
        parm = tf.constant(parm1)
        parm_2 = tf.constant(parm2)
        R1 = tf.reduce_sum(tf.multiply(tf.square(W),tf.square(1-W)))
        R2 = tf.reduce_sum(tf.square(tf.math.divide(W,W.shape[0]*W.shape[1]*W.shape[2]*W.shape[3])-T))
        return tf.multiply(parm,tf.cast(R1, dtype=tf.float32)) + tf.multiply(parm_2,tf.cast(R2, dtype=tf.float32))
    return Binary_0_1


def set_Binary_0_1(parm1):
    def Binary_0_1(W):
        parm = tf.constant(parm1)
        R1 = tf.reduce_sum(tf.multiply(tf.square(W),tf.square(1-W)))
        return tf.multiply(parm,tf.cast(R1, dtype=tf.float32))
    return Binary_0_1








