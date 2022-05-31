import tensorflow_probability as tfp
import tensorflow as tf

def variance_optimal(wealth=None,p0=None):
    loss = tf.math.reduce_mean((wealth+p0)**2) 
    return loss