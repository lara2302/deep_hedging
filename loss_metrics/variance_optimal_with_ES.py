import tensorflow_probability as tfp
import tensorflow as tf

def variance_optimal_with_ES(wealth=None,alpha=None,lam=None,p0=None):
    VaR = tfp.stats.percentile(wealth,100*(1-alpha))
    mask = (wealth <= VaR)
    ES = -tf.cast(tf.reduce_mean(tf.boolean_mask(wealth,mask)),wealth.dtype)
    loss = tf.math.reduce_mean((wealth+p0)**2) + lam*ES
    return loss