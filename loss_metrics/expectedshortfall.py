import tensorflow_probability as tfp
import tensorflow as tf

def ExpectedShortfall(wealth=None,alpha=None):
  VaR = tfp.stats.percentile(wealth,100*(1-alpha))
  mask = (wealth <= VaR)
  ES = -tf.cast(tf.reduce_mean(tf.boolean_mask(wealth,mask)),wealth.dtype)
  return ES