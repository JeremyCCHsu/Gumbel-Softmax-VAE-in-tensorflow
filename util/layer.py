from tensorflow.contrib import slim
import numpy as np
import tensorflow as tf

EPSILON = 1e-6

def GumbelSampleLayer(y_mu):
    ''' Create Gumbel(0, 1) variable from Uniform[0, 1] '''
    u = tf.random_uniform(
        minval=0.0,
        maxval=1.0,
        shape=tf.shape(y_mu))
    g = -tf.log(-tf.log(u))
    return y_mu + g


def normalize_to_unit_sum(x, EPS=1e-10):
    ''' Along the last dim '''
    EPS = tf.constant(EPS, dtype=tf.float32)
    x = x + EPS
    x_sum = tf.reduce_sum(x, -1, keep_dims=True)
    x = tf.divide(x, x_sum)
    return x


def GumbelSoftmaxLogDensity(y, p, tau):
    # EPS = tf.constant(1e-10)
    k = tf.shape(y)[-1]
    k = tf.cast(k, tf.float32)
    # y = y + EPS
    # y = tf.divide(y, tf.reduce_sum(y, -1, keep_dims=True))
    y = normalize_to_unit_sum(y)
    sum_p_over_y = tf.reduce_sum(tf.divide(p, tf.pow(y, tau)), -1)
    logp = tf.lgamma(k)
    logp = logp + (k - 1) * tf.log(tau)
    logp = logp - k * tf.log(sum_p_over_y)
    logp = logp + sum_p_over_y
    return logp


def lrelu(x, leak=0.02, name="lrelu"):
    ''' Leaky ReLU '''
    return tf.maximum(x, leak*x, name=name)

def dense(
    x,
    output_dim,
    scope=None,
    reuse=False,
    normalized=False,
    params=None,
    activation_fn=lrelu):
    if params is not None:
        w, b = params
    else:
        with tf.variable_scope(scope or 'linear') as scope:
            if reuse:
                scope.reuse_variables()
            w = tf.get_variable(
                name='w',
                shape=[x.get_shape()[1], output_dim],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                name='b',
                shape=[output_dim],
                initializer=tf.constant_initializer(0.0))
    if normalized:
        # J: Normalized initials?
        # w_n = tf.nn.l2_normalize(w, dim=-1, name='w_n')
        w_n = w / tf.reduce_sum(tf.square(w), 1, keep_dims=True)
    else:
        w_n = w
    return activation_fn(tf.add(tf.matmul(x, w_n), b))

def conv2d(
    x,
    filter_shape,
    strides,
    output_dim,
    scope=None,
    reuse=False,
    padding='SAME',
    # normalized=False,
    params=None,
    activation_fn=lrelu):
    '''
    filter_shape: [h_filter_size, w_filter_size, output_dim]
    '''
    if params is not None:
        w, b = params
    else:
        with tf.variable_scope(scope or 'conv') as scope:
            if reuse:
                scope.reuse_variables()
            height, width = filter_shape
            w = tf.get_variable(
                name='w',
                shape=[height, width, x.get_shape()[-1], output_dim],
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable(
                name='b',
                shape=[output_dim],
                initializer=tf.constant_initializer(0.0))
    y = tf.add(tf.nn.conv2d(x, w, strides, padding), b)
    return activation_fn(y)

# def batch_norm(x, mean, variance, offset, scale, variance_epsilon, name=None):
    

# tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None)


def test_fc_cnn():
    x = tf.placeholder(name='x', shape=[50, 32, 32, 1], dtype=tf.float32)
    c1 = conv2d(x, [5, 5], [1, 2, 2, 1], 16, scope='conv1')
    c2 = conv2d(c1, [5, 5], [1, 2, 2, 1], 64, scope='conv2')
    f0 = slim.flatten(c2)
    f1 = dense(f0, 100, scope='dense1')
    f2 = dense(f1, 10, scope='dense2')
    return f2


# [TODO] Need to test BN
# d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

def discriminator(x):
    with tf.variable_scope('Discriminator'):
        c1 = conv2d(x, [5, 5], [1, 2, 2, 1], 16, scope='conv1')
        c2 = conv2d(c1, [5, 5], [1, 2, 2, 1], 64, scope='conv2')
        f0 = slim.flatten(c2)
        f1 = dense(f0, 100, scope='dense1')
        f2 = dense(f1, 10, scope='dense2')
    return f2

def discriminator_from_params(x, params):
    with tf.variable_scope('Discriminator'):
        c1 = conv2d(x, [5, 5], [1, 2, 2, 1], 16, scope='conv1', params=params[0:2])
        c2 = conv2d(c1, [5, 5], [1, 2, 2, 1], 64, scope='conv2', params=params[2:4])
        f0 = slim.flatten(c2)
        f1 = dense(f0, 100, scope='dense1', params=params[4:6])
        f2 = dense(f1, 10, scope='dense2', params=params[6:8])
    return f2


    # hid = dense(x, n_hid, scope='l1', params=params[:2], normalized=True)
    # hid = tf.nn.relu(hid)
    # #hid = tf.tanh(hid)
    # hid = dense(hid, n_hid, scope='l2', params=params[2:4], normalized=True)
    # hid = tf.nn.relu(hid)
    # #hid = tf.tanh(hid)
    # out = tf.nn.sigmoid(dense(hid, 1, scope='d_out', params=params[4:]))
    # # 


def GaussianLogDensity(x, mu, log_var, name='GaussianLogDensity'):
    c = np.log(2 * np.pi)
    var = tf.exp(log_var)
    x_mu2 = tf.square(tf.sub(x, mu))   # [Issue] not sure the dim works or not?
    x_mu2_over_var = tf.div(x_mu2, var + EPSILON)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    log_prob = tf.reduce_sum(log_prob, -1, name=name)   # keep_dims=True,
    return log_prob


def GaussianKLD(mu1, lv1, mu2, lv2):
    ''' Kullback-Leibler divergence of two Gaussians
        *Assuming that each dimension is independent
        mu: mean
        lv: log variance
        Equation: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    '''
    with tf.name_scope('GaussianKLD'):
        v1 = tf.exp(lv1)
        v2 = tf.exp(lv2)
        mu_diff_sq = tf.square(mu1 - mu2)
        dimwise_kld = .5 * (
            (lv2 - lv1) + tf.div(v1 + mu_diff_sq, v2 + EPSILON) - 1.)
        return tf.reduce_sum(dimwise_kld, -1)

# Verification by CMU's implementation
# http://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/divergence.py
# def gau_kl(pm, pv, qm, qv):
#     """
#     Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
#     Also computes KL divergence from a single Gaussian pm,pv to a set
#     of Gaussians qm,qv.
#     Diagonal covariances are assumed.  Divergence is expressed in nats.
#     """
#     if (len(qm.shape) == 2):
#         axis = 1
#     else:
#         axis = 0
#     # Determinants of diagonal covariances pv, qv
#     dpv = pv.prod()
#     dqv = qv.prod(axis)
#     # Inverse of diagonal covariance qv
#     iqv = 1./qv
#     # Difference between means pm, qm
#     diff = qm - pm
#     return (0.5 *
#             (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
#              + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
#              + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
#              - len(pm)))                     # - N

def GaussianSampleLayer(z_mu, z_lv, name='GaussianSampleLayer'):
    with tf.name_scope(name):
        eps = tf.random_normal(tf.shape(z_mu))
        std = tf.sqrt(tf.exp(z_lv))
        return tf.add(z_mu, tf.multiply(eps, std))

