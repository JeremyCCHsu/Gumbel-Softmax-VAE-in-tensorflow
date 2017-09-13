import pdb
from tensorflow.contrib import slim
import tensorflow as tf
from util.layer import GaussianLogDensity, GaussianKLD, \
    GaussianSampleLayer, lrelu, GumbelSampleLayer, GumbelSoftmaxLogDensity

EPS = tf.constant(1e-10)

class CVAE(object):
    '''
      VC-GAN
    = CVAE-CGAN
    = Convolutional Variational Auto-encoder
      with Conditional Generative Adversarial Net
    '''
    def __init__(self, arch, is_training=False):
        self.arch = arch
        self._sanity_check()
        self.is_training = is_training

        with tf.name_scope('SpeakerRepr'):        
            self.y_emb = self._unit_embedding(
                self.arch['y_dim'],
                self.arch['z_dim'],
                'y_embedding')

        with tf.variable_scope('Tau'):
            self.tau = tf.nn.relu(
                10. * tf.Variable(
                    tf.ones([1]),
                    name='tau')) + 0.1

        self._generate = tf.make_template(
            'Generator',
            self._generator)
        self._discriminate = tf.make_template(
            'Discriminator',
            self._discriminator)
        self._encode = tf.make_template(
            'Encoder',
            self._encoder)
        self._classify = tf.make_template(
            'Classifier',
            self._classifier)


    def _sanity_check(self):
        for net in ['encoder', 'generator', 'classifier']:
            assert len(self.arch[net]['output']) > 2
            assert len(self.arch[net]['output']) == len(self.arch[net]['kernel'])
            assert len(self.arch[net]['output']) == len(self.arch[net]['stride'])


    def _unit_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim])
            embeddings = tf.nn.l2_normalize(embeddings, dim=-1, name=var_name+'normalized')
        return embeddings


    def _merge(self, var_list, fan_out, l2_reg=1e-6):
        ''' 
        Note: Don't apply BN on this because 'y' 
              tends to be the same inside a batch.
        '''
        x = 0.
        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=fan_out,
            weights_regularizer=slim.l2_regularizer(l2_reg),
            normalizer_fn=None,
            activation_fn=None):
            for var in var_list:
                x = x + slim.fully_connected(var)
        x = slim.bias_add(x)
        return x


    def _l2_regularized_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim],
                regularizer=slim.l2_regularizer(1e-6))
        return embeddings

    def _classifier(self, x, is_training):
        n_layer = len(self.arch['classifier']['output'])
        subnet = self.arch['classifier']

        with slim.arg_scope(
            [slim.batch_norm],
            scale=True, scope='BN',
            updates_collections=None,
            # decay=0.9, epsilon=1e-5,  # [TODO] Test these hyper-parameters
            is_training=is_training):
            with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu):

                for i in range(n_layer):
                    x = slim.conv2d(
                        x,
                        subnet['output'][i],
                        subnet['kernel'][i],
                        subnet['stride'][i])
                    tf.summary.image(
                        'down-sample{:d}'.format(i),
                        tf.transpose(x[:, :, :, 0:3], [2, 1, 0, 3]))

        x = slim.flatten(x)

        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=self.arch['y_dim'],
            weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
            normalizer_fn=None,
            activation_fn=None):
            y_logit = slim.fully_connected(x)
            # z_mu = slim.fully_connected(x)
            # z_lv = slim.fully_connected(x)
        # return z_mu, z_lv
        return y_logit


    def _encoder(self, x, y, is_training):
        n_layer = len(self.arch['encoder']['output'])
        subnet = self.arch['encoder']
        h, w, c = self.arch['hwc']

        y2x = slim.fully_connected(
            y,
            h * w * c,
            weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
            # normalizer_fn=None,
            activation_fn=tf.nn.sigmoid)
        y2x = tf.reshape(y2x, [-1, h, w, c])

        x = tf.concat([x, y2x], 3)

        with slim.arg_scope(
            [slim.batch_norm],
            scale=True, scope='BN',
            updates_collections=None,
            # decay=0.9, epsilon=1e-5,  # [TODO] Test these hyper-parameters
            is_training=is_training):
            with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu):

                for i in range(n_layer):
                    x = slim.conv2d(
                        x,
                        subnet['output'][i],
                        subnet['kernel'][i],
                        subnet['stride'][i])
                    tf.summary.image(
                        'down-sample{:d}'.format(i),
                        tf.transpose(x[:, :, :, 0:3], [2, 1, 0, 3]))

        x = slim.flatten(x)

        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=self.arch['z_dim'],
            weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
            normalizer_fn=None,
            activation_fn=None):
            z_mu = slim.fully_connected(x)
            z_lv = slim.fully_connected(x)
        return z_mu, z_lv


    def _generator(self, z, y, is_training):
        ''' In this version, we only generate the target, so `y` is useless '''
        subnet = self.arch['generator']
        n_layer = len(subnet['output'])
        h, w, c = subnet['hwc']

        # y = tf.nn.embedding_lookup(self.y_emb, y)

        x = self._merge([z, y], subnet['merge_dim'])
        x = lrelu(x)
        with slim.arg_scope(
            [slim.batch_norm],
            scale=True, scope='BN',
            updates_collections=None,
            # decay=0.9, epsilon=1e-5,
            is_training=is_training):

            x = slim.fully_connected(
                x,
                h * w * c,
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu)

            x = tf.reshape(x, [-1, h, w, c])

            with slim.arg_scope(
                [slim.conv2d_transpose],
                weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu):

                for i in range(n_layer -1):
                    x = slim.conv2d_transpose(
                        x,
                        subnet['output'][i],
                        subnet['kernel'][i],
                        subnet['stride'][i]
                        # normalizer_fn=None
                        )

                # Don't apply BN for the last layer of G
                x = slim.conv2d_transpose(
                    x,
                    subnet['output'][-1],
                    subnet['kernel'][-1],
                    subnet['stride'][-1],
                    normalizer_fn=None,
                    activation_fn=None)

                # pdb.set_trace()
                logit = x
                # x = tf.nn.tanh(logit)
        # return x, logit
        return tf.nn.sigmoid(logit), logit


    def _discriminator(self, x, is_training):
    # def _discriminator(self, x, is_training):
        ''' Note: In this version, `y` is useless '''
        subnet = self.arch['discriminator']
        n_layer = len(subnet['output'])

        # y_dim = self.arch['y_dim']
        # h, w, _ = self.arch['hwc']
        # y_emb = self._l2_regularized_embedding(y_dim, h * w, 'y_embedding_disc_in')
        # y_vec = tf.nn.embedding_lookup(y_emb, y)
        # y_vec = tf.reshape(y_vec, [-1, h, w, 1])

        intermediate = list()
        intermediate.append(x)

        # x = tf.concat(3, [x, y_vec])   # inject y into x
        with slim.arg_scope(
            [slim.batch_norm],
            scale=True, scope='BN',
            updates_collections=None,
            # decay=0.9, epsilon=1e-5,
            is_training=is_training):
            with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu):

                # Radford: [do] not applying batchnorm to the discriminator input layer
                x = slim.conv2d(
                    x,
                    subnet['output'][0],
                    subnet['kernel'][0],
                    subnet['stride'][0],
                    normalizer_fn=None)
                intermediate.append(x)
                for i in range(1, n_layer):
                    x = slim.conv2d(
                        x,
                        subnet['output'][i],
                        subnet['kernel'][i],
                        subnet['stride'][i])
                    intermediate.append(x)
                    tf.summary.image(
                        'upsampling{:d}'.format(i),
                        tf.transpose(x[:, :, :, 0:3], [2, 1, 0, 3]))

        # Don't apply BN for the last layer
        x = slim.flatten(x)
        h = slim.flatten(intermediate[subnet['feature_layer'] - 1])

        x = slim.fully_connected(
            x,
            1,
            weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
            activation_fn=None)

        return x, h  # no explicit `sigmoid`

    def circuit_loop(self, x, y_L=None):
        '''
        x:
        y_L: true label
        '''        
        y_logit_pred = self._classify(x, is_training=self.is_training)
        y_u_pred = tf.nn.softmax(y_logit_pred)
        y_logsoftmax_pred = tf.nn.log_softmax(y_logit_pred)

        y_logit_sample = GumbelSampleLayer(y_logsoftmax_pred)
        y_sample = tf.nn.softmax(y_logit_sample / self.tau)
        y_logsoftmax_sample = tf.nn.log_softmax(y_logit_sample)

        if y_L is not None:
            y = y_L
        else:
            y = y_sample
        
        z_mu, z_lv = self._encode(x, y, is_training=self.is_training)
        z = GaussianSampleLayer(z_mu, z_lv)

        xh, xh_sig_logit = self._generate(z, y, is_training=self.is_training)

        return dict(
            z=z,
            z_mu=z_mu,
            z_lv=z_lv,
            y_pred=y_u_pred,
            y_logsoftmax_pred=y_logsoftmax_pred,
            y_logit_pred=y_logit_pred,
            y_sample=y_sample,
            y_logit_sample=y_logit_sample,
            y_logsoftmax_sample=y_logsoftmax_sample,
            xh=xh,
            xh_sig_logit=xh_sig_logit
            )


    def loss(self, x_u, x_l, y_l):
        unlabel = self.circuit_loop(x_u)
        labeled = self.circuit_loop(x_l, y_l)

        with tf.name_scope('loss'):
            # def mean_sigmoid_cross_entropy_with_logits(logit, truth):
            #     '''
            #     truth: 0. or 1.
            #     '''
            #     return tf.reduce_mean(
            #         tf.nn.sigmoid_cross_entropy_with_logits(
            #             logit,
            #             truth * tf.ones_like(logit)))

            loss = dict()

            # Note:
            #   `log p(y)` should be a constant term if we assume that y 
            #   is equally distributed.
            #   That's why I omitted it.
            #   However, since y is now an approximation, I'm not sure
            #   whether omitting it is correct.

            # [TODO] What PDF should I use to compute H(y|x)?
            #   1. Categorical? But now we have a Continuous y @_@ 
            #   2. Gumbel-Softmax? But the PDF is.. clumsy

            with tf.name_scope('Labeled'):
                z_mu = labeled['z_mu']
                z_lv = labeled['z_lv']
                loss['KL(z_l)'] = tf.reduce_mean(
                    GaussianKLD(
                        z_mu, z_lv,
                        tf.zeros_like(z_mu), tf.zeros_like(z_lv)))

                loss['log p(x_l)'] = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=slim.flatten(labeled['xh_sig_logit']),
                            labels=slim.flatten(x_l)),
                        1))

                loss['Labeled'] = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=labeled['y_logit_pred'],
                        labels=y_l))


            with tf.name_scope('Unlabeled'):
                z_mu = unlabel['z_mu']
                z_lv = unlabel['z_lv']
                loss['KL(z_u)'] = tf.reduce_mean(
                    GaussianKLD(
                        z_mu, z_lv,
                        tf.zeros_like(z_mu), tf.zeros_like(z_lv)))

                loss['log p(x_u)'] = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=slim.flatten(unlabel['xh_sig_logit']),
                            labels=slim.flatten(x_u)),
                        1))

                y_prior = tf.ones_like(unlabel['y_sample']) / self.arch['y_dim']

                '''Eric Jang's code
                # loss and train ops
                kl_tmp = tf.reshape(q_y*(log_q_y-tf.log(1.0/K)),[-1,N,K])
                KL = tf.reduce_sum(kl_tmp,[1,2])
                elbo=tf.reduce_sum(p_x.log_prob(x),1) - KL
                '''
                # https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb

                # J: I chose not to use 'tf.nn.softmax_cross_entropy'
                #    because it takes logits as arguments but we need
                #    to subtract `log p` before `mul` p
                loss['H(y)'] = tf.reduce_mean(                
                    tf.reduce_sum(
                        tf.multiply(
                            unlabel['y_pred'],
                            tf.log(unlabel['y_pred'] + EPS) - tf.log(y_prior)),
                        -1))


                # Using Gumbel-Softmax Distribution: 
                #   1. Incorrect because p(y..y) is a scalar-- unless we can get
                #      the parametic form of the H(Y).
                #   2. The numerical value can be VERY LARGE, causing trouble!
                #   3. You should regard 'Gumbel-Softmax' as a `sampling step`

                # log_qy = GumbelSoftmaxLogDensity(
                #     y=unlabel['y_sample'],
                #     p=unlabel['y_pred'],
                #     tau=self.tau)
                # # loss['H(y)'] = tf.reduce_mean(- tf.mul(tf.exp(log_qy), log_qy))
                # loss['H(y)'] = tf.reduce_mean(- log_qy)


                # # [TODO] How to define this term? log p(y)
                # loss['log p(y)'] = - tf.nn.softmax_cross_entropy_with_logits(
                loss['log p(y)'] = 0.0
                
            loss['KL(z)'] = loss['KL(z_l)'] + loss['KL(z_u)']
            loss['Dis'] = loss['log p(x_l)'] + loss['log p(x_u)']
            loss['H(y)'] = loss['H(y)'] + loss['log p(y)']

            # For summaries
            with tf.name_scope('Summary'):
                # tf.summary.scalar('DKL_x', loss['KL(x)'])
                tf.summary.scalar('DKL_z', loss['KL(z)'])
                tf.summary.scalar('MMSE', loss['Dis'])

                tf.summary.histogram('z_s', unlabel['z'])

                tf.summary.histogram('z_mu_s', unlabel['z_mu'])

                tf.summary.histogram('z_lv_s', unlabel['z_lv'])
                # tf.summary.histogram('z_lv_t', t['z_lv'])

                # tf.summary.histogram('y_logit', unlabel['y_logit'])
                # tf.summary.histogram('y', unlabel['y'])

        return loss

    # def sample(self, z=128):
    #     ''' Generate fake samples given `z`
    #     if z is not given or is an `int`,
    #     this fcn generates (z=128) samples
    #     '''
    #     z = tf.random_uniform(
    #         shape=[z, self.arch['z_dim']],
    #         minval=-1.0,
    #         maxval=1.0,
    #         name='z_test')
    #     return self._generate(z, is_training=False)

    def encode(self, x):

        y_logit = self._classify(x, is_training=False)
        y = tf.nn.softmax(y_logit / self.tau)

        z_mu, z_lv = self._encode(x, y, is_training=False)
        # y_logit = self._classify(x, is_training=False)
        return dict(mu=z_mu, log_var=z_lv, y=y) #, y_logit=y_logit)

    def classify(self, x):
        y_logit = self._classify(x, is_training=False)
        y = tf.nn.softmax(y_logit / self.tau)
        return y

    def decode(self, z, y, tanh=False):
        # if tanh:
        #     return self._generate(z, y, is_training=False)
        # else:
        #     return self._generate(z, y, is_training=False)
        xh, _ = self._generate(z, y, is_training=False)
        # tf.summary.image('xh', tf.transpose(xh, [2, 1, 0, 3]))
        # return self._filter(xh, is_training=False)
        return xh

    # def classify(self, x):
    #     return self._classify(tf.nn.softmax(x), is_training=False)

    # def interpolate(self, x1, x2, n):
    #     ''' Interpolation from the latent space '''
    #     x1 = tf.expand_dims(x1, 0)
    #     x2 = tf.expand_dims(x2, 0)
    #     z1, _ = self._encode(x1, is_training=False)
    #     z2, _ = self._encode(x2, is_training=False)
    #     a = tf.reshape(tf.linspace(0., 1., n), [n, 1])

    #     z1 = tf.matmul(1. - a, z1)
    #     z2 = tf.matmul(a, z2)
    #     z = tf.nn.tanh(tf.add(z1, z2))  # Gaussian-to-Uniform
    #     xh = self._generate(z, is_training=False)
    #     xh = tf.concat(0, [x1, xh, x2])
    #     return xh
