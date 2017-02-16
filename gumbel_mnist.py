import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import json
from model.gvae import VCGAN
# import pdb
# from tqdm import tqdm

import os
from sklearn import metrics

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', 'tmp', 'log dir')


def load_mnist(filename, N):
    ''' Load MNIST from the downloaded and unzipped file
    into [N, 28, 28, 1] dimensions with [0, 1] values
    '''
    with open(filename) as f:
        x = np.fromfile(file=f, dtype=np.uint8)
    x = x[16:].reshape([N, 28, 28, 1]).astype(np.float32)
    x = x / 255.
    return x

def load_mnist_label(filename, N):
    with open(filename) as f:
        x = np.fromfile(file=f, dtype=np.uint8)
    x = x[8:].reshape([N]).astype(np.int32)
    return x


def get_optimization_ops(loss, args, arch):
    '''
    [TODO]
    Although most of the trainer structures are the same,
    I think we have to use different training scripts for VAE- and DC-GAN
    (but do we have to have two different classes of VAE- and DC-?)
    '''
    optimizer_d = tf.train.RMSPropOptimizer(
        arch['training']['lr'])
    optimizer_g = tf.train.RMSPropOptimizer(
        arch['training']['lr'])

    trainables = tf.trainable_variables()
    trainables = [v for v in trainables if 'Tau' not in v.name]

    a = arch['training']['alpha']

    obj_Ez = loss['KL(z)'] + loss['Dis'] - loss['H(y)'] + a * loss['Labeled']  # + a * loss['H(y)']

    # Note:
    # 1. minimizing H(y) cause the classifier to assign all the prediction to a single class.
    #    Ironically, in this case, classification error for labeled data is also minimized
    #    but in the wrong way.

    opt_g = optimizer_g.minimize(
        obj_Ez,
        var_list=trainables)
        # var_list=e_vars+y_vars+g_vars)

    return dict(g=opt_g)


def pick_supervised_samples(x, y, smp_per_class=10):
    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    # pdb.set_trace()
    x_s, y_s = list(), list()
    for i in range(10):
        count = 0
        idx = list()
        for j in range(y.shape[0]):
            if y[j] == i and count < smp_per_class:
                idx.append(j)
                count += 1
        y_s = y_s + idx
    x_s = x[y_s]
    y_s = y[y_s]
    return x_s, y_s

# 
def halflife(t, N0=1., T_half=1., thresh=0.0):
    l = np.log(2.) / T_half
    Nt = (N0 - thresh) * np.exp(-l * t) + thresh
    return np.asarray(Nt).reshape([1,])


def reshape(b, sqrt_bz):
    b = np.reshape(b, [sqrt_bz, sqrt_bz, 28, 28])
    b = np.transpose(b, [0, 2, 1, 3])
    b = np.reshape(b, [sqrt_bz*28, sqrt_bz*28])
    return b


def make_thumbnail(y, z, arch, net):
    ''' Make a K-by-K thumbnail images '''
    with tf.name_scope('Thumbnail'):
        k = arch['y_dim']
        h, w, c = arch['hwc']

        y = tf.tile(y, [k, 1])

        z = tf.expand_dims(z, -1)
        z = tf.tile(z, [1, 1, k])
        z = tf.reshape(z, [-1, arch['z_dim']])

        xh = net.decode(z, y)  # 100, 28, 28, 1
        xh = tf.reshape(xh, [k, k, h, w, c])
        xh = tf.transpose(xh, [0, 2, 1, 3, 4])
        xh = tf.reshape(xh[:, :, :, :, 0], [k*h, k*w])
    return xh



N_TRAIN = 60000
N_TEST = 10000

if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)

x = load_mnist('/home/jrm/corpora/mnist/train-images-idx3-ubyte', N=N_TRAIN)
y_trn = load_mnist_label('/home/jrm/corpora/mnist/train-labels-idx1-ubyte', N=N_TRAIN)

x_s, y_s = pick_supervised_samples(x[:10000], y_trn[:10000], 10)
x_1, y_1 = pick_supervised_samples(x_s, y_s, 1)

x_l_show = reshape(x_s, 10)

plt.figure()
plt.imshow(x_l_show, cmap='gray')
plt.axis('off')
plt.savefig(os.path.join(args.logdir, 'x_labeled.png'))
plt.close()

x = x[10000:]
y_trn = y_trn[10000:]

N_TRAIN = 50000  # [TODO]

x_t = load_mnist(
    '/home/jrm/corpora/mnist/t10k-images-idx3-ubyte',
    N=N_TEST)
y_t = load_mnist_label(
    '/home/jrm/corpora/mnist/t10k-labels-idx1-ubyte',
    N=N_TEST)

with open('architecture-g.json') as f:
    arch = json.load(f)


batch_size = arch['training']['batch_size']
N_EPOCH = arch['training']['epoch']
N_ITER = N_TRAIN // batch_size
N_HALFLIFE = arch['training']['halflife']
SMALLEST_TAU = arch['training']['smallest_tau']

h, w, c = arch['hwc']
X_u = tf.placeholder(shape=[None, h, w, c], dtype=tf.float32)

# X_l = tf.placeholder(shape=[None, h, w, c], dtype=tf.float32)
# Y_l = tf.placeholder(shape=[None, arch['y_dim']], dtype=tf.float32)

X_l = tf.constant(x_s)
Y_l = tf.one_hot(y_s, arch['y_dim'])

net = VCGAN(arch)
loss = net.loss(X_u, X_l, Y_l)

encodings = net.encode(X_u)
Z_u = encodings['mu']
Y_u = encodings['y']
Xh = net.decode(Z_u, Y_u)

label_pred = tf.argmax(Y_u, 1)
Y_pred = tf.one_hot(label_pred, arch['y_dim'])
Xh2 = net.decode(Z_u, Y_pred)

thumbnail = make_thumbnail(Y_u, Z_u, arch, net)

opt = get_optimization_ops(loss, args=None, arch=arch)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# ===============================
# [TODO] 
#   1. batcher class
#      1) for train and for test
#      2) binarization
#      3) shffule as arg
#   2. plot()
#   3. MNIST loader to
#      0) load images and label
#      1) Separate trn, vld, tst
#      2) preprocess with 'sigmoid' or 'tanh' nrm
#   4. Args
#   5. TBoard (training tracker to monitor the convergence)
#   6. Model saver
# ===============================

sqrt_bz = int(np.sqrt(batch_size))

# for step in range(N_EPOCH * N_ITER):
#     ep, it = divmod(step, N_ITER)
logfile = os.path.join(args.logdir, 'log.txt')
for ep in range(N_EPOCH):
    np.random.shuffle(x)

    for it in range(N_ITER):
        step = ep * N_ITER + it

        # idx = np.random.randint(0, N_TRAIN, [batch_size])
        idx = range(it * batch_size, (it + 1) * batch_size)
        tau = halflife(
            step,
            N0=arch['training']['largest_tau'],
            T_half=N_ITER*N_HALFLIFE,
            thresh=arch['training']['smallest_tau'])

        batch = np.random.binomial(1, x[idx])

        _, l_x, l_z, l_y, l_l = sess.run(
            [opt['g'], loss['Dis'], loss['KL(z)'], loss['H(y)'], loss['Labeled']],
            {X_u: batch,
             net.tau: tau})

        msg = 'Ep [{:03d}/{:d}]-It[{:03d}/{:d}]: Lx: {:6.2f}, KL(z): {:4.2f}, L:{:.2e}: H(u): {:.2e}'.format(
            ep, N_EPOCH, it, N_ITER, l_x, l_z, l_l, l_y)
        print(msg)

        if it == (N_ITER -1):
            b, z, y, xh, xh2 = sess.run(
                [X_u, Z_u, Y_u, Xh, Xh2],
                {X_u: batch,
                 net.tau: tau})

            b = reshape(b, sqrt_bz)
            xh = reshape(xh, sqrt_bz)
            xh2 = reshape(xh2, sqrt_bz)

            y = np.argmax(y, 1).astype(np.int32)
            y = np.reshape(y, [sqrt_bz, sqrt_bz])

            png = os.path.join(args.logdir, 'Ep-{:03d}-reconst.png'.format(ep))
            with open(logfile, 'a') as f:
                f.write(png + '  ')
                f.write('Tau: {:.3f}\n'.format(tau[0]))
                f.write(msg + '\n')
                n, m = y.shape
                for i in range(n):
                    for j in range(m):
                        f.write('{:d} '.format(y[i, j]))
                    f.write('\n')
                f.write('\n\n')


            plt.figure(figsize=(30, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(b, cmap='gray')
            plt.axis('off')
            plt.title('Ground-truth')
            plt.subplot(1, 3, 2)
            plt.imshow(xh, cmap='gray')
            plt.axis('off')
            plt.title('Reconstructed using dense label')
            plt.subplot(1, 3, 3)
            plt.imshow(xh2, cmap='gray')
            plt.axis('off')
            plt.title('Reconstructed using onehot label')
            plt.savefig(png)
            plt.close()

        if it == (N_ITER - N_ITER) and ep % arch['training']['summary_freq'] == 0:
            # ==== Classification ====
            y_all = list()
            bz = 100
            for i in range(N_TEST // bz):
                b_t = x_t[i * bz: (i + 1) * bz]
                b_t[b_t > 0.5] = 1.0  # [MAKESHIFT] Binarization
                b_t[b_t <= 0.5] = 0.0
                p = sess.run(
                    label_pred,
                    {X_u: b_t,
                     net.tau: tau})
                y_all.append(p)
            y_all = np.concatenate(y_all, 0)

            # ==== Style Conversion ====
            y_u = np.eye(arch['y_dim'])
            tn = sess.run(
                thumbnail,
                {Y_u: y_u, X_u: x_1})
            plt.figure()
            plt.imshow(tn, cmap='gray')
            plt.axis('off')
            plt.savefig(
                os.path.join(
                    args.logdir,
                    'Ep-{:03d}-conv.png'.format(ep)))
            plt.close()

            # =========== [False] Accuracy ======== 
            # acc = 0
            # mapping = dict()
            # for i in range(10):
            #     idx = y_all == i
            #     # y_all[idx]
            #     count_over_class = y_t[idx].tolist()
            #     count_over_class = [count_over_class.count(c) for c in range(10)]
            #     y_true = np.asarray(count_over_class).argmax()
            #     mapping[i] = y_true
            #     acc += count_over_class[y_true]

            
            # MI = metrics.adjusted_mutual_info_score(y_t, y_all)
            # print('Acc: ', acc / y_all.shape[0])
            with open(logfile, 'a') as f:
                # == Confusion Matrix ==
                cm = metrics.confusion_matrix(y_t, y_all)
                n, m = cm.shape
                for i in range(n):
                    for j in range(m):
                        f.write('{:4d} '.format(cm[i, j]))
                    f.write('\n')
                # f.write('Ajusted MI score: {:.4f}\n'.format(MI))
                # == Accuarcy ==
                acc = metrics.accuracy_score(y_t, y_all)
                f.write('Accuracy: {:.4f}\n'.format(acc))
                f.write('\n\n')



