import tensorflow as tf
import numpy as np


class SingleAE(object):
    def __init__(self, shape, para, data, layer_idx, activation_fun1, activation_fun2):
        self.para = para
        self.data = data
        self.layer_idx = layer_idx

        self.x = tf.placeholder(tf.float32, [None, shape[0]])
        self.dropout = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)

        self.var_list = []
        with tf.variable_scope('SAE') as scope:
            stddev = 1.0 / np.sqrt(shape[0])
            # self.x_c = gaussian_noise_layer(self.x, 0.001)
            self.x_c = tf.nn.dropout(self.x, self.dropout)
            self.W1 = tf.Variable(tf.random_normal([shape[0], shape[1]], stddev=stddev), name="W1")
            self.b1 = tf.Variable(tf.zeros([shape[1]], name='b1'))
            self.h = tf.add(tf.matmul(self.x_c, self.W1), self.b1)
            if activation_fun1 != None:
                self.h = activation_fun1(self.h)
            self.h = tf.nn.dropout(self.h, self.dropout)

            stddev = 1.0 / np.sqrt(shape[1])
            self.W2 = tf.Variable(tf.random_normal([shape[1], shape[0]], stddev=stddev), name='W2')
            self.b2 = tf.Variable(tf.zeros([shape[0]], name='b2'))
            self.x_hat = tf.add(tf.matmul(self.h, self.W2), self.b2)
            if activation_fun2 != None:
                self.x_hat = activation_fun2(self.x_hat)

            self.var_list.extend([self.W1, self.b1, self.W2, self.b2])

        self.loss = tf.reduce_mean(tf.square(self.x - self.x_hat))
        # self.loss = -tf.reduce_mean(self.x * tf.log(self.x_hat)+(1-self.x)*tf.log(1-self.x_hat))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=self.loss, var_list=self.var_list)

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())

    def doTrain(self):
        num_samples = len(self.data)
        lr = self.para['lr']

        for i in range(self.para['iters']):
            batch_indices = np.random.randint(num_samples, size=self.para['batch_size'])
            batch_x = self.data[batch_indices,:]
            self.sess.run(self.opt, feed_dict={self.x: batch_x, self.dropout: self.para['dropout'], self.lr: lr})

            if i % 1000 == 0:
                error = self.sess.run(self.loss, feed_dict={self.x: self.data, self.dropout: 1.0})
                print('pretrain layer-{}, epoch-{}: {}'.format(self.layer_idx, i, error))

            if i % 20000 == 0:
                lr /= 10.0



    def getWb(self):
        return self.sess.run([self.W1, self.b1, self.W2, self.b2])

    def getH(self):
        return self.sess.run(self.h, feed_dict={self.x: self.data, self.dropout: 1.0})

    def close(self):
        return self.sess.close()

