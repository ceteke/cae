import tensorflow as tf
from tf_utils import l2_regulazier
import numpy as np

class SWWAE:
    def __init__(self, sess, image_shape, mode, layers, rep_size=None, fc_layers=None, learning_rate=None, lambda_rec=None,
                 lambda_M=None, dtype=tf.float32, tensorboard_id=None, num_classes=None, encoder_train=True, batch_size=32,
                 sparsity=0.05, beta=0.5):
        self.layers = layers
        self.dtype = dtype
        self.mode = mode
        self.lambda_M = lambda_M
        self.lambda_rec = lambda_rec
        self.sess = sess
        self.learning_rate = learning_rate
        self.image_shape = image_shape
        self.tensorboard_id = tensorboard_id
        self.fc_layers = fc_layers
        self.num_classes = num_classes
        self.encoder_train = encoder_train
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.sparsity = np.array([sparsity] * rep_size).astype(np.float32)
        self.beta = beta
        self.regulazier = l2_regulazier(0.01, collection_name='losses')
        self.kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1e-2)
        self.bias_initializer = tf.constant_initializer(0.0)
        self.form_variables()
        self.form_graph()

    def form_variables(self):
        self.input = tf.placeholder(shape=[self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]],
                                    dtype=self.dtype, name='input_batch')
        self.expected_output = tf.placeholder(shape=[self.batch_size] + self.image_shape, dtype=self.dtype, name='output_batch')
        self.train_time = tf.placeholder(shape=(), dtype=tf.bool)
        self.global_step = tf.Variable(0, trainable=False)
        self.dropout_rate = tf.placeholder(shape=(), dtype=tf.float32)

    def encoder_forward(self):
        encoder_what = self.input
        encoder_whats = []

        for i, layer in enumerate(self.layers):
            # convn
            with tf.variable_scope('conv{}'.format(i+1)):
                encoder_what = tf.layers.conv2d(encoder_what, layer.channel_size, layer.filter_size, padding='same',
                                                activation=tf.nn.relu, kernel_initializer=self.kernel_initializer,
                                                kernel_regularizer=self.regulazier, bias_initializer=self.bias_initializer,
                                                strides=2)
            encoder_whats.append(encoder_what)

        self.encoder_whats = encoder_whats
        pool_shape = encoder_what.get_shape()
        self.encoder_what = encoder_what
        self.flatten = tf.reshape(encoder_what, [-1, (pool_shape[1] * pool_shape[2] * pool_shape[3]).value])

        if self.rep_size is None:
            self.representation = self.flatten
        else:
            with tf.name_scope('encoder_fc'):
                encoder_fc = tf.layers.dense(self.flatten, self.rep_size, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.regulazier, bias_initializer=self.bias_initializer)
                tf.summary.histogram('representation', encoder_fc)
                sparsity_loss = self.beta * self.kl_divergence(encoder_fc)
                tf.add_to_collection('losses', sparsity_loss)

            self.representation = encoder_fc

    def kl_divergence(self, p_hat):
        p = self.sparsity
        return tf.reduce_mean(p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat), name='sparsity')

    def decoder_forward(self):
        if self.rep_size is None:
            decoder_what = self.encoder_what
        else:
            with tf.name_scope('decoder_fc'):
                decoder_what = tf.layers.dense(self.representation, self.flatten.get_shape()[1].value,kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.regulazier, bias_initializer=self.bias_initializer, activation=tf.nn.relu)
                pool_shape = self.encoder_what.get_shape()
                decoder_what = tf.reshape(decoder_what, [self.batch_size, pool_shape[1].value, pool_shape[2].value, pool_shape[3].value])

        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            decoder_what = tf.add(decoder_what, self.encoder_whats[i])
            with tf.variable_scope('deconv{}'.format(i+1)):
                if i == 0:  # Does not use non-linearity at the last layer
                    output_shape = self.input.get_shape()
                    filter_size = [layer.filter_size, layer.filter_size, self.image_shape[-1], layer.channel_size]
                    bias_size = self.image_shape[-1]
                else:
                    up = self.encoder_whats[i - 1]
                    output_shape = up.get_shape()
                    filter_size = [layer.filter_size, layer.filter_size, self.layers[i - 1].channel_size,
                                   layer.channel_size]
                    bias_size = self.layers[i - 1].channel_size

                filter = tf.get_variable('filter', shape=filter_size, dtype=self.dtype,
                                         initializer=self.kernel_initializer, regularizer=self.regulazier)
                bias = tf.get_variable('bias', shape=bias_size, dtype=self.dtype,
                                       initializer=self.bias_initializer)

                decoder_what = tf.nn.conv2d_transpose(decoder_what, filter, output_shape,
                                                      strides=[1, 2, 2, 1])
                decoder_what = tf.nn.bias_add(decoder_what, bias)

                if i != 0:
                    decoder_what = tf.nn.relu(decoder_what)
            print(decoder_what)
        self.decoder_what = decoder_what

    def ae_loss(self):
        reconstruction_loss = tf.reduce_sum(tf.pow(tf.subtract(self.expected_output, self.decoder_what), 2.0))
        tf.add_to_collection('losses', reconstruction_loss)
        losses = tf.get_collection('losses')

        total_loss = tf.add_n(losses, name='total_loss')

        for l in losses + [total_loss]:
            loss_name = l.op.name
            tf.summary.scalar(loss_name, l)

        return total_loss

    def init_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt_op = optimizer.minimize(loss, global_step=self.global_step)

    def form_graph(self):
        print("Forming encoder", flush=True)
        self.encoder_forward()
        print("Forming decoder", flush=True)
        self.decoder_forward()
        self.loss = self.ae_loss()
        print("Forming L2 optimizer with learning rate {}".format(self.learning_rate), flush=True)
        if self.mode == 'autoencode':
            self.init_optimizer(self.loss)
        tf.summary.image('whatwhere/stacked', tf.concat((self.input, self.decoder_what), axis=2))

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('tensorboard/{}/train'.format(self.tensorboard_id), self.sess.graph)
        self.test_writer = tf.summary.FileWriter('tensorboard/{}/test'.format(self.tensorboard_id), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def train(self, input, expected_output):
        _, batch_loss, global_step, tb_merge = self.sess.run([self.opt_op, self.loss, self.global_step, self.merged],
                                                       feed_dict={self.input: input, self.expected_output: expected_output, self.train_time:True,
                                                                  self.dropout_rate:0.25})
        self.train_writer.add_summary(tb_merge, global_step)
        return batch_loss, global_step

    def eval(self, input, expected_output):
        loss, tb_merge, global_step = self.sess.run([self.loss, self.merged, self.global_step],
                                                        feed_dict={self.input:input, self.expected_output:expected_output, self.train_time:False,
                                                                   self.dropout_rate:0.0})
        self.test_writer.add_summary(tb_merge, global_step)
        return loss

    def get_representation(self, input):
        return self.sess.run(self.representation, feed_dict={self.input:input, self.train_time:False, self.dropout_rate:0.0})

    def save(self, path, ow=True):
        saver = tf.train.Saver()
        if ow:
            save_path = saver.save(self.sess, save_path=path)
        else:
            save_path = saver.save(self.sess, save_path=path, global_step=self.global_step)

        print('model saved at {}'.format(save_path), flush=True)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, save_path=path)