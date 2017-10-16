import tensorflow as tf
from tf_utils import max_unpool, variable_on_cpu, variable_with_weight_decay, max_pool_with_argmax
import re

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
        self.sparsity = sparsity
        self.beta = beta

        self.form_variables()
        self.form_graph()

    def form_variables(self):
        self.input = tf.placeholder(shape=[self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]],
                                    dtype=self.dtype, name='input_batch')

        self.train_time = tf.placeholder(shape=(), dtype=tf.bool)
        self.global_step = tf.Variable(0, trainable=False)
        if self.mode == 'classification':
            self.labels = tf.placeholder(shape=[None,], dtype=tf.int64, name='labels')

    def encoder_forward(self):
        encoder_whats = []
        encoder_wheres = []
        encoder_convs = []
        encoder_what = self.input

        for i, layer in enumerate(self.layers):
            # convn
            with tf.variable_scope('conv{}'.format(i+1)):
                encoder_what = tf.layers.conv2d(encoder_what, layer.channel_size, layer.filter_size, padding='valid',
                                                activation=tf.nn.relu)
                encoder_convs.append(encoder_what)

            # pooln
            if layer.pool_size is not None:
                encoder_what, encoder_where = max_pool_with_argmax(encoder_what, layer.pool_size, layer.pool_size)
                encoder_wheres.append(encoder_where)

            else:
                encoder_wheres.append(None)

            encoder_whats.append(encoder_what)

        self.encoder_whats = encoder_whats
        pool_shape = encoder_whats[-1].get_shape()
        self.flatten = tf.reshape(encoder_whats[-1], [-1, (pool_shape[1] * pool_shape[2] * pool_shape[3]).value])
        self.encoder_wheres = encoder_wheres
        self.encoder_convs = encoder_convs

        if self.rep_size is None:
            self.representation = self.flatten
        else:
            with tf.name_scope('encoder_fc'):
                encoder_fc = tf.layers.dense(self.flatten,self.rep_size, activation=tf.nn.relu)
                tf.summary.histogram('representation', encoder_fc)

                p_hat = tf.reduce_mean(encoder_fc, axis=0) # Mean over the batch
                p = tf.get_variable(name='p', shape=(self.rep_size), dtype=tf.float32, initializer=tf.constant_initializer(self.sparsity),
                                    trainable=False)
                one = tf.get_variable(name='1', shape=(self.rep_size), dtype=tf.float32, initializer=tf.constant_initializer(1.0),
                                      trainable=False)
                kl_divergence = tf.multiply(p, (tf.log(p) - tf.log(p_hat + 0.0001))) + tf.multiply(tf.subtract(one, p),
                                                                                          (tf.log(tf.subtract(one, p)) - tf.log(tf.subtract(one, p_hat))))
                kl_divergence = tf.multiply(self.beta, tf.reduce_sum(kl_divergence), name='sparsity')
                tf.add_to_collection('losses', kl_divergence)

            self.representation = encoder_fc

    def decoder_forward(self):
        if self.rep_size is None:
            decoder_what = self.encoder_whats[-1]
        else:
            with tf.name_scope('decoder_fc'):
                decoder_what = self.representation

                decoder_what = tf.layers.dense(decoder_what,self.flatten.get_shape()[1].value)
                fc_loss = tf.multiply(self.lambda_M, tf.nn.l2_loss(tf.subtract(decoder_what, self.flatten)), name='dense')
                tf.add_to_collection('losses', fc_loss)

                pool_shape = self.encoder_whats[-1].get_shape()
                decoder_what = tf.reshape(decoder_what, [-1, pool_shape[1].value, pool_shape[2].value, pool_shape[3].value])

        decoder_whats = []
        for i in range(len(self.layers)-1, -1, -1):
            print(i, len(self.encoder_whats))
            layer = self.layers[i]
            #unpooln
            if self.encoder_wheres[i] is not None:
                decoder_what = max_unpool(decoder_what, self.encoder_convs[i], self.encoder_wheres[i])

            with tf.variable_scope('deconv{}'.format(i+1)):
                if i == 0: # Does not use non-linearity at the last layer
                    shape = self.image_shape[-1]
                    decoder_what = tf.nn.conv2d_transpose(decoder_what, [1,layer.filter_size, layer.filter_size, 1],
                                           output_shape=self.input.get_shape(), strides=[1,1,1,1], padding='VALID')
                    # decoder_what = tf.layers.conv2d_transpose(decoder_what, shape, layer.filter_size, padding='valid')
                else:
                    shape = self.layers[i - 1].channel_size
                    print(decoder_what)
                    decoder_what = tf.nn.conv2d_transpose(decoder_what, [1,layer.filter_size, layer.filter_size, 1],
                                           output_shape=self.encoder_whats[i-1].get_shape(), strides=[1,1,1,1], padding='VALID')

                decoder_whats.append(decoder_what)

            if i != 0:
                middle_loss = tf.multiply(self.lambda_M, tf.nn.l2_loss(tf.subtract(decoder_what, self.encoder_whats[i-1])), name='middle')
                tf.add_to_collection('losses', middle_loss)

        self.decoder_what = decoder_what

    def fully_connected_forward(self):
        representation = self.representation
        dim = self.representation.get_shape()[1].value
        locals = []

        for i, units in enumerate(self.fc_layers):
            with tf.variable_scope('fc{}'.format(i+1)):
                if i == 0:
                    inp_dim = dim
                else:
                    inp_dim = self.fc_layers[i-1]

                weights = variable_with_weight_decay('weights', shape=[inp_dim, units],
                                                      stddev=0.04, wd=0.001, dtype=self.dtype, trainable=True)
                biases = variable_on_cpu('biases', [units], tf.constant_initializer(0.1), dtype=self.dtype,
                                         trainable=True)
                if i == 0:
                    rep_drop = tf.nn.dropout(representation, 0.5)
                    local = tf.nn.relu(tf.matmul(rep_drop, weights) + biases, name='local')
                else:
                    local = tf.nn.relu(tf.matmul(locals[i-1], weights) + biases, name='local')

                locals.append(local)

        with tf.variable_scope('softmax_linear'):
            non_linear = locals[-1]
            weights = variable_with_weight_decay('weights', shape=[self.fc_layers[-1], self.num_classes],
                                                 stddev=1 / 192.0, wd=0.0, dtype=self.dtype, trainable=True)
            biases = variable_on_cpu('biases', [self.num_classes], tf.constant_initializer(0.0), dtype=self.dtype,
                                     trainable=True)

            output = tf.add(tf.matmul(non_linear, weights), biases, name='output')

        return output


    def ae_loss(self):
        reconstruction_loss = tf.multiply(self.lambda_rec, tf.nn.l2_loss(tf.subtract(self.input, self.decoder_what)), name='reconstruction')
        tf.add_to_collection('losses', reconstruction_loss)
        losses = tf.get_collection('losses')

        total_loss = tf.add_n(losses, name='total_loss')

        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            loss_name = l.op.name
            tf.summary.scalar(loss_name, l)

        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)

        return total_loss

    def softmax_loss(self, fc_out):
        labels = tf.cast(self.labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=fc_out, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        s_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar('classification loss', s_loss)
        return s_loss

    def init_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt_op = optimizer.minimize(loss, global_step=self.global_step)

    def form_graph(self):
        print("Forming encoder", flush=True)
        self.encoder_forward()
        if self.mode == 'autoencode':
            print("Forming decoder", flush=True)
            self.decoder_forward()
            self.ae_loss = self.ae_loss()
            print("Forming L2 optimizer with learning rate {}".format(self.learning_rate), flush=True)
            self.init_optimizer(self.ae_loss)
            tf.summary.image('whatwhere/stacked', tf.concat((self.input, self.decoder_what), axis=2))

        elif self.mode == 'classification':
            print("Forming fully connected")
            fc_out = self.fully_connected_forward()
            self.s_loss = self.softmax_loss(fc_out)
            print("Forming classification optimizier with learning rate{}".format(self.learning_rate))
            self.predictions = tf.argmax(fc_out, axis=1)
            correct_pred = tf.equal(self.labels, self.predictions)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('batch accuracy', self.accuracy)
            self.init_optimizer(self.s_loss)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('tensorboard/{}/train'.format(self.tensorboard_id), self.sess.graph)
        self.test_writer = tf.summary.FileWriter('tensorboard/{}/test'.format(self.tensorboard_id), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def train(self, input, labels=None):
        if self.mode == 'autoencode':
            _, batch_loss, global_step, tb_merge = self.sess.run([self.opt_op, self.ae_loss, self.global_step, self.merged],
                                                       feed_dict={self.input: input})
            self.train_writer.add_summary(tb_merge, global_step)
            return batch_loss, global_step
        elif self.mode == 'classification':
            _, batch_loss, predictions, accuracy, global_step, tb_merge = self.sess.run([self.opt_op, self.s_loss, self.predictions, self.accuracy,
                                                                                        self.global_step, self.merged],
                                                                                        feed_dict={self.input: input, self.labels:labels})
            self.train_writer.add_summary(tb_merge, global_step)
            return batch_loss, accuracy, global_step

    def eval(self, input, labels=None):
        if self.mode == 'autoencode':
            loss, tb_merge, global_step = self.sess.run([self.ae_loss, self.merged, self.global_step],
                                                        feed_dict={self.input:input})
            self.test_writer.add_summary(tb_merge, global_step)
            return loss
        elif self.mode == 'classification':
            loss, accuracy, tb_merge, global_step = self.sess.run([self.s_loss, self.accuracy, self.merged, self.global_step],
                                                                  feed_dict={self.input:input, self.labels:labels})
            self.test_writer.add_summary(tb_merge, global_step)
            return loss, accuracy

    def get_representation(self, input):
        return self.sess.run(self.representation, feed_dict={self.input:input})

    def save(self, path, ow=True):
        saver = tf.train.Saver()
        if ow:
            save_path = saver.save(self.sess, save_path=path)
        else:
            save_path = saver.save(self.sess, save_path=path, global_step=self.global_step)

        print('model saved at {}'.format(save_path), flush=True)

    def restore(self, path):
        if self.mode == 'classification':
            var_list = []
            for i, _ in enumerate(self.layers):
                var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv{}'.format(i+1))
            saver = tf.train.Saver(var_list=var_list)
        else:
            saver = tf.train.Saver()
        saver.restore(self.sess, save_path=path)