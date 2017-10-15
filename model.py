import tensorflow as tf
from tf_utils import max_unpool,  max_pool_with_argmax
import re

class SWWAE:
    def __init__(self, sess, image_shape, mode, layers, fc_ae_layers=None, fc_layers=None, learning_rate=None, lambda_rec=None,
                 lambda_M=None, dtype=tf.float32, tensorboard_id=None, num_classes=None, encoder_train=True, batch_size=32,
                 num_gpu=1):
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
        self.fc_ae_layers = fc_ae_layers
        self.batch_size = batch_size
        self.num_gpu = num_gpu

        self.form_variables()
        self.form_graph()

    def form_variables(self):
        self.input = tf.placeholder(shape=[self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]],
                                    dtype=self.dtype, name='input_batch')
        self.dropout_rate = tf.placeholder(shape=(), dtype=tf.float32)
        self.train_time = tf.placeholder(shape=(), dtype=tf.bool)
        self.global_step = tf.Variable(0, trainable=False)

    def inference(self, input):
        encoder_whats = []
        encoder_wheres = []
        encoder_what = input

        # ENCODER CONVOLUTIONS
        for i, layer in enumerate(self.layers):
            # convn
            with tf.variable_scope('conv{}'.format(i+1)) as scope:
                encoder_what = tf.layers.conv2d(encoder_what, layer.channel_size, layer.filter_size, padding='same',
                                                activation=tf.nn.relu, name=scope)

            # pooln
            if layer.pool_size is not None:
                with tf.variable_scope('pool{}'.format(i+1)) as scope:
                    encoder_what, encoder_where = max_pool_with_argmax(encoder_what, layer.pool_size, layer.pool_size)
                encoder_wheres.append(encoder_where)

            else:
                encoder_wheres.append(None)

            encoder_whats.append(encoder_what)

        # END OF ENCODER CONVOLUTIONS

        # ENCODER FULLY CONNECTED
        pool_shape = encoder_whats[-1].get_shape()
        flatten = tf.reshape(encoder_whats[-1], [-1, (pool_shape[1] * pool_shape[2] * pool_shape[3]).value])

        if len(self.fc_ae_layers) == 0:
            representation = tf.identity(flatten, name='representation')
            print(representation)
        else:
            encoder_fcs = []
            for i, layer in enumerate(self.fc_ae_layers):
                with tf.variable_scope('encoder_fc{}'.format(i+1)):
                    if i == 0:
                        encoder_fc = tf.layers.dense(flatten,self.fc_ae_layers[i], activation=tf.nn.relu)
                    else:
                        encoder_fc = tf.layers.dense(encoder_fc,self.fc_ae_layers[i], activation=tf.nn.relu)
                    encoder_fcs.append(encoder_fc)
            representation = tf.identity(encoder_fc, name='representation')

        # DECODER REVERSE FULLY CONNECTED

        if len(self.fc_ae_layers) == 0:
            decoder_what = encoder_whats[-1]
        else:
            decoder_what = tf.layers.dropout(encoder_fc, self.dropout_rate)
            for i in range(len(self.fc_ae_layers)-1, -1, -1):
                with tf.variable_scope('decoder_fc{}'.format(i+1)):
                    if i == 0:
                        decoder_what = tf.layers.dense(decoder_what, flatten.get_shape()[1].value,activation=tf.nn.relu)
                        fc_middle_loss = tf.multiply(self.lambda_M,
                                                     tf.nn.l2_loss(tf.subtract(decoder_what, flatten)), name='fc_middle')
                        tf.add_to_collection('losses', fc_middle_loss)
                    else:
                        decoder_what = tf.layers.dense(decoder_what,self.fc_ae_layers[i-1],activation=tf.nn.relu)

                        fc_middle_loss = tf.multiply(self.lambda_M,
                                                  tf.nn.l2_loss(tf.subtract(decoder_what, encoder_fcs[i - 1])), name='fc_middle')
                        tf.add_to_collection('losses', fc_middle_loss)

            decoder_what = tf.reshape(decoder_what, [-1, pool_shape[1].value, pool_shape[2].value, pool_shape[3].value])

        # END OF DECODER REVERSE FULLY CONNECTED

        # DECODER REVERSE CONVOLUTIONS

        decoder_whats = []
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            # unpooln
            if encoder_wheres[i] is not None:
                with tf.variable_scope('unpool{}'.format(i+1)):
                    decoder_what = max_unpool(decoder_what, encoder_wheres[i], layer.pool_size)

            with tf.variable_scope('deconv{}'.format(i + 1)):
                if i == 0:  # Does not use non-linearity at the last layer
                    shape = self.image_shape[-1]
                    decoder_what = tf.layers.conv2d_transpose(decoder_what, shape, layer.filter_size, padding='same')
                else:
                    shape = self.layers[i - 1].channel_size
                    decoder_what = tf.layers.conv2d_transpose(decoder_what, shape, layer.filter_size, padding='same',
                                                              activation=tf.nn.relu)

                decoder_whats.append(decoder_what)

            if i != 0:
                middle_loss = tf.multiply(self.lambda_M,
                                          tf.nn.l2_loss(tf.subtract(decoder_what, encoder_whats[i - 1])), name='deconv_middle')
                tf.add_to_collection('losses', middle_loss)

        # END OF THE MODEL

        return decoder_what, representation

    def get_tower_loss(self, scope):
        decoder_what, _ = self.inference(self.input)
        reconstruction_loss = tf.multiply(self.lambda_rec, tf.nn.l2_loss(tf.subtract(self.input, decoder_what)),
                                          name='rec_loss')
        tf.add_to_collection('losses', reconstruction_loss)

        losses = tf.get_collection('losses', scope)
        total_loss = tf.add_n(losses, name='total_loss')

        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            loss_name = l.op.name
            tf.summary.scalar(loss_name + ' (raw)', l)
            tf.summary.scalar(loss_name, loss_averages.average(l))

        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)
        return total_loss

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def form_graph(self):
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        tower_grads = []
        tower_losses = []
        for i in range(self.num_gpu):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)) as scope:
                    loss = self.get_tower_loss(scope)
                    tf.get_variable_scope().reuse_variables()
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
                    tower_losses.append(loss)

        grads = self.average_gradients(tower_grads)
        self.opt_op = opt.apply_gradients(grads, global_step=self.global_step)
        self.ae_loss = tf.reduce_mean(tower_losses)
        self.representation = tf.get_variable('representation')

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('tensorboard/{}/train'.format(self.tensorboard_id), self.sess.graph)
        self.test_writer = tf.summary.FileWriter('tensorboard/{}/test'.format(self.tensorboard_id), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def train(self, input, labels=None):
        if self.mode == 'autoencode':
            _, batch_loss, global_step, tb_merge = self.sess.run([self.opt_op, self.ae_loss, self.global_step, self.merged],
                                                       feed_dict={self.input: input, self.dropout_rate: 0.5, self.train_time:True})
            self.train_writer.add_summary(tb_merge, global_step)
            return batch_loss, global_step

    def eval(self, input, labels=None):
        if self.mode == 'autoencode':
            loss, tb_merge, global_step = self.sess.run([self.ae_loss, self.merged, self.global_step],
                                                        feed_dict={self.input:input, self.dropout_rate:0.0, self.train_time:False})
            self.test_writer.add_summary(tb_merge, global_step)
            return loss

    def get_representation(self, input):
        return self.sess.run(self.representation, feed_dict={self.input:input, self.dropout_rate:0.0, self.train_time:False})

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