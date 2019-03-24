from __future__ import division
import os
import time
import datetime
from svhn_part2 import test_mnist
# import svhn_part
import tensorflow as tf
import numpy as np

class GAN(object):
    
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir, g_from_scratch):
        self.sess = sess
        self.g_from_scratch = g_from_scratch
        self.dataset_name = dataset_name
        self.mnist_images_test = np.load('mnist_rgb_images_test.npy')
        self.mnist_labels_test = np.load('mnist_rgb_labels_test.npy')
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_name = "GAN"     # name for checkpoint

        self.z_np = np.loadtxt('train_gen_encodings.txt', delimiter=',')
        self.input_np = np.load('mnist_rgb_images_train.npy')

        print("tttttttttttttttttt")
        print(np.sum(self.input_np[0][10][10]))

        
        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':

            # parameters
            self.input_to_disc_size = 500
            self.output_from_gen_size = 500

            self.input_to_gen_height = 32
            self.input_to_gen_width = 32

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 3

            # train
            # train
            self.learning_rate = 0.0001
            self.beta1 = 0.5

            # test
            self.sample_num = 64  

            print('=======================================')
            print(len(self.z_np))
            self.num_batches = len(self.z_np) // self.batch_size
        else:
            raise NotImplementedError

    def discriminator(self, x, is_training=True, reuse=False):

        with tf.variable_scope("discriminator", reuse=reuse):

            net = tf.nn.leaky_relu(tf.layers.dense(x, 500, kernel_initializer=tf.keras.initializers.he_normal(),
                                        trainable=is_training,
                                        name = 'hidden1'))

            net = tf.nn.leaky_relu(tf.layers.dense(net, 500, kernel_initializer=tf.keras.initializers.he_normal(),
                                        trainable=is_training,
                                        name='hidden2'))

            out_logit = tf.layers.dense(net, 2, kernel_initializer=tf.keras.initializers.he_normal(),
                                        trainable=is_training,
                                        name='out')

            return out_logit, out_logit, net

    def generator(self, z, is_training=True, reuse=False):

        is_training = True

        with tf.variable_scope("encoder", reuse=reuse):

            ##################################

            net = tf.layers.conv2d(inputs=z, filters=8, kernel_size=5, padding='valid', strides=1,
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   name='conv1', trainable=is_training)

            net = tf.nn.relu(net)

            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='VALID', name='maxpool1')

            ###############################

            net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=5, padding='VALID', strides=1,
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   name='conv2', trainable=is_training, reuse=reuse)

            net = tf.nn.relu(net)

            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='VALID', name='maxpool2')

            ##################################

            net = tf.layers.conv2d(inputs=net, filters=120, kernel_size=4, padding='VALID', strides=1,
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   name='conv3', trainable=is_training, reuse=reuse)

            net = tf.nn.dropout(tf.nn.relu(net), 0.5)

            #####################

            net = tf.contrib.layers.flatten(net)

            net = tf.nn.dropout(tf.layers.dense(inputs=net, units=500,
                                  kernel_initializer=tf.keras.initializers.he_normal(),
                                  activation=tf.nn.relu, name='fc1', trainable=is_training, reuse=reuse), 0.5)

            out = tf.reshape(net, [-1, 500])

            return out


    def build_model(self):
        image_dims = [self.input_to_gen_height, self.input_to_gen_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images (64,28,28,1)
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises (64,62)
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

      
        D_real, D_real_logits, _ = self.discriminator(self.z, is_training=True, reuse=False)

        self.G = self.generator(self.inputs, is_training=True, reuse=False)

        D_fake, D_fake_logits, _ = self.discriminator(self.G, is_training=True, reuse=True)

        real_labels = np.concatenate((np.ones([self.batch_size,1], dtype=np.float32), np.zeros([self.batch_size,1], dtype=np.float32)), axis=1)

        d_loss_real = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=D_real_logits, labels=real_labels))

        fake_labels = np.concatenate((np.zeros([self.batch_size,1], dtype=np.float32), np.ones([self.batch_size, 1], dtype=np.float32)), axis=1)

        d_loss_fake = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=D_fake_logits, labels=fake_labels))
        self.d_loss = d_loss_real + d_loss_fake

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=real_labels))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator/' in var.name]
        g_vars = [var for var in t_vars if 'encoder/' in var.name]


        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.RMSPropOptimizer(self.learning_rate) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.RMSPropOptimizer(self.learning_rate) \
                      .minimize(self.g_loss, var_list=g_vars)

            # self.d_optim = tf.train.AdamOptimizer(self.learning_rate) \
            #           .minimize(self.d_loss, var_list=d_vars)
            #
            # self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 3) \
            #           .minimize(self.g_loss, var_list=g_vars)

        self.fake_images = self.generator(self.inputs, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)


        hist_d_vars = []
        hist_g_vars = []

        for var in d_vars:
            hist_d_vars.append(tf.summary.histogram(var.name, var))

        for var in g_vars:
            hist_g_vars.append(tf.summary.histogram(var.name, var))

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum] + hist_g_vars)
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum] + hist_d_vars)

        self.saver = tf.train.Saver(max_to_keep=5)

    def train(self):
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name + datetime.datetime.now().isoformat(), self.sess.graph)

        # restore check-point if it exits
        if self.g_from_scratch == 'True':
            could_load, checkpoint_counter = self.load_new(self.checkpoint_dir)

        else:
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            start_epoch = int(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.input_np[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_z = self.z_np[idx*self.batch_size:(idx+1)*self.batch_size].astype(np.float32)

                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict={self.inputs: batch_images})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                if np.mod(counter, 50) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss= %.8f, g_loss= %.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

            self.save(self.checkpoint_dir, counter)
            test_mnist(self.mnist_images_test, self.mnist_labels_test, 256, self, self.sess)
    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                    '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        print(ckpt)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


    def load_gen(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        print(ckpt)

        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder/')

        for var in g_vars:
            print(var, '\n')

        saver_load_gen = tf.train.Saver(g_vars)

        if ckpt and ckpt.model_checkpoint_path:
            self.sess.run(tf.global_variables_initializer())
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver_load_gen.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))

            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_new(self, checkpoint_dir):

        print(" [*] Reading checkpoints...")
        g_checkpoint_dir = os.path.join('checkpoints_svhn')

        g_ckpt = tf.train.get_checkpoint_state(g_checkpoint_dir)

        print(g_ckpt)

        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder/')

        for var in g_vars:
            print(var, '\n')

        saver_load_gen = tf.train.Saver(g_vars)

        self.sess.run(tf.global_variables_initializer())

        if g_ckpt and g_ckpt.model_checkpoint_path:
            print(" [*] Reading generator checkpoint...")
            ckpt_name = os.path.basename(g_ckpt.model_checkpoint_path)
            saver_load_gen.restore(self.sess, os.path.join(g_checkpoint_dir, ckpt_name))
            print(" [*] Success to read generator {}".format(ckpt_name))
            return True, 0
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

        # self.sess.run(tf.global_variables_initializer())
        # return False, 0
