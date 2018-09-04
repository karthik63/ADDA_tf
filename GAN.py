#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import datetime
from svhn_part2 import test_mnist
# import svhn_part
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

class GAN(object):
    #初始化各类定义
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

        #mnist和fashion-mnist都是由28x28的灰色图像组成的
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
            self.sample_num = 64  # number of generated images to be saved,保存图片的数量

            # load mnist, data_X为mnist数据图片归一化结果，data_y为加上标签的数组记录

            # get number of batches for a single epoch 在单个批次下处理的数据长度，此处为70000整除64=1093

            print('=======================================')
            print(len(self.z_np))
            self.num_batches = len(self.z_np) // self.batch_size
        else:
            raise NotImplementedError

    #送入鉴别器的输入为(64,28,28,1)
    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S

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

        # with tf.variable_scope("discriminator", reuse=reuse):
        #     # 经过这一步卷积后，(64,28,28,1)-->(64,14,14,64)
        #     net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
        #     # 经过这一步卷积后，(64,14,14,64)-->(64,7,7,128)
        #     net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))#数据标准化
        #     #经过数组重构后，(64,7,7,128)-->(64,6272)
        #     net = tf.reshape(net, [self.batch_size, -1])
        #     #经过线性处理后将矩阵，(64,6272)-->(64,1024)
        #     net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
        #     #经过线性处理后将矩阵，(64,1024)-->(64,1)
        #     out_logit = linear(net, 1, scope='d_fc4')
        #     #将数据处理在（0,1）之间
        #     out = tf.nn.sigmoid(out_logit)
        #
        #     return out, out_logit, net


    # 送入生成器的输入噪声z为(64,62)
    def generator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S



            # net = tf.layers.conv2d(inputs=z, filters=64, kernel_size=3, padding='SAME', strides=1,
            #                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            #                         bias_initializer=tf.constant_initializer(0.0),
            #                         name='conv1', trainable=is_training)
            #
            # net = tf.layers.batch_normalization(net, scale=True, trainable=is_training, name='bn1')
            #
            # net = tf.nn.relu(net)
            #
            # net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='VALID', name='maxpool1')
            #
            # net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=3, padding='SAME', strides=1,
            #                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            #                         bias_initializer=tf.constant_initializer(0.0),
            #                         name='conv2', trainable=is_training)
            #
            # net = tf.layers.batch_normalization(net, scale=True, trainable=is_training, name='bn2')
            #
            # net = tf.nn.relu(net)
            #
            # net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='VALID', name='maxpool2')
            #
            # net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=3, padding='SAME', strides=1,
            #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            #                        bias_initializer=tf.constant_initializer(0.0),
            #                        name='conv3', trainable=is_training)
            #
            # net = tf.layers.batch_normalization(net, scale=True, trainable=is_training, name='bn3')
            #
            # net = tf.nn.relu(net)
            #
            # net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='VALID', name='maxpool3')
            #
            # net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=3, padding='SAME', strides=1,
            #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            #                        bias_initializer=tf.constant_initializer(0.0),
            #                        name='conv4', trainable=is_training)
            #
            # net = tf.layers.batch_normalization(net, scale=True, trainable=is_training, name='bn4')
            #
            # net = tf.nn.relu(net)
            #
            # net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='VALID', name='maxpool4')

        # with tf.variable_scope("encoder", reuse=reuse):
        #     # 经过这一步卷积后，(64,28,28,1)-->(64,14,14,64)
        #     net = lrelu(conv2d(z, 64, 4, 4, 2, 2, name='conv1'))
        #     # 经过这一步卷积后，(64,14,14,64)-->(64,7,7,128)
        #     net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='conv2'), is_training=is_training, scope='bn2'))#数据标准化
        #     #经过数组重构后，(64,7,7,128)-->(64,6272)
        #     net = tf.reshape(net, [self.batch_size, -1])
        #     #经过线性处理后将矩阵，(64,6272)-->(64,1024)
        #     net = lrelu(bn(linear(net, 1024, scope='fc3'), is_training=is_training, scope='bn3'))
        #     #经过线性处理后将矩阵，(64,1024)-->(64,64)
        #     out_logit = linear(net, 500, scope='fc4')
        #     #将数据处理在（0,1）之间
        #     out = tf.nn.relu(out_logit)
        #
        #     return out

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

        # with tf.variable_scope("generator", reuse=reuse):
        #     # 经过线性处理后将矩阵，(64,62)-->(64,1024)
        #     net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
        #     # 经过线性处理后将矩阵，(64,1024)-->(64,6272), 6272=128*7*7
        #     net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
        #     # 经过重构后，形状变为(64,7,7,128)
        #     net = tf.reshape(net, [self.batch_size, 7, 7, 128])
        #     # 经过deconv2d,(64,7,7,128)-->(64,14,14,128)
        #     net = tf.nn.relu(
        #         bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
        #            scope='g_bn3'))
        #     # 经过deconv2d,(64,14,14,128)-->(64,28,28,1),将值处理用sigmoid处理至（0,1）之间,`y = 1 / (1 + exp(-x))`
        #     out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))
        #
        #     return out

    #建立GAN模型，此函数非常重要
    def build_model(self):
        # some parameters
        # 对于mnist数据集，图片大小为（28,28,1），此处用list列表存储
        image_dims = [self.input_to_gen_height, self.input_to_gen_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images (64,28,28,1)
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises (64,62)
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        # output of D for real images, D_real((64,1),介于(0,1)),D_real_logits未经历过sigmoid，_临时存储net(64,1024)
        D_real, D_real_logits, _ = self.discriminator(self.z, is_training=True, reuse=False)

        # output of D for fake images G为由噪声z（64,62）生成的图片数据(64,28,28,1)
        self.G = self.generator(self.inputs, is_training=True, reuse=False)

        #D_fake((64,1),介于(0,1)),D_fake_logits未经历过sigmoid，_临时存储net(64,1024),送入鉴别器的是G生成的假的数据
        D_fake, D_fake_logits, _ = self.discriminator(self.G, is_training=True, reuse=True)

        # get loss for discriminator
        #它对于输入的logits先通过sigmoid函数计算，再计算它们的交叉熵，但是它对交叉熵的计算方式进行了优化，使得结果不至于溢出
        #tf.ones_like的使用默认交叉商前面的系数为1数组
        #d_loss_real=-log(sigmoid(D_real_logits))等价于d_loss_real=-log(D(x))

        real_labels = np.concatenate((np.ones([self.batch_size,1], dtype=np.float32), np.zeros([self.batch_size,1], dtype=np.float32)), axis=1)

        d_loss_real = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=D_real_logits, labels=real_labels))
        # d_loss_fake=-log(sigmoid(D_fake_logits))等价于d_loss_fake=-log(D(G(z))

        fake_labels = np.concatenate((np.zeros([self.batch_size,1], dtype=np.float32), np.ones([self.batch_size, 1], dtype=np.float32)), axis=1)

        d_loss_fake = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=D_fake_logits, labels=fake_labels))
        #d_loss为生成器和鉴别器传出的loss之和
        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        #g_loss=-log(sigmoid(D_fake_logits))等价于g_loss=-log(D(G(z))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=real_labels))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator/' in var.name]
        g_vars = [var for var in t_vars if 'encoder/' in var.name]

        # optimizers 优化器用于减小损失函数loss，采用Adam优化器，可直接用minimize


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

        """" Testing """
        # for test 由噪声生成一张图片
        self.fake_images = self.generator(self.inputs, is_training=False, reuse=True)

        """ Summary """
        #Summary的含义是将参数打包后用于tensorboard的观察和模型的记录
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

    #最为重要的一个函数，控制着GAN模型的训练
    def train(self):

        # initialize all variables初始化各个变量

        # graph inputs for visualize training results
        #创造噪声z,GAN中应用的为均值分布，创造(64,62)大小的-1到1之间的
        #self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        # saver to save model 将训练好的模型参数保存起来

        # summary writer 将训练记录在log下
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

            # get batch data
            # 由于batchsize为64，遍历70000张图片需要1093次



            for idx in range(start_batch_id, self.num_batches):
                #提取处理好的固定位置图片，data_X的按批次处理后的图片位置，一个批次64张图片
                batch_images = self.input_np[idx*self.batch_size:(idx+1)*self.batch_size]

                #构造均匀分布的噪声z
                batch_z = self.z_np[idx*self.batch_size:(idx+1)*self.batch_size].astype(np.float32)

                # update D network sess.run喂入数据优化更新D网络，并在tensorboard中更新
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network sess.run喂入数据优化更新G网络，并在tensorboard中更新
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict={self.inputs: batch_images})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                #训练一个batchsize打印一下loss，一个epoch打印1093次我认为没这个必要,50次batchsize后打印一下
                if np.mod(counter, 50) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss= %.8f, g_loss= %.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

            self.save(self.checkpoint_dir, counter)
            test_mnist(self.mnist_images_test, self.mnist_labels_test, 256, self, self.sess)
            #     # save training results for every 300 steps 训练,300步保存一张图片
            #     # if np.mod(counter, 300) == 0:
            #     #     #生成一张该阶段下的由生成器生成的“假图片”
            #     #     samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
            #     #     #此处计算生成图片的小框图片的排布，本处为8×8排布
            #     #     tot_num_samples = min(self.sample_num, self.batch_size)
            #     #     manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
            #     #     manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
            #     #     save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
            #     #                 './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
            #     #                 '_train_{:02d}_{:04d}.png'.format(epoch, idx))
            #
            # # After an epoch, start_batch_id is set to zero 经过一个epoch后start_batch_id置为0
            # # non-zero value is only for the first epoch after loading pre-trained model
            # start_batch_id = 0
            #
            # # save model


            # show temporal results 经过一个epoch后输出一张全类别的图片，用于比较epoch后的差别
            # self.visualize_results(epoch)

        # save model for final step 当epoch全部训练完后保存checkpoint


    #用于可视化epoch后输出图片
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
    #加载创建固定模型下的路径，本处为GAN下的训练
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    #本函数的目的是在于保存训练模型后的checkpoint
    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name), global_step=step)

    #本函数的意义在于读取训练好的模型参数的checkpoint
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