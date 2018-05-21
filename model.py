#! /usr/bin/python
# -*- coding: utf8 -*-
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os

"""Adversarially Learned Inference
Page 14: CelebA model hyperparameters
Optimizer Adam (α = 10−4, β1 = 0.5)
Batch size 100 Epochs 123
Leaky ReLU slope 0.02
Weight, bias initialization Isotropic gaussian (µ = 0, σ = 0.01), Constant(0)
"""
batch_size = 64

z_dim = 512  # Noise dimension
image_size = 64  # 64 x 64
c_dim = 3  # for rgb

t_dim = 128  # text feature dimension
rnn_hidden_size = t_dim
vocab_size = 8000
word_embedding_size = 256
keep_prob = 1.0


def rnn_embed(input_seqs, is_train=True, reuse=False):
    """ txt --> t_dim """
    w_init = tf.random_normal_initializer(stddev=0.02)
    if tf.__version__ <= '0.12.1':
        LSTMCell = tf.nn.rnn_cell.LSTMCell
    else:
        LSTMCell = tf.contrib.rnn.BasicLSTMCell
    with tf.variable_scope("rnnftxt", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = EmbeddingInputlayer(
            inputs=input_seqs,
            vocabulary_size=vocab_size,
            embedding_size=word_embedding_size,
            E_init=w_init,
            name='rnn/wordembed')
        network = DynamicRNNLayer(network,
                                  cell_fn=LSTMCell,
                                  cell_init_args={'state_is_tuple': True, 'reuse': reuse},
                                  # for TF1.1, TF1.2 dont need to set reuse
                                  n_hidden=rnn_hidden_size,
                                  dropout=(keep_prob,keep_prob),
                                  initializer=w_init,
                                  sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
                                  return_last=True,
                                  name='rnn/dynamic')
        return network


def cnn_encoder(inputs, reuse=False, name='cnnftxt'):
    """ 64x64 --> t_dim, for text-image mapping """
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64

    with tf.variable_scope(name, reuse=reuse):
        net_input = inputs
        net_h0 = tf.layers.conv2d(net_input, df_dim, (4, 4), (2, 2), padding='same', activation=tf.nn.leaky_relu,
                                  kernel_initializer=w_init, use_bias=False)

        net_h1 = tf.layers.conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                  use_bias=False)
        net_h1 = tf.layers.batch_normalization(net_h1, gamma_initializer=gamma_init, training=True)
        net_h1 = tf.nn.leaky_relu(net_h1)

        net_h2 = tf.layers.conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                  use_bias=False)
        net_h2 = tf.layers.batch_normalization(net_h2, gamma_initializer=gamma_init, training=True)
        net_h2 = tf.nn.leaky_relu(net_h2)

        net_h3 = tf.layers.conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                  use_bias=False)
        net_h3 = tf.layers.batch_normalization(net_h3, gamma_initializer=gamma_init, training=True)
        net_h3 = tf.nn.leaky_relu(net_h3)

        net_h4 = tf.layers.flatten(net_h3)
        net_h4 = tf.layers.dense(net_h4, units=t_dim, activation=tf.identity, kernel_initializer=w_init, use_bias=False)

        return net_h4


def residualblcok(input, w_init, gamma_init):
    net_input = input
    net_h0 = tf.layers.conv2d(inputs=net_input,
                              filters=512,
                              kernel_size=3,
                              padding='same',
                              use_bias=False,
                              kernel_initializer=w_init)
    net_h0 = tf.layers.batch_normalization(net_h0, gamma_initializer=gamma_init, training=True)
    net_h0 = tf.nn.relu(net_h0)

    net_h1 = tf.layers.conv2d(inputs=net_h0,
                              filters=512,
                              kernel_size=3,
                              padding='same',
                              use_bias=False,
                              kernel_initializer=w_init)
    net_h1 = tf.layers.batch_normalization(net_h1, gamma_initializer=gamma_init, training=True)
    net_h2 = tf.nn.relu(input + net_h1)
    return net_h2


def generator_imgtxt2img(input_img, input_rnn_emned, reuse=False):
    s = image_size
    s2, s4 = int(s / 2), int(s / 4)

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    gf_dim = 128

    with tf.variable_scope("generator", reuse=reuse):
        net_in = input_img
        ##encoder
        net_h0 = tf.layers.conv2d(net_in, gf_dim, (3, 3), (1, 1), padding='same', kernel_initializer=w_init,
                                  activation=tf.nn.relu)

        net_h1 = tf.layers.conv2d(net_h0, gf_dim * 2, (4, 4), (2, 2), padding='same', kernel_initializer=w_init)
        net_h1 = tf.layers.batch_normalization(net_h1, gamma_initializer=gamma_init, training=True)
        net_h1 = tf.nn.relu(net_h1)

        net_h2 = tf.layers.conv2d(net_h1, gf_dim * 4, (4, 4), (2, 2), padding='same', kernel_initializer=w_init)
        net_h2 = tf.layers.batch_normalization(net_h2, gamma_initializer=gamma_init, training=True)
        net_h2 = tf.nn.relu(net_h2)

        # ca+concat
        net_txt = input_rnn_emned
        net_txt = tf.layers.dense(net_txt, units=128, activation=tf.nn.leaky_relu, kernel_initializer=w_init,
                                  use_bias=False)
        net_txt = tf.expand_dims(net_txt, 1)
        net_txt = tf.expand_dims(net_txt, 1)
        net_txt = tf.tile(net_txt, [1, 16, 16, 1])

        net_h2_concat = tf.concat([net_txt, net_h2], 3)
        fusion = tf.layers.conv2d(net_h2_concat, gf_dim * 4, (3, 3), (1, 1), padding='same', kernel_initializer=w_init,
                                  use_bias=False)
        fusion = tf.layers.batch_normalization(fusion, gamma_initializer=gamma_init, training=True)
        fusion = residualblcok(fusion, w_init, gamma_init)
        fusion = residualblcok(fusion, w_init, gamma_init)
        fusion = residualblcok(fusion, w_init, gamma_init)
        fusion = residualblcok(fusion, w_init, gamma_init)

        # decoder
        net_input = fusion
        net_h0 = tf.layers.conv2d_transpose(net_input, gf_dim * 2, (4, 4), (2, 2), padding='same',
                                            kernel_initializer=w_init, use_bias=False)
        net_h0 = tf.layers.batch_normalization(net_h0, gamma_initializer=gamma_init, training=True)
        net_h0 = tf.nn.relu(net_h0)

        net_h1 = tf.layers.conv2d_transpose(net_h0, gf_dim, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                            use_bias=False)
        net_h1 = tf.layers.batch_normalization(net_h1, gamma_initializer=gamma_init, training=True)
        net_h1 = tf.nn.relu(net_h1)

        net_h2 = tf.layers.conv2d_transpose(net_h1, c_dim, (4, 4), (2, 2), padding='same', kernel_initializer=w_init)
        logits = net_h2
        net_h2 = tf.nn.tanh(net_h2)
        output = net_h2

        return output, logits


def discriminator_imgtxt2img(input_images, input_rnn_embed, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64

    with tf.variable_scope("discriminator", reuse=reuse):
        net_in = input_images
        net_h0 = tf.layers.conv2d(net_in, df_dim, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                  activation=tf.nn.leaky_relu)

        net_h1 = tf.layers.conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                  use_bias=False)
        net_h1 = tf.layers.batch_normalization(net_h1, gamma_initializer=gamma_init, training=True)
        net_h1 = tf.nn.leaky_relu(net_h1)

        net_h2 = tf.layers.conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                  use_bias=False)
        net_h2 = tf.layers.batch_normalization(net_h2, gamma_initializer=gamma_init, training=True)
        net_h2 = tf.nn.leaky_relu(net_h2)

        net_h3 = tf.layers.conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                  use_bias=False)

        net_h3 = tf.layers.batch_normalization(net_h3, gamma_initializer=gamma_init, training=True)

        net_txt = input_rnn_embed
        net_txt = tf.layers.dense(net_txt, units=t_dim, activation=tf.nn.leaky_relu, kernel_initializer=w_init,
                                  use_bias=False)
        net_txt = tf.expand_dims(net_txt, 1)
        net_txt = tf.expand_dims(net_txt, 1)
        net_txt = tf.tile(net_txt, [1, 4, 4, 1])

        net_h3_concat = tf.concat([net_txt, net_h3], 3)

        net_h3 = tf.layers.conv2d(net_h3_concat, df_dim * 8, (1, 1), (1, 1), padding='same', kernel_initializer=w_init,
                                  use_bias=False)
        net_h3 = tf.layers.batch_normalization(net_h3, gamma_initializer=gamma_init, training=True)
        net_h3 = tf.nn.leaky_relu(net_h3)

        net_h4 = tf.layers.flatten(net_h3)
        net_h4 = tf.layers.dense(net_h4, units=1, activation=tf.identity, kernel_initializer=w_init)

        logits = net_h4
        net_h4 = tf.nn.sigmoid(net_h4)
    return net_h4, logits


## simple g1, d1 ===============================================================
def generator_txt2img_simple(input_z, input_rnn_embed, reuse=False):
    """ z + (txt) --> 64x64 """
    s = image_size
    s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    gf_dim = 128

    with tf.variable_scope("generator", reuse=reuse):
        net_in = input_z

        net_txt = input_rnn_embed
        net_txt = tf.layers.dense(net_txt, units=t_dim, activation=tf.nn.leaky_relu, kernel_initializer=w_init,
                                  use_bias=False)
        net_in = tf.concat([net_in, net_txt], 1)

        net_h0 = tf.layers.dense(net_in, gf_dim * 8 * s16 * s16, activation=tf.identity, kernel_initializer=w_init,
                                 use_bias=False)
        net_h0 = tf.reshape(net_h0, [-1, s16, s16, gf_dim * 8])
        net_h0 = tf.layers.batch_normalization(net_h0, gamma_initializer=gamma_init, training=True)
        net_h0 = tf.nn.relu(net_h0)

        net_h1 = tf.layers.conv2d_transpose(net_h0, gf_dim * 4, (4, 4), (2, 2), padding='same',
                                            kernel_initializer=w_init, use_bias=False)
        net_h1 = tf.layers.batch_normalization(net_h1, gamma_initializer=gamma_init, training=True)
        net_h1 = tf.nn.relu(net_h1)

        net_h2 = tf.layers.conv2d_transpose(net_h1, gf_dim * 2, (4, 4), (2, 2), padding='same',
                                            kernel_initializer=w_init, use_bias=False)
        net_h2 = tf.layers.batch_normalization(net_h2, gamma_initializer=gamma_init, training=True)
        net_h2 = tf.nn.relu(net_h2)

        net_h3 = tf.layers.conv2d_transpose(net_h2, gf_dim, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                            use_bias=False)
        net_h3 = tf.layers.batch_normalization(net_h3, gamma_initializer=gamma_init, training=True)
        net_h3 = tf.nn.relu(net_h3)

        net_h4 = tf.layers.conv2d_transpose(net_h3, c_dim, (4, 4), (2, 2), padding='same', kernel_initializer=w_init)
        logits = net_h4
        net_h4 = tf.nn.tanh(net_h4)
    return net_h4, logits


def discriminator_txt2img_simple(input_images, input_rnn_embed=None, is_train=True, reuse=False):
    """ 64x64 + (txt) --> real/fake """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64

    with tf.variable_scope("discriminator", reuse=reuse):
        net_in = input_images
        net_h0 = tf.layers.conv2d(net_in, df_dim, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                  activation=tf.nn.leaky_relu)

        net_h1 = tf.layers.conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                  use_bias=False)
        net_h1 = tf.layers.batch_normalization(net_h1, gamma_initializer=gamma_init, training=True)
        net_h1 = tf.nn.leaky_relu(net_h1)

        net_h2 = tf.layers.conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                  use_bias=False)
        net_h2 = tf.layers.batch_normalization(net_h2, gamma_initializer=gamma_init, training=True)
        net_h2 = tf.nn.leaky_relu(net_h2)

        net_h3 = tf.layers.conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), padding='same', kernel_initializer=w_init,
                                  use_bias=False)

        net_h3 = tf.layers.batch_normalization(net_h3, gamma_initializer=gamma_init, training=True)

        net_txt = input_rnn_embed
        net_txt = tf.layers.dense(net_txt, units=t_dim, activation=tf.nn.leaky_relu, kernel_initializer=w_init,
                                  use_bias=False)
        net_txt = tf.expand_dims(net_txt, 1)
        net_txt = tf.expand_dims(net_txt, 1)
        net_txt = tf.tile(net_txt, [1, 4, 4, 1])

        net_h3_concat = tf.concat([net_txt, net_h3], 3)

        net_h3 = tf.layers.conv2d(net_h3_concat, df_dim * 8, (1, 1), (1, 1), padding='same', kernel_initializer=w_init,
                                  use_bias=False)
        net_h3 = tf.layers.batch_normalization(net_h3, gamma_initializer=gamma_init, training=True)
        net_h3 = tf.nn.leaky_relu(net_h3)

        net_h4 = tf.layers.flatten(net_h3)
        net_h4 = tf.layers.dense(net_h4, units=1, activation=tf.identity, kernel_initializer=w_init)

        logits = net_h4
        net_h4 = tf.nn.sigmoid(net_h4)
    return net_h4, logits


## default g1, d1 ==============================================================
def generator_txt2img_resnet(input_z, t_txt=None, is_train=True, reuse=False, batch_size=batch_size):
    """ z + (txt) --> 64x64 """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    s = image_size  # output image size [64]
    s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
    gf_dim = 128

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_z, name='g_inputz')

        if t_txt is not None:
            net_txt = InputLayer(t_txt, name='g_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim,
                                 act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init, name='g_reduce_text/dense')
            net_in = ConcatLayer([net_in, net_txt], concat_dim=1, name='g_concat_z_txt')

        net_h0 = DenseLayer(net_in, gf_dim * 8 * s16 * s16, act=tf.identity,
                            W_init=w_init, b_init=None, name='g_h0/dense')
        net_h0 = BatchNormLayer(net_h0,  # act=tf.nn.relu,
                                is_train=is_train, gamma_init=gamma_init, name='g_h0/batch_norm')
        net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim * 8], name='g_h0/reshape')

        net = Conv2d(net_h0, gf_dim * 2, (1, 1), (1, 1),
                     padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                             gamma_init=gamma_init, name='g_h1_res/batch_norm')
        net = Conv2d(net, gf_dim * 2, (3, 3), (1, 1),
                     padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d2')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                             gamma_init=gamma_init, name='g_h1_res/batch_norm2')
        net = Conv2d(net, gf_dim * 8, (3, 3), (1, 1),
                     padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d3')
        net = BatchNormLayer(net,  # act=tf.nn.relu,
                             is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm3')
        net_h1 = ElementwiseLayer(layers=[net_h0, net], combine_fn=tf.add, name='g_h1_res/add')
        net_h1.outputs = tf.nn.relu(net_h1.outputs)

        # Note: you can also use DeConv2d to replace UpSampling2dLayer and Conv2d
        # net_h2 = DeConv2d(net_h1, gf_dim*4, (4, 4), out_size=(s8, s8), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h2/decon2d')
        net_h2 = UpSampling2dLayer(net_h1, size=[s8, s8], is_scale=False, method=1,
                                   align_corners=False, name='g_h2/upsample2d')
        net_h2 = Conv2d(net_h2, gf_dim * 4, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2,  # act=tf.nn.relu,
                                is_train=is_train, gamma_init=gamma_init, name='g_h2/batch_norm')

        net = Conv2d(net_h2, gf_dim, (1, 1), (1, 1),
                     padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                             gamma_init=gamma_init, name='g_h3_res/batch_norm')
        net = Conv2d(net, gf_dim, (3, 3), (1, 1),
                     padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d2')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                             gamma_init=gamma_init, name='g_h3_res/batch_norm2')
        net = Conv2d(net, gf_dim * 4, (3, 3), (1, 1),
                     padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d3')
        net = BatchNormLayer(net,  # act=tf.nn.relu,
                             is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm3')
        net_h3 = ElementwiseLayer(layers=[net_h2, net], combine_fn=tf.add, name='g_h3/add')
        net_h3.outputs = tf.nn.relu(net_h3.outputs)

        # net_h4 = DeConv2d(net_h3, gf_dim*2, (4, 4), out_size=(s4, s4), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h4/decon2d'),
        net_h4 = UpSampling2dLayer(net_h3, size=[s4, s4], is_scale=False, method=1,
                                   align_corners=False, name='g_h4/upsample2d')
        net_h4 = Conv2d(net_h4, gf_dim * 2, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,
                                is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')

        # net_h5 = DeConv2d(net_h4, gf_dim, (4, 4), out_size=(s2, s2), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h5/decon2d')
        net_h5 = UpSampling2dLayer(net_h4, size=[s2, s2], is_scale=False, method=1,
                                   align_corners=False, name='g_h5/upsample2d')
        net_h5 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu,
                                is_train=is_train, gamma_init=gamma_init, name='g_h5/batch_norm')

        # net_ho = DeConv2d(net_h5, c_dim, (4, 4), out_size=(s, s), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_ho/decon2d')
        net_ho = UpSampling2dLayer(net_h5, size=[s, s], is_scale=False, method=1,
                                   align_corners=False, name='g_ho/upsample2d')
        net_ho = Conv2d(net_ho, c_dim, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, name='g_ho/conv2d')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)
    return net_ho, logits


def discriminator_txt2img_resnet(input_images, t_txt=None, is_train=True, reuse=False):
    """ 64x64 + (txt) --> real/fake """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    # Discriminator with ResNet : line 197 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64  # 64 for flower, 196 for MSCOCO
    s = 64  # output image size [64]
    s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='d_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                        padding='SAME', W_init=w_init, name='d_h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=None, name='d_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm')
        net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=None, name='d_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')
        net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=None, name='d_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3,  # act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm')

        net = Conv2d(net_h3, df_dim * 2, (1, 1), (1, 1), act=None,
                     padding='VALID', W_init=w_init, b_init=None, name='d_h4_res/conv2d')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                             is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm')
        net = Conv2d(net, df_dim * 2, (3, 3), (1, 1), act=None,
                     padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d2')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                             is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm2')
        net = Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=None,
                     padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d3')
        net = BatchNormLayer(net,  # act=lambda x: tl.act.lrelu(x, 0.2),
                             is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm3')
        net_h4 = ElementwiseLayer(layers=[net_h3, net], combine_fn=tf.add, name='d_h4/add')
        net_h4.outputs = tl.act.lrelu(net_h4.outputs, 0.2)

        if t_txt is not None:
            net_txt = InputLayer(t_txt, name='d_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim,
                                 act=lambda x: tl.act.lrelu(x, 0.2),
                                 W_init=w_init, name='d_reduce_txt/dense')
            net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim2')
            net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='d_txt/tile')
            net_h4_concat = ConcatLayer([net_h4, net_txt], concat_dim=3, name='d_h3_concat')
            # 243 (ndf*8 + 128 or 256) x 4 x 4
            net_h4 = Conv2d(net_h4_concat, df_dim * 8, (1, 1), (1, 1),
                            padding='VALID', W_init=w_init, b_init=None, name='d_h3/conv2d_2')
            net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                                    is_train=is_train, gamma_init=gamma_init, name='d_h3/batch_norm_2')

        net_ho = Conv2d(net_h4, 1, (s16, s16), (s16, s16), padding='VALID', W_init=w_init, name='d_ho/conv2d')
        # 1 x 1 x 1
        # net_ho = FlattenLayer(net_h4, name='d_ho/flatten')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
    return net_ho, logits
