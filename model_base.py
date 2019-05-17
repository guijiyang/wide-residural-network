# !/usr/bin/python3
# -*- coding:utf-8 -*-

'''ResNet模型.
相关文献：
https://arxiv.org/pdf/1605.07146v1.pdf
'''

import tensorflow as tf
from collections import namedtuple

HParams = namedtuple('HParams',
                     'is_training,data_format, batch_norm_decay, batch_norm_epsilon,dropout,relu_leakiness')


class ResNet(object):
    '''ResNet模型'''

    def __init__(self, hps):
        '''ResNet构造器
        hps-构造模型所需的超参数
        '''
        self.hps = hps
        # self._extra_train_ops = []

    def _conv(self, name, x, kernel_size, filters, strides, is_atrous=False):
        '''卷积'''
        with tf.variable_scope(name):
            padding = 'same'
            if not is_atrous and strides > 1:
                pad = kernel_size-1
                pad_beg = pad//2
                pad_end = pad-pad_beg
                if self.hps.data_format == 'channels_first':
                    x = tf.pad(
                        x, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
                else:
                    x = tf.pad(x, [[0, 0], [pad_beg, pad_end],
                                   [pad_beg, pad_end], [0, 0]])
                padding = 'valid'
            return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides,
                                    padding=padding, use_bias=False, data_format=self.hps.data_format)

    def _batch_norm(self, x, name='batch_norm'):
        '''使用Batch Normalization，注意需要手动更新moving_mean和moving_variance'''
        if self.hps.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        return tf.layers.batch_normalization(
            inputs=x,
            axis=channel_axis,
            momentum=self.hps.batch_norm_decay,
            epsilon=self.hps.batch_norm_epsilon,
            training=self.hps.is_training,
            fused=True,
            name=name)

    def _relu(self, x, leakiness=0.0, name='leaky_relu'):
        '''Relu，支持额外的leaky relu'''
        return tf.where(tf.less(x, 0.0), leakiness*x, x, name=name)

    def forward_pass(self, x):
        raise NotImplementedError(
            'forward_pass() is implemented in ResNet sub classes')

    def _fully_connected(self, x, out_dim):
        '''全链接层，用于最终的输出'''
        with tf.name_scope('fully_connected') as name_scope:
            x = tf.layers.dense(x, out_dim)
        tf.logging.info('经过%s 的图像尺寸 %s', name_scope, x.get_shape())
        return x

    def _global_avg_pool(self, x):
        '''全局平均池化
        参数：x表示输入张量
        返回：输出张量，尺寸是(x.get_shape()[0],x.get_shape()[-1])
        '''
        with tf.name_scope('global_avg_pool') as name_scope:
            assert x.get_shape().ndims == 4
            if self.hps.data_format == 'channels_first':
                x = tf.reduce_mean(x, [2, 3])
            else:
                x = tf.reduce_mean(x, [1, 2])

        tf.logging.info('经过%s 的图像尺寸 %s', name_scope, x.get_shape())
        return x

    def _avg_pool(self, x, pool_size, stride):
        '''平均池化
        参数：x表示输入张量
        pool_size表示池化窗口大小(pool_width,pool_height)
        stride表示步长
        返回：输入张量池化后的张量
        '''
        with tf.name_scope('avg_pool') as name_scope:
            x = tf.layers.average_pooling2d(
                x, pool_size=pool_size, strides=stride)
        tf.logging.info('经过%s 的图像尺寸 %s', name_scope, x.get_shape())
        return x

    def _wide_residual(self,
                       x,
                       kernel_size,
                       in_filter,
                       out_filter,
                       stride,
                       activate_before_residual=False):
        '''宽度残差网络的基础构造器'''
        with tf.name_scope('wide_residual') as name_scope:
            if activate_before_residual:
                with tf.variable_scope('shared_activation'):
                    x = self._batch_norm(name='init_bn', x=x)
                    x = self._relu(x, leakiness=self.hps.relu_leakiness)
                    orig_x = x
            else:
                orig_x = x
                x = self._batch_norm(name='init_bn', x=x)
                x = self._relu(x, leakiness=self.hps.relu_leakiness)

            with tf.variable_scope('sub1'):
                x = self._conv('conv1', x, kernel_size,
                               out_filter, stride, is_atrous=True)

            with tf.variable_scope('sub2'):
                x = self._batch_norm(name='bn2', x=x)
                x = self._relu(x, leakiness=self.hps.relu_leakiness)
                dropout = tf.nn.dropout(x, self.hps.dropout)
                x = self._conv('conv2', dropout, kernel_size,
                               out_filter, strides=1, is_atrous=True)

            if in_filter != out_filter:
                orig_x = self._conv('conv_size', orig_x, 1,
                                    out_filter, strides=stride, is_atrous=True)
            # print(x.get_shape())
            # print(orig_x.get_shape())
            x = tf.add(x, orig_x)

            tf.logging.info('经过%s 的图像尺寸 %s', name_scope, x.get_shape())
            return x
