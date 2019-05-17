# !/usr/bin/python3
# -*- coding:utf-8 -*-

'''Model class for classify'''

import tensorflow as tf
import model_base


class WRNModel(model_base.ResNet):
    '''宽度残差网络模型'''

    def __init__(self,
                 num_layers,
                 wide_factor,
                 is_training,
                 batch_norm_decay,
                 batch_norm_epsilon,
                 dropout,
                 num_classes,
                 relu_leakiness,
                 data_format='channels_first'):
        hps = model_base.HParams(
            is_training, data_format, batch_norm_decay, batch_norm_epsilon, dropout, relu_leakiness)
        super(WRNModel, self).__init__(hps)
        self._num_classes = num_classes+1
        self._residual_unit_nums = (num_layers-4)//6
        self._wide_factor = wide_factor
        self._filters = [16, 16, 32, 64]
        self._strides = [1, 2, 2]

    def forward_pass(self, x, input_data_format='channels_last'):
        '''在图中构建核心模型'''
        if self.hps.data_format != input_data_format:
            if input_data_format == 'channels_last':
                # Computation requires channels_first.
                x = tf.transpose(x, [0, 3, 1, 2])
            else:
                # Computation requires channels_last.
                x = tf.transpose(x, [0, 2, 3, 1])

        # 图片归一化
        x = x/128-1
        # 第一次通过卷积层，输出图片32*32
        x = self._conv(name='init_conv', x=x, kernel_size=3,
                       filters=self._filters[0], strides=1)
        # x=self._batch_norm(name='first_bn',x=x)
        # x=self._relu(x=x)

        # 使用宽度残差单元
        res_func = self._wide_residual

        # 3种不同的过滤单元
        for i in range(3):
            with tf.name_scope('stage'):
                for j in range(self._residual_unit_nums):
                    with tf.variable_scope('residual_{}_{}'.format(i, j)):
                        if j == 0:
                            x = res_func(x=x,
                                         kernel_size=3,
                                         in_filter=self._filters[i],
                                         out_filter=self._filters[i+1]*self._wide_factor,
                                         stride=self._strides[i])
                        else:
                            x = res_func(x=x,
                                         kernel_size=3,
                                         in_filter=self._filters[i+1]*self._wide_factor,
                                         out_filter=self._filters[i+1]*self._wide_factor,
                                         stride=1)

        x = self._batch_norm(x=x, name='final_bn')
        x = self._relu(x=x, leakiness=self.hps.relu_leakiness,
                       name='final_relu')
        x = self._global_avg_pool(x)
        x = self._fully_connected(x, self._num_classes)
        return x
