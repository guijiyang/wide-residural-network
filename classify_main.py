# !/usr/bin/python3
# -*- coding:utf-8 -*-

import logging
import os
import sys
import numpy as np
import tensorflow as tf
import argparse
import data_utils
import classify_model
import itertools
import cifar10_input
import functools
import six


def get_model_fn(num_gpus, variable_strategy, num_workers):
    '''返回一个函数构建一个残差网络模型(WRN model)'''

    def _resnet_model_fn(features, labels, mode, params):
        '''残差网络模型主干
        参数：features表示一个输入张量列表；
        labels表示一个目标张量列表；
        mode表示tf.estimator.ModeKeys.TRAIN还是EVAL
        params表示超参数
        返回：一个EstimatorSpec对象
        '''
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        wide_num = params['wide_num']
        weight_decay = params['weight_decay']
        momentum = params['momentum']
        num_layers = params['num_layers']
        num_classes = params['classified_num']
        tower_features = features
        tower_labels = labels

        tower_losses = []
        tower_gradvars = []
        tower_preds = []

        # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
        # on CPU. The exception is Intel MKL on CPU which is optimal with
        # channels_last.
        data_format = params['data_format']
        if not data_format:
            if num_gpus == 0:
                data_format = 'channels_last'
            else:
                data_format = 'channels_first'

        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'

        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if variable_strategy == 'CPU':
                device_setter = data_utils.local_device_setter(
                    worker_device=worker_device)
            elif variable_strategy == 'GPU':
                ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(
                    num_gpus,
                    tf.contrib.training.byte_size_load_fn)
                device_setter = data_utils.local_device_setter(
                    ps_device_type='gpu',
                    worker_device=worker_device,
                    ps_strategy=ps_strategy)
            with tf.variable_scope('resnet', reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        loss, gradvars, preds = _tower_fn(
                            is_training=is_training,
                            wide_num=wide_num,
                            weight_decay=weight_decay,
                            feature=tower_features[i],
                            label=tower_labels[i],
                            data_format=data_format,
                            num_layers=num_layers,
                            batch_norm_decay=params['batch_norm_decay'],
                            batch_norm_epsilon=params['batch_norm_epsilon'],
                            dropout=params['dropout'],
                            relu_leakiness=params['relu_leakiness'],
                            num_classes=num_classes)
                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                            # 只有第一个tower的批标准化的移动平均值和方差需要更新。理想情况下，我们需要获取所有Tower的更新
                            # 但是他们的状态累积的十分迅速所以我们可以忽略其他的没有决定意义的状态
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                           name_scope)

        # 现在计算全局损失和梯度
        gradvars = []
        with tf.name_scope('gradient_aeraging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
                # 同一个设备上的应用的参数进行平均梯度计算
            for var, grads in six.iteritems(all_grads):
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1./len(grads))
                gradvars.append((avg_grad, var))

        # 设备上运行操作去更新全局梯度
        consolidation_device = '/gpu:0' if variable_strategy == "GPU" else '/cpu:0'
        with tf.device(consolidation_device):
            # 建议的学习率计划源自https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py#L155
            num_batches_per_epoch = cifar10_input.Cifar10Dataset.num_examples_per_epoch(
                'train')//(params['train_batch_size']*num_workers)
            boundaries = [
                num_batches_per_epoch*x for x in np.array(params['decay_epoch'], dtype=np.int64)
            ]
            # boundaries = [40000, 60000, 80000]
            staged_lr = [params['learning_rate']
                         * x for x in [1, params['lr_decay'], 0.01, 0.001]]
            learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
                                                        boundaries, staged_lr)
            loss = tf.reduce_mean(tower_losses, name='loss')
            examples_sec_hook = data_utils.ExamplesPerSecondHook(params['train_batch_size'],
                                                                 every_n_steps=10)
            tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}
            tf.summary.scalar('learning_rate', learning_rate)
            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=100)
            train_hooks = [logging_hook, examples_sec_hook]

            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=momentum)

            if params['sync']:
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer, replicas_to_aggregate=num_workers)
                sync_replicas_hook = optimizer.make_session_run_hook(
                    params['is_chief'])
                train_hooks.append(sync_replicas_hook)

            # 创建一个train op group
            train_op = [
                optimizer.apply_gradients(
                    gradvars, global_step=tf.train.get_global_step()
                )
            ]

            train_op.extend(update_ops)
            train_op = tf.group(*train_op)

            predictions = {
                'classes': tf.concat([p['classes'] for p in tower_preds], axis=0),
                'probabilities':
                    tf.concat([p['probabilities']
                               for p in tower_preds], axis=0)
            }

            stacked_labels = tf.concat(labels, axis=0)
            metrics = {
                'accuracy':
                    tf.metrics.accuracy(stacked_labels, predictions['classes'])
            }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            training_hooks=train_hooks,
            eval_metric_ops=metrics)
    return _resnet_model_fn


def _tower_fn(is_training, wide_num, weight_decay, feature, label, data_format, num_layers,
              batch_norm_decay, batch_norm_epsilon, dropout, relu_leakiness, num_classes):
    """建立计算层次
    参数：is_training-如果是训练图则为True
    wide_num-输出放在倍数，用来加强网络的宽度
    weight_decay-权重正则化强度，一个浮点值
    feature-输入的特征张量
    label-目标张量
    data_format-输入张量格式(channels_first\channels_last)
    num_layers-神经网络层数
    batch_norm_decay-批处理的衰减，浮点值
    batch_norm_epsilon：批处理的一个分母项，浮点值，防止分母项为0
    num_classes:目标的分类数
    返回：一个存放计算层次的损失、梯度、参数和预测的元组
    """
    model = classify_model.WRNModel(
        num_layers=num_layers,
        wide_factor=wide_num,
        is_training=is_training,
        data_format=data_format,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        dropout=dropout,
        relu_leakiness=relu_leakiness,
        num_classes=num_classes)
    logits = model.forward_pass(feature, input_data_format='channels_last')
    tower_pred = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits)
    }

    tower_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=label, logits=logits)
    tower_loss = tf.reduce_mean(tower_loss)

    model_params = tf.trainable_variables()
    tower_loss += weight_decay * \
        tf.add_n([tf.nn.l2_loss(v) for v in model_params])
    tower_grad = tf.gradients(tower_loss, model_params)

    return tower_loss, zip(tower_grad, model_params), tower_pred


def create_log(num_layers, width):
    tf.logging.set_verbosity(tf.logging.INFO)
    formater = logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s', '%m-%d %H:%M:%S')
    # log_path = os.path.join('./log/', os.path.basename(__file__)+'.log')
    log_path = os.path.join(os.getcwd(), './log/WRD_' +
                            str(num_layers)+'_'+str(width)+'.log')
    handlers = [
        logging.FileHandler(log_path)
        # logging.StreamHandler(sys.stdout)
    ]
    handlers[0].setFormatter(formater)
    logging.getLogger('tensorflow').handlers = handlers


def input_fn(subset,
             num_shards,
             batch_size,
             epochs,
             use_distortion=True):
    """
    构建输入模型的图
    参数：data_dir-存放TFRecords的路径
    subset-'train'，'validate'，'eval'三种模式中的一种
    num_shards-数据并行时候的Tower数量
    batch_size-总的批尺寸用于训练，需要除于num_shards
    use_distortion-是否使用打乱操作
    返回：两列特征和标签张量，每一个都是num_shards的长度
    """
    with tf.device('/cpu:0'):
        use_distortion = subset == 'train' and use_distortion
        dataset = cifar10_input.Cifar10Dataset(subset, use_distortion)
        image_batch, label_batch = dataset.make_batch(batch_size, epochs)
        if num_shards <= 1:
            return [image_batch], [label_batch]
        image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
        label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
        feature_shards = [[] for i in range(num_shards)]
        lbael_shards = [[] for i in range(num_shards)]
        for i in range(batch_size):
            indx = i % num_shards
            feature_shards[indx].append(image_batch[i])
            lbael_shards[indx].append(label_batch[i])
        feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        lbael_shards = [tf.parallel_stack(x) for x in lbael_shards]
        return feature_shards, lbael_shards


def main(checkpoint_dir, variable_strategy, num_gpus,
         use_distortion_for_training, log_device_placement,
         num_intra_threads, **hparams):
    # 环境变量在一个过时的路径上，默认设置为off
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # 会话配置
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=log_device_placement,
                                 intra_op_parallelism_threads=num_intra_threads,
                                 gpu_options=tf.GPUOptions(force_gpu_compatible=True,per_process_gpu_memory_fraction=0.1))

    config = data_utils.RunConfig(
        session_config=sess_config, model_dir=checkpoint_dir)
    estimator = tf.estimator.Estimator(model_fn=get_model_fn(num_gpus, variable_strategy, 1),
                                       config=config, params=hparams)

    train_input_fn = functools.partial(input_fn,
                                       subset='train',
                                       num_shards=num_gpus,
                                       batch_size=hparams['train_batch_size'],
                                       use_distortion=use_distortion_for_training,
                                       epochs=hparams['train_epochs'])
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=None)
    eval_input_fn = functools.partial(input_fn,
                                      subset='eval',
                                      num_shards=num_gpus,
                                      batch_size=hparams['eval_batch_size'],
                                      epochs=hparams['eval_epochs'])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, steps=None)
    tf.estimator.train_and_evaluate(
        estimator, train_spec=train_spec, eval_spec=eval_spec)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--data-dir',
    #     type=str,
    #     default='./dataset',
    #     help='The directory where the input data is stored.')
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoint/wrn_28_10',
        help='The directory where the checkpoint will be stored.')
    parser.add_argument(
        '--variable-strategy',
        choices=['CPU', 'GPU'],
        type=str, default='GPU',
        help='Where to locate variable operations')
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='The number of gpus used. Uses only CPU if set to 0.')
    parser.add_argument(
        '--data-format',
        type=str,
        default=None,
        help="""\
            If not set, the data format best for the training device is used.
            Allowed values: channels_first (NCHW) channels_last (NHWC).\
            """)
    parser.add_argument(
        '--num-layers',
        type=int,
        default=28,
        help='The number of residual unit used')
    parser.add_argument(
        '--wide-num',
        type=int,
        default=10,
        help='The number used for convolution channels augment.')
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.7,
        help='The ratio dropout of residual unit between convolution layers.')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for MomentumOptimizer.')
    parser.add_argument(
        '--train-epochs',
        type=int,
        default=300,
        help='The number of epochs to use for training.')
    parser.add_argument(
        '--eval-epochs',
        type=int,
        default=1,
        help='The number of epochs to use for validation.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help="""\
            This is the inital learning rate value. The learning rate will decrease
            during training. For more details check the model_fn implementation in
            this file.\
            """)
    parser.add_argument(
        '--lr-decay',
        type=float,
        default=0.1,
        help='The learning rate decay ratio by specified epochs')
    parser.add_argument(
        '--decay-epoch',
        type=tuple,
        default=(60, 140, 220),
        help='The epoch num when to decay learning rate')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=2e-4,
        help='Weight decay for convolutions.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=128,
        help='Batch size for training.')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=100,
        help='Batch size for validation.')
    parser.add_argument(
        '--classified-num',
        type=int,
        default=10,
        help='The number of species of data')
    parser.add_argument(
        '--batch-norm-decay',
        type=float,
        default=0.99,
        help='The momentum for batch normalization moving average')
    parser.add_argument(
        '--batch-norm-epsilon',
        type=float,
        default=1e-3,
        help="The small value add to batch normalization' variance to avoid dividing by zero")
    parser.add_argument(
        '--sync',
        action='store_true',
        default=False,
        help="""\
            If present when running in a distributed environment will run on sync mode.\
            """)
    parser.add_argument(
        '--num-intra-threads',
        type=int,
        default=0,
        help="""\
            Number of threads to use for intra-op parallelism. When training on CPU
            set to 0 to have the system pick the appropriate number or alternatively
            set it to the number of physical CPU cores.\
            """)
    parser.add_argument(
        '--num-inter-threads',
        type=int,
        default=0,
        help="""\
        Number of threads to use for inter-op parallelism. If set to 0, the
        system will pick an appropriate number.\
        """)
    parser.add_argument(
        '--use-distortion-for-training',
        type=bool,
        default=True,
        help='If doing image distortion for training.')
    parser.add_argument(
        '--log-device-placement',
        action='store_true',
        default=False,
        help='Whether to log device placement.')
    parser.add_argument(
        '--relu-leakiness',
        type=float,
        default=0.1,
        help='the ratio use for relu unactive position')
    # parser.add_argument(
    #     '--train-steps',
    #     type=int,
    #     default=80000,
    #     help='The number of steps to use for training.')
    args = parser.parse_args()
    create_log(args.num_layers, args.wide_num)
    if args.num_gpus > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if args.num_gpus == 0 and args.variable_strategy == 'GPU':
        raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                         '--variable-strategy=CPU.')
    if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
        raise ValueError('--train-batch-size must be multiple of --num-gpus.')
    if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
        raise ValueError('--eval-batch-size must be multiple of --num-gpus.')
    main(**vars(args))
