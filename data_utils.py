# !/usr/bin/python3
# -*- coding:utf-8 -*-
import six
import collections
import tensorflow as tf
import logging
from tensorflow.python.training import device_setter
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.framework import device as pydev
from tensorflow.core.framework import node_def_pb2
from tensorflow.contrib.learn.python.learn import run_config


class RunConfig(tf.estimator.RunConfig):
    def uid(self, whitelist=None):
        '''基于所有的内部空间生成一个唯一的标识符
        调用这需要在一个会话中使用uid字符串去检查RunConfig实例的完整性，但是不能依赖于实现细节，那会受到变化的影响
        参数：whitelist-一列关于属性uid的字符串名不应该被包含
        返回：一个uid字符串
        '''
        if whitelist is None:
            whitelist = run_config._DEFAULT_UID_WHITE_LIST

        state = {k: v for k, v in self.__dict__.items()
                 if not k.startswith('__')}
        # 弹出whitelist中的key
        for k in whitelist:
            state.pop('_'+k, None)

        ordered_state = collections.OrderedDict(
            sorted(state.items(), key=lambda t: t[0]))
        # 对类实例没有__repr__,一些特殊的关照是需要的。
        # 另外，对象地址将被用到。
        if '_cluster_spec' in ordered_state:
            ordered_state['_cluster_spec'] = collections.OrderedDict(
                sorted(ordered_state['_cluster_spec'].as_dict().items(),
                       key=lambda t: t[0]))
        return ', '.join('%s=%r'%(k,v) for (k,v) in six.iteritems(ordered_state))


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
    '''钩子用于每秒打印我们的样本
    总时间被追踪然后被除于总步数用于或缺平均每步的时间
    batch_size用于决定每秒运行的样本平均值。
    最近的间隙每秒的样本也被记录到logging
    '''

    def __init__(self, batch_size, every_n_steps=100, every_n_secs=None,):
        '''ExamplesPerSecondHook构造器
        参数：batch_size-总的批尺寸用于从全局时间中计算样本/秒
        every_n_steps-打印log每every_n_steps步
        every_n_secs-打印log每every_n_secs秒
        '''
        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('exactly one of every_n_steps'
                             ' and every_n_secs should be provided.')
        self._timer = basic_session_run_hooks.SecondOrStepTimer(
            every_steps=every_n_steps, every_secs=every_n_secs)

        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size

    def begin(self):
        self._global_step_tensor = training_util.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                'Global step should be created to use StepCounterHook.'
            )

    def before_run(self, run_context):
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            elapased_time, elapsed_steps = self._timer.update_last_triggered_step(
                global_step)
            if elapased_time is not None:
                steps_per_sec = elapsed_steps/elapased_time
                self._step_train_time += elapased_time
                self._total_steps += elapsed_steps

                average_examples_per_sec = self._batch_size*(
                    self._total_steps/self._step_train_time)
                current_examples_per_sec = steps_per_sec*self._batch_size
                # 平均样本数/秒跟随这当前样本数/秒
                logging.info('%s: %g (%g), step=%g', 'Average examples/sec',
                             average_examples_per_sec, current_examples_per_sec, self._total_steps)


def local_device_setter(num_devices=1, ps_device_type='cpu', worker_device='/cpu:0', ps_ops=None, ps_strategy=None):
    if ps_ops == None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))
            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(
                worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()
    return _local_device_chooser


if __name__ == "__main__":
    local_device_setter(worker_device='/cpu:1')
