# 基于宽度神经网络的cifar10图片分类

这个代码是我基于文章[Wide residural network](https://arxiv.org/pdf/1605.07146.pdf)实现的宽度神经网络结构模型，在cifar10数据集上测试得到95%分类准确率。


## 运行环境
* python3.6
* tensorflow1.12

## 运行方式

```python
python classify_main.py --data-dir='数据路径' --checkpoint-dir='模型保存路径'
```

## 超参数调节
可以指定以下参数：
```python
variable-strategy: 以CPU或GPU进行模型的训练
num-gpus：制定用于训练的GPU的数量
data-format : 'channels_first' 或者 'channels_last'
num-layers : 网络总层数
wide-num ： 宽化系数，乘以参差单元的基础过滤器个数
dropout： 0~1之间
momentum ： 用于MomentumOptimizer的参数，一般去0.9或者0.99
train-epochs，eval-epochs：训练和评估次数
learning-rate： 学习率
lr-decay：指定学习率递减系数
decay-epoch：递减的位置
weight-decay：权重系数衰减
train-batch-size： mini-batch
eval-batch-size：不用管
classified-num：数据集的种类数，cifar10默认10
batch-norm-decay，batch-norm-epsilon：用于batch-norm的系数


