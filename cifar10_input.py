# !/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3


class Cifar10Dataset(object):
    '''cifar10数据集'''

    def __init__(self, subset='train', use_distortion=True):
        self.subset = subset
        self.use_distortion = use_distortion

    def get_filenames(self):
        if self.subset in ['train', 'validation', 'eval']:
            file_path = tf.keras.utils.get_file(
                'cifar_tfrecords', origin='https://github.com/guijiyang/wide-residural-network/releases/download/v1.0.0/cifar10_tfrecords.tar.gz',
                md5_hash='37e85d853d3c02501063940e687b6dd521ba8e31b1bdf4f902dc3c6d4f210242',
                extract=True)
            return [os.path.join(file_path, self.subset+'.tfrecords')]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
            tf.float32)
        label = tf.cast(features['label'], tf.int32)

        # Custom preprocessing.
        image = self.preprocess(image)

        return image, label

    def make_batch(self, batch_size, epochs):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames).repeat(epochs)

        # Parse records.
        dataset = dataset.map(
            self.parser, num_parallel_calls=batch_size)

        # Potentially shuffle records.
        if self.subset == 'train':
            min_queue_examples = int(
                self.num_examples_per_epoch(self.subset) * 0.4)
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
            dataset = dataset.shuffle(
                buffer_size=min_queue_examples + 3 * batch_size)

        # Batch it up.
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()

        return image_batch, label_batch

    def preprocess(self, image):
        """Preprocess a single image in [height, width, depth] layout."""
        if self.subset == 'train' and self.use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            image = tf.image.resize_image_with_crop_or_pad(
                image, HEIGHT+4, WIDTH+4)
            image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
            image = tf.image.random_flip_left_right(image)
            image = tf.image.per_image_standardization(image)
        else:
            image = tf.image.resize_image_with_crop_or_pad(
                image, HEIGHT, WIDTH)
            image = tf.image.per_image_standardization(image)
        return image

    # @staticmethod
    def num_examples_per_epoch(self, subset='train'):
        if subset == 'train':
            return 45000
        elif subset == 'validation':
            return 5000
        elif subset == 'eval':
            return 10000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)
