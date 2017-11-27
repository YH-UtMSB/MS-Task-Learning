from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

DATA_PATH = '/data2/MNIST/MNIST_tfrecord/'
TRAIN_PATH = DATA_PATH + 'TrainSet/Rot/mnist.tfrecords'
VALID_PATH = DATA_PATH + 'TestSet/Rot/mnist.tfrecords'

BATCH_SIZE = 20
NUM_BATCH = 10000
SUMRY_SAVE_PATH = './MyOEN/logs/Sumry/mnist_DAdp/exp15'
MODEL_SAVE_PATH = './MyOEN/logs/Model/mnist_DAdp/exp15'
MODEL_NAME = 'model.ckpt'


"""
Predefine all necessary layer for the AlexNet
"""


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels / groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, name, relu=True, regularizer=None):
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(weights))

        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def get_input():
    FiledirList = tf.train.match_filenames_once(VALID_PATH)

    filename_queue = tf.train.string_input_producer(FiledirList, shuffle=False)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'theta': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
        }
    )

    decoded_image = tf.decode_raw(features['image'], tf.uint8)
    reshaped_image = tf.reshape(decoded_image, [227, 227, 3])
    retyped_image = tf.image.convert_image_dtype(reshaped_image, tf.float32)
    retyped_image = tf.subtract(retyped_image, 0.5)
    retyped_image = tf.multiply(retyped_image, 2.0)
    theta = tf.cast(features['theta'], tf.float32)
    label = tf.cast(features['label'], tf.float32)

    image_and_label = [retyped_image, theta, label]

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * BATCH_SIZE

    image_batch, theta_batch, label_batch = tf.train.batch(
        image_and_label,
        batch_size=BATCH_SIZE,
        capacity=capacity
    )

    return image_batch, theta_batch, label_batch


def get_theta(sin_cos_batch):
    """
    This function is designed to obtain theta from [sin(theta), cos(theta)] pairs
    with straightforward calculations and without cumbersome judgments.
    :param sin_cos_batch: an ndarray with shape [batch_size, 2]
    :return theta_batch: a ndarray consisting of thetas corresponding to sin_cos_batch. 
    """
    sin_cos_mat = np.mat(sin_cos_batch)
    print('type of sin_cos_batch',type(sin_cos_batch))
    print('shape of sin_cos_batch',sin_cos_batch.shape,' shape of np.mat(sin_cos_batch)',sin_cos_mat.shape)
    A = np.mat([[-0.5j, 0.5], [0.5j, 0.5]])
    exp_mat = sin_cos_mat * A.I
    exp_vec = exp_mat.T.tolist()[0]

    # calculation of half_theta via sin_cos pair
    exp_vec = np.array(exp_vec)
    exp_vec = exp_vec ** 0.5
    half_theta_batch = np.arctan2(exp_vec.imag, exp_vec.real)
    theta_batch = half_theta_batch * 2 * 180.0 / np.pi

    return theta_batch

def SummarySave(thetaLabel_batch, thetaLogit_batch, Label_batch):
    Dev = 0.0
    zipped_batch = zip(thetaLabel_batch, thetaLogit_batch, Label_batch)
    batch_size = float(len(zipped_batch))
    Below15 = 0
    label_stat = []

    for index, zipped in enumerate(zipped_batch):
        (theta_label, theta_logit, Label) = zipped

        if(theta_label > 180.0):
            theta_label = theta_label - 360.0
        dev = np.abs(theta_label - theta_logit)
        if(dev > 180.0):
            dev = np.abs(360.0 - dev)

        Dev += dev

        if(dev < 15.0):
            Below15 += 1
        else:
            label_stat.append(Label)


    return Dev/batch_size, Below15, label_stat


def inference(images, regularizer=None):
    # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
    conv1 = conv(images, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
    pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
    norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

    # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
    conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
    pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
    norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened_r = tf.reshape(pool5, [BATCH_SIZE, -1])
    dim_r = flattened_r.get_shape()[1].value
    fc6_r = fc(flattened_r, dim_r, 4096, name='fc6_r', regularizer=regularizer)
    dropout6_r = dropout(fc6_r, 1.0)

    # 7th Layer: FC (w ReLu) -> Dropout
    fc7_r = fc(dropout6_r, 4096, 4096, name='fc7_r', regularizer=regularizer)
    dropout7_r = dropout(fc7_r, 1.0)

    fc8_r = fc(dropout7_r, 4096, 2, relu=False, name='fc8_r')
    normalized_logits = tf.nn.l2_normalize(fc8_r, 1, name='l2_normalize')

    return normalized_logits


def main():
    with tf.Graph().as_default():
        image_batch, theta_batch, label_batch = get_input()

        with tf.variable_scope("OEN") as scope:
            logit_batch = inference(image_batch)

        saver = tf.train.Saver()

        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)

            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
                saver.restore(sess, ckpt.model_checkpoint_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            Dev_total = 0.0
            BL15 = 0.0
            Stat = []

            for step in range(NUM_BATCH):
                thetaLable_batch, cur_logit_batch, cur_label_batch = sess.run(
                    [theta_batch, logit_batch, label_batch]
                )
                thetaLogit_batch = get_theta(cur_logit_batch)

                del_dev, dev_bl15, label_stat = SummarySave(
                    thetaLabel_batch=thetaLable_batch,
                    thetaLogit_batch=thetaLogit_batch,
                    Label_batch=cur_label_batch
                )
                Dev_total += del_dev
                BL15 += dev_bl15
                Stat += label_stat
                print('current total deviation:%.4f' % Dev_total, 'current num of BL15:%.4f' % BL15)

            num_of_batches = float(NUM_BATCH)
            Dev_total = float(Dev_total)
            hist = np.histogram(Stat, bins=10, range=(0.0,10.0))
            print('total deviation: %.4f' % (Dev_total / num_of_batches),'portion of BL15: %.4f' % (BL15 / (num_of_batches * BATCH_SIZE)))
            print('Histogram:', hist)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main()
