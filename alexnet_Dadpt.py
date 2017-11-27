from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from gradient_synthesizer import GradSys

TRAIN_PATH = '/data1/SubImagenet/tfrecords/train/train.tfrecords'

BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.0001
REGULARIZATION_RATE = 0.00005
LEARING_RATE_DECAY = 0.96
SUMRY_SAVE_PATH = './MyOEN/logs/Sumry/imnet_DAdp/exp2'
MODEL_SAVE_PATH = './MyOEN/logs/Model/imnet_DAdp/exp2'
MODEL_NAME = 'model.ckpt'
TRAINING_STEPS = 1000001

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
    FiledirList = tf.train.match_filenames_once(TRAIN_PATH)

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
    sin_cos = tf.stack([tf.sin(theta * np.pi / 180.0), tf.cos(theta * np.pi / 180.0)])
    label = tf.cast(features['label'], tf.int32)

    image_and_label = [retyped_image, label, sin_cos]

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * BATCH_SIZE

    image_batch, label_batch, sin_cos_batch = tf.train.shuffle_batch(
        image_and_label,
        batch_size=BATCH_SIZE,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return image_batch, label_batch, sin_cos_batch


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

    features = GradSys(pool5, l1=1.0, l2=0.0)
    fr, fd = tf.unstack(features)

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened_r = tf.reshape(fr, [BATCH_SIZE, -1])
    dim_r = flattened_r.get_shape()[1].value
    fc6_r = fc(flattened_r, dim_r, 4096, name='fc6_r', regularizer=regularizer)
    dropout6_r = dropout(fc6_r, 1.0)

    flattened_d = tf.reshape(fd, [BATCH_SIZE, -1])
    dim_d = flattened_d.get_shape()[1].value
    fc6_d = fc(flattened_d, dim_d, 4096, name='fc6_d', regularizer=regularizer)
    dropout6_d = dropout(fc6_d, 1.0)

    # 7th Layer: FC (w ReLu) -> Dropout
    fc7_r = fc(dropout6_r, 4096, 4096, name='fc7_r', regularizer=regularizer)
    dropout7_r = dropout(fc7_r, 1.0)

    fc7_d = fc(dropout6_d, 4096, 4096, name='fc7_d', regularizer=regularizer)
    dropout7_d = dropout(fc7_d, 1.0)

    fc8_r = fc(dropout7_r, 4096, 2, relu=False, name='fc8_r')
    fc8_d = fc(dropout7_d, 4096, 81, relu=False, name='fc8_d')

    class_prediction = fc8_d
    normalized_logits = tf.nn.l2_normalize(fc8_r, 1, name='l2_normalize')

    return normalized_logits, class_prediction


def get_loss(logits, labels):
    loss = tf.nn.l2_loss(logits - labels)
    return loss

def training(loss):
    global_step = tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0), trainable=False
    )

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        2700000 / BATCH_SIZE,
        LEARING_RATE_DECAY,
        staircase=True
    )

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def main():
    with tf.Graph().as_default():
        image_batch, label_batch, sin_cos_batch = get_input()
        #regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

        with tf.variable_scope("OEN") as scope:
            logits_reg, logits_cls = inference(image_batch, regularizer=None)

        loss_train = get_loss(
           logits=logits_reg, labels=sin_cos_batch
        )
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_cls, labels=label_batch)
        loss_class = tf.reduce_mean(cross_entropy)

        #l2_regular = tf.add_n(tf.get_collection('losses'))
        loss = loss_train + loss_class

        train_op = training(loss)
        tf.summary.scalar('loss', loss_train)
        tf.summary.scalar('cls_loss', loss_class)
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            global_step = 0

            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                global_step = int(global_step)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_writer = tf.summary.FileWriter(SUMRY_SAVE_PATH, sess.graph)

            for step in range(global_step, TRAINING_STEPS + 1):
                _, loss_value_train, loss_value_class, summary = sess.run([train_op, loss_train, loss_class, summary_op])

                if step % 100 == 0:
                    #r = l2_regular_value / loss_value_train
                    print('step %d: loss = %.4f, cls_loss = %.4f' % (step, loss_value_train, loss_value_class))
                    summary_writer.add_summary(summary, step)

                if step % 5000 == 0:
                    checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

        summary_writer.close()


if __name__ == '__main__':
    main()
