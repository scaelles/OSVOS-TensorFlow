from __future__ import print_function
"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
import sys
from datetime import datetime
import os
import scipy.misc
from PIL import Image
import six

slim = tf.contrib.slim


def osvos_arg_scope(weight_decay=0.0002):
    """Defines the OSVOS arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=None,
                        padding='SAME') as arg_sc:
        return arg_sc


def crop_features(feature, out_size):
    """Crop the center of a feature map
    Args:
    feature: Feature map to crop
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    """
    up_size = tf.shape(feature)
    ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
    ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
    slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
    # slice_input = tf.slice(feature, (0, ini_w, ini_w, 0), (-1, out_size[1], out_size[2], -1))  # Caffe cropping way
    return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])


def osvos(inputs, scope='osvos'):
    """Defines the OSVOS network
    Args:
    inputs: Tensorflow placeholder that contains the input image
    scope: Scope name for the network
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """
    im_size = tf.shape(inputs)

    with tf.variable_scope(scope, 'osvos', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs of all intermediate layers.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net_2 = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net_2, [2, 2], scope='pool2')
            net_3 = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net_3, [2, 2], scope='pool3')
            net_4 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net_4, [2, 2], scope='pool4')
            net_5 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

            # Get side outputs of the network
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None):
                side_2 = slim.conv2d(net_2, 16, [3, 3], scope='conv2_2_16')
                side_3 = slim.conv2d(net_3, 16, [3, 3], scope='conv3_3_16')
                side_4 = slim.conv2d(net_4, 16, [3, 3], scope='conv4_3_16')
                side_5 = slim.conv2d(net_5, 16, [3, 3], scope='conv5_3_16')

                # Supervise side outputs
                side_2_s = slim.conv2d(side_2, 1, [1, 1], scope='score-dsn_2')
                side_3_s = slim.conv2d(side_3, 1, [1, 1], scope='score-dsn_3')
                side_4_s = slim.conv2d(side_4, 1, [1, 1], scope='score-dsn_4')
                side_5_s = slim.conv2d(side_5, 1, [1, 1], scope='score-dsn_5')
                with slim.arg_scope([slim.convolution2d_transpose],
                                    activation_fn=None, biases_initializer=None, padding='VALID',
                                    outputs_collections=end_points_collection, trainable=False):
                    # Side outputs
                    side_2_s = slim.convolution2d_transpose(side_2_s, 1, 4, 2, scope='score-dsn_2-up')
                    side_2_s = crop_features(side_2_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/score-dsn_2-cr', side_2_s)
                    side_3_s = slim.convolution2d_transpose(side_3_s, 1, 8, 4, scope='score-dsn_3-up')
                    side_3_s = crop_features(side_3_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/score-dsn_3-cr', side_3_s)
                    side_4_s = slim.convolution2d_transpose(side_4_s, 1, 16, 8, scope='score-dsn_4-up')
                    side_4_s = crop_features(side_4_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/score-dsn_4-cr', side_4_s)
                    side_5_s = slim.convolution2d_transpose(side_5_s, 1, 32, 16, scope='score-dsn_5-up')
                    side_5_s = crop_features(side_5_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/score-dsn_5-cr', side_5_s)

                    # Main output
                    side_2_f = slim.convolution2d_transpose(side_2, 16, 4, 2, scope='score-multi2-up')
                    side_2_f = crop_features(side_2_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/side-multi2-cr', side_2_f)
                    side_3_f = slim.convolution2d_transpose(side_3, 16, 8, 4, scope='score-multi3-up')
                    side_3_f = crop_features(side_3_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/side-multi3-cr', side_3_f)
                    side_4_f = slim.convolution2d_transpose(side_4, 16, 16, 8, scope='score-multi4-up')
                    side_4_f = crop_features(side_4_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/side-multi4-cr', side_4_f)
                    side_5_f = slim.convolution2d_transpose(side_5, 16, 32, 16, scope='score-multi5-up')
                    side_5_f = crop_features(side_5_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'osvos/side-multi5-cr', side_5_f)
                concat_side = tf.concat([side_2_f, side_3_f, side_4_f, side_5_f], axis=3)

                net = slim.conv2d(concat_side, 1, [1, 1], scope='upscore-fuse')

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# Set deconvolutional layers to compute bilinear interpolation
def interp_surgery(variables):
    interp_tensors = []
    for v in variables:
        if '-up' in v.name:
            h, w, k, m = v.get_shape()
            tmp = np.zeros((m, k, h, w))
            if m != k:
                raise ValueError('input + output channels need to be the same')
            if h != w:
                raise ValueError('filters need to be square')
            up_filter = upsample_filt(int(h))
            tmp[range(m), range(k), :, :] = up_filter
            interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
    return interp_tensors


# TO DO: Move preprocessing into Tensorflow
def preprocess_img(image):
    """Preprocess the image to adapt it to network requirements
    Args:
    Image we want to input the network (W,H,3) numpy array
    Returns:
    Image ready to input the network (1,W,H,3)
    """
    if type(image) is not np.ndarray:
        image = np.array(Image.open(image), dtype=np.uint8)
    in_ = image[:, :, ::-1]
    in_ = np.subtract(in_, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
    # in_ = tf.subtract(tf.cast(in_, tf.float32), np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
    in_ = np.expand_dims(in_, axis=0)
    # in_ = tf.expand_dims(in_, 0)
    return in_


# TO DO: Move preprocessing into Tensorflow
def preprocess_labels(label):
    """Preprocess the labels to adapt them to the loss computation requirements
    Args:
    Label corresponding to the input image (W,H) numpy array
    Returns:
    Label ready to compute the loss (1,W,H,1)
    """
    if type(label) is not np.ndarray:
        label = np.array(Image.open(label).split()[0], dtype=np.uint8)
    max_mask = np.max(label) * 0.5
    label = np.greater(label, max_mask)
    label = np.expand_dims(np.expand_dims(label, axis=0), axis=3)
    # label = tf.cast(np.array(label), tf.float32)
    # max_mask = tf.multiply(tf.reduce_max(label), 0.5)
    # label = tf.cast(tf.greater(label, max_mask), tf.float32)
    # label = tf.expand_dims(tf.expand_dims(label, 0), 3)
    return label


def load_vgg_imagenet(ckpt_path):
    """Initialize the network parameters from the VGG-16 pre-trained model provided by TF-SLIM
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initializes the network
    """
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_corresp = dict()
    for v in var_to_shape_map:
        if "conv" in v:
            vars_corresp[v] = slim.get_model_variables(v.replace("vgg_16", "osvos"))[0]
    init_fn = slim.assign_from_checkpoint_fn(
        ckpt_path,
        vars_corresp)
    return init_fn


def class_balanced_cross_entropy_loss(output, label):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = tf.cast(tf.greater(label, 0.5), tf.float32)

    num_labels_pos = tf.reduce_sum(labels)
    num_labels_neg = tf.reduce_sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32)
    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))

    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    return final_loss


def class_balanced_cross_entropy_loss_theoretical(output, label):
    """Theoretical version of the class balanced cross entropy loss to train the network (Produces unstable results)
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """
    output = tf.nn.sigmoid(output)

    labels_pos = tf.cast(tf.greater(label, 0), tf.float32)
    labels_neg = tf.cast(tf.less(label, 1), tf.float32)

    num_labels_pos = tf.reduce_sum(labels_pos)
    num_labels_neg = tf.reduce_sum(labels_neg)
    num_total = num_labels_pos + num_labels_neg

    loss_pos = tf.reduce_sum(tf.multiply(labels_pos, tf.log(output + 0.00001)))
    loss_neg = tf.reduce_sum(tf.multiply(labels_neg, tf.log(1 - output + 0.00001)))

    final_loss = -num_labels_neg / num_total * loss_pos - num_labels_pos / num_total * loss_neg

    return final_loss


def load_caffe_weights(weights_path):
    """Initialize the network parameters from a .npy caffe weights file
    Args:
    Path to the .npy file containing the value of the network parameters
    Returns:
    Function that takes a session and initializes the network
    """
    osvos_weights = np.load(weights_path).item()
    vars_corresp = dict()
    vars_corresp['osvos/conv1/conv1_1/weights'] = osvos_weights['conv1_1_w']
    vars_corresp['osvos/conv1/conv1_1/biases'] = osvos_weights['conv1_1_b']
    vars_corresp['osvos/conv1/conv1_2/weights'] = osvos_weights['conv1_2_w']
    vars_corresp['osvos/conv1/conv1_2/biases'] = osvos_weights['conv1_2_b']

    vars_corresp['osvos/conv2/conv2_1/weights'] = osvos_weights['conv2_1_w']
    vars_corresp['osvos/conv2/conv2_1/biases'] = osvos_weights['conv2_1_b']
    vars_corresp['osvos/conv2/conv2_2/weights'] = osvos_weights['conv2_2_w']
    vars_corresp['osvos/conv2/conv2_2/biases'] = osvos_weights['conv2_2_b']

    vars_corresp['osvos/conv3/conv3_1/weights'] = osvos_weights['conv3_1_w']
    vars_corresp['osvos/conv3/conv3_1/biases'] = osvos_weights['conv3_1_b']
    vars_corresp['osvos/conv3/conv3_2/weights'] = osvos_weights['conv3_2_w']
    vars_corresp['osvos/conv3/conv3_2/biases'] = osvos_weights['conv3_2_b']
    vars_corresp['osvos/conv3/conv3_3/weights'] = osvos_weights['conv3_3_w']
    vars_corresp['osvos/conv3/conv3_3/biases'] = osvos_weights['conv3_3_b']

    vars_corresp['osvos/conv4/conv4_1/weights'] = osvos_weights['conv4_1_w']
    vars_corresp['osvos/conv4/conv4_1/biases'] = osvos_weights['conv4_1_b']
    vars_corresp['osvos/conv4/conv4_2/weights'] = osvos_weights['conv4_2_w']
    vars_corresp['osvos/conv4/conv4_2/biases'] = osvos_weights['conv4_2_b']
    vars_corresp['osvos/conv4/conv4_3/weights'] = osvos_weights['conv4_3_w']
    vars_corresp['osvos/conv4/conv4_3/biases'] = osvos_weights['conv4_3_b']

    vars_corresp['osvos/conv5/conv5_1/weights'] = osvos_weights['conv5_1_w']
    vars_corresp['osvos/conv5/conv5_1/biases'] = osvos_weights['conv5_1_b']
    vars_corresp['osvos/conv5/conv5_2/weights'] = osvos_weights['conv5_2_w']
    vars_corresp['osvos/conv5/conv5_2/biases'] = osvos_weights['conv5_2_b']
    vars_corresp['osvos/conv5/conv5_3/weights'] = osvos_weights['conv5_3_w']
    vars_corresp['osvos/conv5/conv5_3/biases'] = osvos_weights['conv5_3_b']

    vars_corresp['osvos/conv2_2_16/weights'] = osvos_weights['conv2_2_16_w']
    vars_corresp['osvos/conv2_2_16/biases'] = osvos_weights['conv2_2_16_b']
    vars_corresp['osvos/conv3_3_16/weights'] = osvos_weights['conv3_3_16_w']
    vars_corresp['osvos/conv3_3_16/biases'] = osvos_weights['conv3_3_16_b']
    vars_corresp['osvos/conv4_3_16/weights'] = osvos_weights['conv4_3_16_w']
    vars_corresp['osvos/conv4_3_16/biases'] = osvos_weights['conv4_3_16_b']
    vars_corresp['osvos/conv5_3_16/weights'] = osvos_weights['conv5_3_16_w']
    vars_corresp['osvos/conv5_3_16/biases'] = osvos_weights['conv5_3_16_b']

    vars_corresp['osvos/score-dsn_2/weights'] = osvos_weights['score-dsn_2_w']
    vars_corresp['osvos/score-dsn_2/biases'] = osvos_weights['score-dsn_2_b']
    vars_corresp['osvos/score-dsn_3/weights'] = osvos_weights['score-dsn_3_w']
    vars_corresp['osvos/score-dsn_3/biases'] = osvos_weights['score-dsn_3_b']
    vars_corresp['osvos/score-dsn_4/weights'] = osvos_weights['score-dsn_4_w']
    vars_corresp['osvos/score-dsn_4/biases'] = osvos_weights['score-dsn_4_b']
    vars_corresp['osvos/score-dsn_5/weights'] = osvos_weights['score-dsn_5_w']
    vars_corresp['osvos/score-dsn_5/biases'] = osvos_weights['score-dsn_5_b']

    vars_corresp['osvos/upscore-fuse/weights'] = osvos_weights['new-score-weighting_w']
    vars_corresp['osvos/upscore-fuse/biases'] = osvos_weights['new-score-weighting_b']
    return slim.assign_from_values_fn(vars_corresp)


def parameter_lr():
    """Specify the relative learning rate for every parameter. The final learning rate
    in every parameter will be the one defined here multiplied by the global one
    Args:
    Returns:
    Dictionary with the relative learning rate for every parameter
    """

    vars_corresp = dict()
    vars_corresp['osvos/conv1/conv1_1/weights'] = 1
    vars_corresp['osvos/conv1/conv1_1/biases'] = 2
    vars_corresp['osvos/conv1/conv1_2/weights'] = 1
    vars_corresp['osvos/conv1/conv1_2/biases'] = 2

    vars_corresp['osvos/conv2/conv2_1/weights'] = 1
    vars_corresp['osvos/conv2/conv2_1/biases'] = 2
    vars_corresp['osvos/conv2/conv2_2/weights'] = 1
    vars_corresp['osvos/conv2/conv2_2/biases'] = 2

    vars_corresp['osvos/conv3/conv3_1/weights'] = 1
    vars_corresp['osvos/conv3/conv3_1/biases'] = 2
    vars_corresp['osvos/conv3/conv3_2/weights'] = 1
    vars_corresp['osvos/conv3/conv3_2/biases'] = 2
    vars_corresp['osvos/conv3/conv3_3/weights'] = 1
    vars_corresp['osvos/conv3/conv3_3/biases'] = 2

    vars_corresp['osvos/conv4/conv4_1/weights'] = 1
    vars_corresp['osvos/conv4/conv4_1/biases'] = 2
    vars_corresp['osvos/conv4/conv4_2/weights'] = 1
    vars_corresp['osvos/conv4/conv4_2/biases'] = 2
    vars_corresp['osvos/conv4/conv4_3/weights'] = 1
    vars_corresp['osvos/conv4/conv4_3/biases'] = 2

    vars_corresp['osvos/conv5/conv5_1/weights'] = 1
    vars_corresp['osvos/conv5/conv5_1/biases'] = 2
    vars_corresp['osvos/conv5/conv5_2/weights'] = 1
    vars_corresp['osvos/conv5/conv5_2/biases'] = 2
    vars_corresp['osvos/conv5/conv5_3/weights'] = 1
    vars_corresp['osvos/conv5/conv5_3/biases'] = 2

    vars_corresp['osvos/conv2_2_16/weights'] = 1
    vars_corresp['osvos/conv2_2_16/biases'] = 2
    vars_corresp['osvos/conv3_3_16/weights'] = 1
    vars_corresp['osvos/conv3_3_16/biases'] = 2
    vars_corresp['osvos/conv4_3_16/weights'] = 1
    vars_corresp['osvos/conv4_3_16/biases'] = 2
    vars_corresp['osvos/conv5_3_16/weights'] = 1
    vars_corresp['osvos/conv5_3_16/biases'] = 2

    vars_corresp['osvos/score-dsn_2/weights'] = 0.1
    vars_corresp['osvos/score-dsn_2/biases'] = 0.2
    vars_corresp['osvos/score-dsn_3/weights'] = 0.1
    vars_corresp['osvos/score-dsn_3/biases'] = 0.2
    vars_corresp['osvos/score-dsn_4/weights'] = 0.1
    vars_corresp['osvos/score-dsn_4/biases'] = 0.2
    vars_corresp['osvos/score-dsn_5/weights'] = 0.1
    vars_corresp['osvos/score-dsn_5/biases'] = 0.2

    vars_corresp['osvos/upscore-fuse/weights'] = 0.01
    vars_corresp['osvos/upscore-fuse/biases'] = 0.02
    return vars_corresp


def _train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False, config=None, finetune=1,
           test_image_path=None, ckpt_name="osvos"):
    """Train OSVOS
    Args:
    dataset: Reference to a Dataset object instance
    initial_ckpt: Path to the checkpoint to initialize the network (May be parent network or pre-trained Imagenet)
    supervison: Level of the side outputs supervision: 1-Strong 2-Weak 3-No supervision
    learning_rate: Value for the learning rate. It can be a number or an instance to a learning rate object.
    logs_path: Path to store the checkpoints
    max_training_iters: Number of training iterations
    save_step: A checkpoint will be created every save_steps
    display_step: Information of the training will be displayed every display_steps
    global_step: Reference to a Variable that keeps track of the training steps
    iter_mean_grad: Number of gradient computations that are average before updating the weights
    batch_size: Size of the training batch
    momentum: Value of the momentum parameter for the Momentum optimizer
    resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False)
    config: Reference to a Configuration object used in the creation of a Session
    finetune: Use to select the type of training, 0 for the parent network and 1 for finetunning
    test_image_path: If image path provided, every save_step the result of the network with this image is stored
    Returns:
    """
    model_name = os.path.join(logs_path, ckpt_name+".ckpt")
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True

    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare the input data
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    input_label = tf.placeholder(tf.float32, [batch_size, None, None, 1])

    # Create the network
    with slim.arg_scope(osvos_arg_scope()):
        net, end_points = osvos(input_image)

    # Initialize weights from pre-trained model
    if finetune == 0:
        init_weights = load_vgg_imagenet(initial_ckpt)

    # Define loss
    with tf.name_scope('losses'):
        if supervison == 1 or supervison == 2:
            dsn_2_loss = class_balanced_cross_entropy_loss(end_points['osvos/score-dsn_2-cr'], input_label)
            tf.summary.scalar('dsn_2_loss', dsn_2_loss)
            dsn_3_loss = class_balanced_cross_entropy_loss(end_points['osvos/score-dsn_3-cr'], input_label)
            tf.summary.scalar('dsn_3_loss', dsn_3_loss)
            dsn_4_loss = class_balanced_cross_entropy_loss(end_points['osvos/score-dsn_4-cr'], input_label)
            tf.summary.scalar('dsn_4_loss', dsn_4_loss)
            dsn_5_loss = class_balanced_cross_entropy_loss(end_points['osvos/score-dsn_5-cr'], input_label)
            tf.summary.scalar('dsn_5_loss', dsn_5_loss)

        main_loss = class_balanced_cross_entropy_loss(net, input_label)
        tf.summary.scalar('main_loss', main_loss)

        if supervison == 1:
            output_loss = dsn_2_loss + dsn_3_loss + dsn_4_loss + dsn_5_loss + main_loss
        elif supervison == 2:
            output_loss = 0.5 * dsn_2_loss + 0.5 * dsn_3_loss + 0.5 * dsn_4_loss + 0.5 * dsn_5_loss + main_loss
        elif supervison == 3:
            output_loss = main_loss
        else:
            sys.exit('Incorrect supervision id, select 1 for supervision of the side outputs, 2 for weak supervision '
                     'of the side outputs and 3 for no supervision of the side outputs')
        total_loss = output_loss + tf.add_n(tf.losses.get_regularization_losses())
        tf.summary.scalar('total_loss', total_loss)

    # Define optimization method
    with tf.name_scope('optimization'):
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        with tf.name_scope('grad_accumulator'):
            grad_accumulator = {}
            for ind in range(0, len(grads_and_vars)):
                if grads_and_vars[ind][0] is not None:
                    grad_accumulator[ind] = tf.ConditionalAccumulator(grads_and_vars[ind][0].dtype)
        with tf.name_scope('apply_gradient'):
            layer_lr = parameter_lr()
            grad_accumulator_ops = []
            for var_ind, grad_acc in six.iteritems(grad_accumulator):
                var_name = str(grads_and_vars[var_ind][1].name).split(':')[0]
                var_grad = grads_and_vars[var_ind][0]
                grad_accumulator_ops.append(grad_acc.apply_grad(var_grad * layer_lr[var_name],
                                                                local_step=global_step))
        with tf.name_scope('take_gradients'):
            mean_grads_and_vars = []
            for var_ind, grad_acc in six.iteritems(grad_accumulator):
                mean_grads_and_vars.append(
                    (grad_acc.take_grad(iter_mean_grad), grads_and_vars[var_ind][1]))
            apply_gradient_op = optimizer.apply_gradients(mean_grads_and_vars, global_step=global_step)
    # Log training info
    merged_summary_op = tf.summary.merge_all()

    # Log evolution of test image
    if test_image_path is not None:
        probabilities = tf.nn.sigmoid(net)
        img_summary = tf.summary.image("Output probabilities", probabilities, max_outputs=1)
    # Initialize variables
    init = tf.global_variables_initializer()

    # Create objects to record timing and memory of the graph execution
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) # Option in the session options=run_options
    # run_metadata = tf.RunMetadata() # Option in the session run_metadata=run_metadata
    # summary_writer.add_run_metadata(run_metadata, 'step%d' % i)
    with tf.Session(config=config) as sess:
        print('Init variable')
        sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # Create saver to manage checkpoints
        saver = tf.train.Saver(max_to_keep=None)

        last_ckpt_path = tf.train.latest_checkpoint(logs_path)
        if last_ckpt_path is not None and resume_training:
            # Load last checkpoint
            print('Initializing from previous checkpoint...')
            saver.restore(sess, last_ckpt_path)
            step = global_step.eval() + 1
        else:
            # Load pre-trained model
            if finetune == 0:
                print('Initializing from pre-trained imagenet model...')
                init_weights(sess)
            else:
                print('Initializing from specified pre-trained model...')
                # init_weights(sess)
                var_list = []
                for var in tf.global_variables():
                    var_type = var.name.split('/')[-1]
                    if 'weights' in var_type or 'bias' in var_type:
                        var_list.append(var)
                saver_res = tf.train.Saver(var_list=var_list)
                saver_res.restore(sess, initial_ckpt)
            step = 1
        sess.run(interp_surgery(tf.global_variables()))
        print('Weights initialized')

        print('Start training')
        while step < max_training_iters + 1:
            # Average the gradient
            for _ in range(0, iter_mean_grad):
                batch_image, batch_label = dataset.next_batch(batch_size, 'train')
                image = preprocess_img(batch_image[0])
                label = preprocess_labels(batch_label[0])
                run_res = sess.run([total_loss, merged_summary_op] + grad_accumulator_ops,
                                   feed_dict={input_image: image, input_label: label})
                batch_loss = run_res[0]
                summary = run_res[1]

            # Apply the gradients
            sess.run(apply_gradient_op)  # Momentum updates here its statistics

            # Save summary reports
            summary_writer.add_summary(summary, step)

            # Display training status
            if step % display_step == 0:
                print("{} Iter {}: Training Loss = {:.4f}".format(datetime.now(), step, batch_loss), file=sys.stderr)

            # Save a checkpoint
            if step % save_step == 0:
                if test_image_path is not None:
                    curr_output = sess.run(img_summary, feed_dict={input_image: preprocess_img(test_image_path)})
                    summary_writer.add_summary(curr_output, step)
                save_path = saver.save(sess, model_name, global_step=global_step)
                print("Model saved in file: %s" % save_path)

            step += 1

        if (step - 1) % save_step != 0:
            save_path = saver.save(sess, model_name, global_step=global_step)
            print("Model saved in file: %s" % save_path)

        print('Finished training.')


def train_parent(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step,
                 display_step, global_step, iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False,
                 config=None, test_image_path=None, ckpt_name="osvos"):
    """Train OSVOS parent network
    Args:
    See _train()
    Returns:
    """
    finetune = 0
    _train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad, batch_size, momentum, resume_training, config, finetune, test_image_path,
           ckpt_name)


def train_finetune(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step,
                   display_step, global_step, iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False,
                   config=None, test_image_path=None, ckpt_name="osvos"):
    """Finetune OSVOS
    Args:
    See _train()
    Returns:
    """
    finetune = 1
    _train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad, batch_size, momentum, resume_training, config, finetune, test_image_path,
           ckpt_name)


def test(dataset, checkpoint_file, result_path, config=None):
    """Test one sequence
    Args:
    dataset: Reference to a Dataset object instance
    checkpoint_path: Path of the checkpoint to use for the evaluation
    result_path: Path to save the output images
    config: Reference to a Configuration object used in the creation of a Session
    Returns:
    """
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)

    # Input data
    batch_size = 1
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])

    # Create the cnn
    with slim.arg_scope(osvos_arg_scope()):
        net, end_points = osvos(input_image)
    probabilities = tf.nn.sigmoid(net)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(interp_surgery(tf.global_variables()))
        saver.restore(sess, checkpoint_file)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        for frame in range(0, dataset.get_test_size()):
            img, curr_img = dataset.next_batch(batch_size, 'test')
            curr_frame_orig_name = os.path.split(curr_img[0])[1]
            curr_frame = os.path.splitext(curr_frame_orig_name)[0] + '.png'
            image = preprocess_img(img[0])
            res = sess.run(probabilities, feed_dict={input_image: image})
            res_np = res.astype(np.float32)[0, :, :, 0] > 162.0/255.0
            scipy.misc.imsave(os.path.join(result_path, curr_frame), res_np.astype(np.float32))
            print('Saving ' + os.path.join(result_path, curr_frame))



