import os
import random
import argparse
import numpy as np
import tensorflow as tf


def str2bool(v):
    """
    Transform string inputs into boolean.
    :param v: string input.
    :return: string input transformed to boolean.
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed):
    """Set seed"""
    if seed is None:
        seed = random.randrange(100)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    tf.random.set_seed(seed)  # Tensorflow module


def get_options(args=None):
    parser = argparse.ArgumentParser(description="Faster GSAC-DNN")

    # SEED
    parser.add_argument('--seed', type=int, default=1234, help='Seed for reproducibility. None to use random seed')

    # INPUT
    parser.add_argument('--input_dim', type=int, nargs='+', default=[224, 224, 3], help="Input image shape")
    parser.add_argument('--grid_dim', type=int, nargs='+', default=[20, 20], help="Size of the grid of classifiers")

    # MODEL
    # Backbone
    parser.add_argument('--backbone', type=str, default='resnetv1_own', help="Backbone architecture. You can choose"
                        "some from Keras Applications or custom backbones: resnetv1_own, resnetv2_own")
    parser.add_argument('--rn_layers', type=int, default=5, help="Controls the depth of the own resnet network")

    # Head
    parser.add_argument('--dist_th', type=float, default=16, help="Distances threshold")
    parser.add_argument('--score_th', type=float, default=0.3, help="Score threshold")
    parser.add_argument('--max_boxes', type=int, default=20, help="Maximum number of predictions")

    # Gsac-dnn parameters
    parser.add_argument('--faster_gsac_dnn', type=str2bool, default=True, help="True for faster version of GSAC-DNN")
    parser.add_argument('--neighborhood', type=int, default=1, help="Number of classifiers per branch for GSAC-DNN."
                                                                    "Use 0 to include all classifiers in a branch")
    parser.add_argument('--filter_img', type=str, default='', help="Mask that removes unnecessary GASC-DNN classifiers")

    # WEIGHTS
    parser.add_argument('--restore_model', type=str2bool, default=False, help="True to restore a model for training")
    parser.add_argument('--weights', type=str, default='path/to/weights_000.h5', help="Path to file with trained model")
    parser.add_argument('--keras_weights', type=str, default=None, help="Path to pretrained weights of Keras backbones."
                        "Use 'imagenet' to download pretrained ImageNet weights.")

    # TRAIN
    parser.add_argument('--init_lr', type=float, default=1e-3, help="Initial learning rate for Adam")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size during training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--init_epoch', type=int, default=0, help="Initial epoch to start training")
    parser.add_argument('--data_aug', type=str2bool, default=False, help="Apply data augmentation")

    # TEST
    parser.add_argument('--batch_size_test', type=int, default=64, help="Test batch size")
    parser.add_argument('--visualize', type=str2bool, default=False, help="True to see the images while testing")
    parser.add_argument('--save_imgs', type=str2bool, default=False, help="True to save images with predictions")

    # DATASET
    parser.add_argument('--dataset_name', type=str, default='', help="Name of the dataset")
    parser.add_argument('--dataset_path', type=str, default='', help="Path to the dataset")
    parser.add_argument('--classes_path', type=str, default='classes.txt', help="File with a list of classes")
    parser.add_argument('--train_path', type=str, default='train.txt', help="File with labeled train images")
    parser.add_argument('--test_path', type=str, default='test.txt', help="File with labeled test images")
    parser.add_argument('--val_path', type=str, default='val.txt', help="File with labeled val images")
    parser.add_argument('--val_perc', type=float, default=0.2, help="% of train images used to val (if not val_path)")

    # CPU / GPU
    parser.add_argument('--multi_gpu', type=str2bool, default=True, help="Use multiple GPU (if possible)")
    parser.add_argument('--use_onnx', type=str2bool, default=False, help="Convert keras model to onnx during test")

    opts = parser.parse_args(args)
    opts.classes_path = os.path.join(opts.dataset_path, opts.dataset_name, opts.classes_path)
    opts.train_path = os.path.join(opts.dataset_path, opts.dataset_name, opts.train_path)
    opts.test_path = os.path.join(opts.dataset_path, opts.dataset_name, opts.test_path)
    opts.val_path = os.path.join(opts.dataset_path, opts.dataset_name, opts.val_path)
    opts.filter_img = os.path.join(opts.dataset_path, opts.dataset_name, opts.filter_img)
    if not os.path.isfile(opts.filter_img):
        opts.filter_img = None
    else:
        assert opts.neighborhood <= 1, "filter_classifiers is not allowed when neighborhood > 1"

    # Check everything is correct
    assert len(opts.input_dim) == 3, "You must indicate width, height and number of channels"
    assert len(opts.grid_dim) == 2, "You must indicate the size of the grid in the x-axis and y-axis"
    assert opts.grid_dim[0] <= opts.input_dim[0], "Grid size cannot be larger than image size"
    assert opts.grid_dim[1] <= opts.input_dim[1], "Grid size cannot be larger than image size"
    assert opts.grid_dim[1] > 0 or opts.grid_dim[0] > 0, "Grid sizes must be positive numbers"
    assert opts.input_dim[2] in [1, 3], "Number of image channels can only be 1 (grayscale) or 3 (rgb)"
    assert opts.batch_size > 0 or opts.batch_size_test > 0, "Batch size must be a positive number"
    assert 0 < opts.dist_th <= np.min(opts.input_dim[:2]),  "d_th must be in [0, max(image width, image height)]"
    assert 0 < opts.score_th <= 1, "score_th must be in the range [0, 1]"
    assert os.path.isfile(opts.val_path) or 0 <= opts.val_perc <= 1, \
        "Either val_imgs must exist or val_perc must be in the range [0, 1]"
    assert opts.epochs > 0, "epochs must be a positive integer"
    assert opts.init_epoch >= 0, "initial_epoch must be a non-negative integer"
    assert (opts.restore_model and os.path.isfile(opts.weights)) or not opts.restore_model, "weights not found"
    assert opts.neighborhood >= 0, "neighborhood has to be a non-negative integer"
    assert np.sqrt(opts.neighborhood).astype(int)**2 == opts.neighborhood, "The sqrt of neighborhood must be an integer"

    # Set seed
    set_seed(opts.seed)
    return opts
