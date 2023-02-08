import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate, Conv3D, Activation

from nets.backbones import resnet_layer


def create_gsac_dnn(feature_maps, num_classifiers, neighborhood, grid_dim=(20, 20), grid=None, faster=False,
                    test_mode=False):

    # Grid of Spatial Aware Classifiers (GSAC)
    outputs = classifiers(feature_maps, num_classifiers, neighborhood, grid_dim=grid_dim, faster=faster)
    if test_mode:
        outputs = compute_markers(outputs, neighborhood, grid_dim, grid)

    return outputs


def classifiers(x, num_classifiers, neighborhood=1, grid_dim=(20, 20), faster=False):
    """
    Implement the classifiers (the output) of the network.

    # Arguments
        x (tensor): feature maps from the backbone.
        num_classifiers (int): number of classifiers that compose the grid.
        neighborhood (int): number of classifiers per branch.
        grid_dim (list): grid shape.
        faster (boolean): use Faster GSAC-DNN.

    # Returns
        outputs (list): list containing the classifiers.
    """
    outputs = []

    # Faster GSAC-DNN
    if faster:
        x = resnet_layer(x, num_filters=16 * num_classifiers)
        x = Conv3D(
            filters=1,
            kernel_size=[*x.shape[1:3], 16],
            strides=[*x.shape[1:3], 16],
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(x[..., None])[:, 0, 0, :, 0]
        x = Activation('sigmoid')(x)
        return x

    # All classifiers come from the same unique branch
    if neighborhood == 0:
        x_net = resnet_layer(x)
        x_net = Flatten()(x_net)
        outputs = Dense(num_classifiers, activation='sigmoid', name='Classifier')(x_net)

    # One classifier per branch
    elif neighborhood == 1:
        print('Loading classifiers...')
        for i in tqdm(range(num_classifiers)):
            x_net = resnet_layer(x)
            x_net = Flatten()(x_net)
            x_net = Dense(1, activation='sigmoid', name='Classifier_' + str(i))(x_net)
            outputs.append(x_net)
        outputs = Concatenate(name='out')(outputs)

    # Each set of 'neighborhood' classifiers come from a branch
    else:
        print('Loading classifiers...')
        n, step = 0, int(math.sqrt(neighborhood))
        x_grid, y_grid = grid_dim
        for i in tqdm(range(0, x_grid, step)):
            for j in range(0, y_grid, step):
                v_step, h_step = min(j + step, y_grid), min(i + step, x_grid)
                num_neighbors = (v_step - j) * (h_step - i)
                x_net = resnet_layer(x)
                x_net = Flatten()(x_net)
                x_net = Dense(num_neighbors, activation='sigmoid', name='Classifier_' + str(j + i * y_grid))(x_net)
                outputs.append(x_net)
                n += num_neighbors
        assert n == num_classifiers, "num_classifiers does not match the number of outputs of the network"
        outputs = Concatenate(name='out')(outputs)
    return outputs


def compute_markers(activations, neighborhood, grid_dim, grid):
    x_grid, y_grid = grid_dim

    # Coordinates of classifiers (B x V x H x 2)
    grid = tf.tile(tf.expand_dims(tf.constant(grid, dtype=tf.float32), axis=0), [tf.shape(activations)[0], 1, 1])
    grid = tf.reshape(grid, [tf.shape(activations)[0], y_grid, x_grid, 2])

    # Scores of classifiers (B x V x H x 1)
    activations = tf.reshape(activations, [tf.shape(activations)[0], y_grid, x_grid, 1])
    if neighborhood > 1:
        activations = tf.transpose(activations)

    # Coordinates + scores (B x V x H x 3)
    classifiers = tf.concat((grid, activations), axis=-1)

    # Get classifiers that are neighbors
    kernel = tf.constant(
        [[[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
        classifiers.dtype
    )
    top_left = tf.nn.conv2d(classifiers, kernel, [1, 1, 1, 1], padding='VALID')
    kernel = tf.constant(
        [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
        classifiers.dtype
    )
    top_right = tf.nn.conv2d(classifiers, kernel, [1, 1, 1, 1], padding='VALID')
    kernel = tf.constant(
        [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
         [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]],
        classifiers.dtype
    )
    bot_left = tf.nn.conv2d(classifiers, kernel, [1, 1, 1, 1], padding='VALID')
    kernel = tf.constant(
        [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
         [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]],
        classifiers.dtype
    )
    bot_right = tf.nn.conv2d(classifiers, kernel, [1, 1, 1, 1], padding='VALID')

    # Coordinates + scores of neighbors (B x (V-1)(H-1) x 4 x 3) -> 4 is the number of neighbors | 3 is for X, Y, Value
    neighbors = tf.concat(
        tuple(
            [tf.reshape(n, [tf.shape(classifiers)[0], -1, 1, 3]) for n in [top_left, top_right, bot_left, bot_right]]
        ), axis=2
    )

    # Bi-linear interpolation
    x = neighbors[..., 0]
    y = neighbors[..., 1]
    v = tf.transpose(neighbors[..., 2], perm=[0, 2, 1])
    v_sum = tf.math.reduce_sum(neighbors[..., 2], axis=-1)
    cx = tf.linalg.diag_part(tf.matmul(x, v)) / v_sum
    cy = tf.linalg.diag_part(tf.matmul(y, v)) / v_sum

    # Final predictions
    coords = tf.concat(tuple([tf.expand_dims(c, axis=-1) for c in [cx, cy]]), axis=-1)
    scores = v_sum / 4
    return tf.concat((coords, scores[..., None]), axis=-1)


def pred_reorder(pred, v, h, neighborhood=1):
    """Reshape the predictions into a 2D array according to the neighborhood size."""
    out = np.zeros((v, h))
    n, step = 0, np.sqrt(neighborhood).astype(int)
    for i in range(0, h, step):
        for j in range(0, v, step):
            v_step, h_step = min(j + step, v), min(i + step, h)
            num_neighbors = (v_step - j) * (h_step - i)
            out[j:v_step, i:h_step] = pred[n:n + num_neighbors].reshape(((v_step - j), (h_step - i)))
            n += num_neighbors
    return out.reshape(pred.shape, order='F')


def gt_reorder(gt, grid_dim, neighborhood=1):
    """Flatten the ground-truth grid according to the neighborhood size."""
    out = np.zeros(gt.shape)
    x, y = grid_dim
    gt = gt.reshape((y, x), order='F')
    n, step = 0, np.sqrt(neighborhood).astype(int)
    for i in range(0, x, step):
        for j in range(0, y, step):
            v_step, h_step = min(j + step, y), min(i + step, x)
            num_neighbors = (v_step - j) * (h_step - i)
            out[n:n + num_neighbors] = gt[j:v_step, i:h_step].reshape(-1)
            n += num_neighbors
    return out


def gt_reform(dx, dy, grid, x, y):

    """
    Transform the format x|y|h|w of the gt to a binary vector where '1' means that a car is present on the corresponding
    grid point and '0' means that there is no car.

    # Arguments
        dx: grid step in x-axis
        dy: grid step in y-axis
        grid: grid positions
        x: position in x axis of the gt
        y: position in y axis of the gt

    # Returns
        Binary vector ordered by the columns of the grid that indicates with 1.
    """

    # Initialize output gt
    gt = np.zeros(grid.shape[0])

    # Avoid checking all the grid if there are not markers
    if (x <= 0) | (y <= 0):
        return gt

    # Create square around the marker
    x1 = x - dx
    x2 = x + dx
    y1 = y - dy
    y2 = y + dy

    # Iterate over grids
    for i in range(grid.shape[0]):
        if (x1 < grid[i, 1] <= x2) and (y1 < grid[i, 0] <= y2):
            gt[i] = 1
    return gt
