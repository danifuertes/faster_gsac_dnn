import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.callbacks import Callback


def print_opts(opts, time_txt):
    print('\nOptions:')
    if not os.path.exists(os.path.join(opts.model_dir, 'log_dir')):
        os.makedirs(os.path.join(opts.model_dir, 'log_dir'))
    save_opts = open(os.path.join(opts.model_dir, 'log_dir', 'options_{}.txt'.format(time_txt)), 'w')
    for k, v in vars(opts).items():
        save_opts.write("'{}': {}\n".format(k, v))
        print("'{}': {}".format(k, v))
    print()
    save_opts.close()


def load_img(path, img_mode=3, target_size=None, original_shape=False):

    """
    Load an image and convert it to NumPy array.

    # Arguments
        path: Path to image file.
        img_mode: 3 (rgb) or 1 (grayscale).
        target_size: Either `None` (default to original size) or tuple of ints `(img_height, img_width)`.
        original_shape: whether to output the original shape or not.

    # Returns
        Image as NumPy array.
    """

    # Load image
    image = Image.open(path).convert('RGB')
    shape = image.size

    # Resize
    if target_size is not None:
        image = image.resize(target_size)

    # Channels
    if img_mode == 1:
        image = ImageOps.grayscale(image)

    # Return NumPy image
    if original_shape:
        return np.asarray(image, dtype=np.float64) / 255, shape
    return np.asarray(image, dtype=np.float64) / 255


def load_lines(path_to_lines, dataset_path, shuffle=True):
    """
    Load lines in path_to_lines, add to them the absolute path, and (maybe) shuffle them.
    """
    assert os.path.isdir(dataset_path), "dataset_path {} does not exist".format(dataset_path)
    assert os.path.isfile(path_to_lines), "path_to_lines {} does not exist".format(path_to_lines)
    with open(path_to_lines) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if lines[i][0] == '/':
            lines[i] = lines[i][1:]
        lines[i] = os.path.join(dataset_path, lines[i].replace('\n', ''))
    if shuffle:
        np.random.shuffle(lines)
    return lines


def plot_hist(model_dir):
    path = os.path.join(model_dir, 'log_dir', 'history.pkl')
    if os.path.isfile(path):
        with open(os.path.join(model_dir, 'log_dir', 'history.pkl'), 'rb') as f:
            history = pickle.load(f)

        # Plot history
        elements = [e for e in list(history[0].keys()) if not e.startswith('val_')]
        for element in elements:
            hist_train, hist_val = [], []
            for k, v in history.items():
                hist_train.append(v[element])
                if 'val_' + element in list(history[0].keys()):
                    hist_val.append(v['val_' + element])
            plt.plot(hist_train)
            plt.plot(hist_val)
            plt.title('model ' + element)
            plt.ylabel(element)
            plt.xlabel('epoch')
            if len(hist_val) > 0:
                plt.legend(['train', 'val'], loc='upper left')
            else:
                plt.legend(['train'], loc='upper left')
            plt.savefig(os.path.join(model_dir, 'log_dir', element), dpi=200)
            plt.show()
            plt.clf()
    else:
        print('Cannot find history.pkl file containing the training history in {}'.format(path))


def plot_predictions(markers, gt, input_dim, path, img_dir, sequence, visualize=True, save_imgs=False):

    # Load the image not resized
    img = load_img(path)

    # Variables to extrapolate coordinates to the real size of the image
    dpi = 200
    h, w = img.shape[:2]
    lx = w / input_dim[0]
    ly = h / input_dim[1]

    # Show image
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect='auto')

    # Plot gt
    gt[:, 0] = gt[:, 0] * lx
    gt[:, 1] = gt[:, 1] * ly
    for g in gt:
        ax.scatter(g[0], g[1], c='red', marker='o', alpha=0.5)

    # Extrapolate the coordinates to the real size of the image and plot the markers
    if len(markers) > 0:
        markers[:, 0] = markers[:, 0] * lx
        markers[:, 1] = markers[:, 1] * ly
        for marker in markers:
            ax.annotate(str(round(marker[2], 2)), xy=(marker[0], marker[1]), xytext=(marker[0] + 20, marker[1] + 20),
                        xycoords='data', bbox=dict(boxstyle='round,pad=0.2', fc='green', alpha=0.25))
            ax.scatter(marker[0], marker[1], c='green', marker='o', alpha=0.5)

    # Save images
    if save_imgs:
        p = os.path.join(img_dir, sequence)
        if not os.path.exists(p):
            os.makedirs(p)
        s = path.split('/')[-1]
        fig.savefig(os.path.join(p, s), dpi=dpi)

    # Show image and clean figure
    if visualize:
        fig.show()
    plt.close(fig)
    plt.clf()


class HistoryCallback(Callback):

    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        if os.path.isfile(os.path.join(self.path, 'history.pkl')):
            with open(os.path.join(self.path, 'history.pkl'), 'rb') as f:
                history = pickle.load(f)
            history[epoch] = logs
        else:
            history = {epoch: logs}
        with open(os.path.join(self.path, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
