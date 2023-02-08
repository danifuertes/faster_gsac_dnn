import numpy as np
from PIL import ImageFile
from tensorflow import keras

from nets.gsac_dnn import gt_reform, gt_reorder
from utils.data_aug import data_augmentation
from utils.utils import load_img
ImageFile.LOAD_TRUNCATED_IMAGES = True


def data_gen(**kwargs):
    return DataGeneratorGSACDNN(**kwargs)


class DataGeneratorGSACDNN(keras.utils.Sequence):
    """Generates data for GSAC-DNN"""
    def __init__(self, lines, grid_dim, batch_size=32, input_dim=(224, 224, 3), filter_img=None, data_aug=False,
                 test=False, shuffle=False, max_boxes=20, neighborhood=1, **kwargs):
        """Initialization"""

        # Files
        self.lines = lines
        self.samples = len(lines)

        # Data augmentation
        self.data_aug = data_aug

        # Shapes
        self.batch_size = batch_size
        self.input_dim = input_dim

        # GSAC grid
        self.neighborhood = neighborhood
        self.grid_dim = grid_dim
        self.dx = round(input_dim[0] / grid_dim[0])
        self.dy = round(input_dim[1] / grid_dim[1])
        grid_x = np.arange(round(self.dx / 2), input_dim[0] - 1, self.dx).astype(int)
        grid_y = np.arange(round(self.dy / 2), input_dim[1] - 1, self.dy).astype(int)
        if filter_img is not None:
            mask = np.squeeze(load_img(filter_img, target_size=input_dim[:2], img_mode=1))
            aux = np.zeros((input_dim[1], input_dim[0]))
            for gx in grid_x:
                for gy in grid_y:
                    aux[gy, gx] = 1
            aux = np.logical_and(aux, mask)
            self.grid = np.argwhere(aux == 1)
        else:
            grid = []
            for gx in grid_x:
                for gy in grid_y:
                    grid.append([gx, gy])
            self.grid = np.array(grid)

        # Other info
        self.test = test
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.lines))
        self.max_boxes = max_boxes
        self.paths = []
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.lines) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        if self.test and len(self.lines) < (index + 1) * self.batch_size:
            indexes = self.indexes[index * self.batch_size:]
        else:
            indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        self.batch_lines = [self.lines[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(self.batch_lines)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.lines))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, lines):
        """Generates data containing batch_size samples"""

        # Initialization
        batch_x = np.empty((len(lines), *self.input_dim))  # X : (n_samples, *dim, n_channels)
        if self.test:
            batch_y = np.zeros((len(lines), self.max_boxes, 2), dtype=float)
        else:
            batch_y = np.zeros((len(lines), self.grid.shape[0]), dtype=int)

        # Generate data
        for i, line in enumerate(lines):
            line_split = line.split(' ')

            # Load image
            x, true_shape = load_img(line_split[0], img_mode=self.input_dim[2], target_size=self.input_dim[:2],
                                     original_shape=True)

            # Scale factor if the images are resized
            scale_x = true_shape[0] / self.input_dim[0]
            scale_y = true_shape[1] / self.input_dim[1]

            # Get the gt of the chosen image
            boxes = []
            for box in range(1, len(line_split)):
                left = float(line_split[box].split(',')[0]) / scale_x
                top = float(line_split[box].split(',')[1]) / scale_y
                boxes.append([left, top])
            boxes = np.array(boxes)

            # Data augmentation
            if self.data_aug:
                x, boxes = data_augmentation(x, boxes)

            # Save image
            batch_x[i] = x

            # Save the gts
            for j, box in enumerate(boxes):
                left, top = box[0], box[1]
                if self.test:
                    assert j <= self.max_boxes, \
                        'Image {} has more bounding boxes than the maximum ({}). You should consider a higher value ' \
                        'for the option max_boxes (in options.py)'.format(line_split[0], self.max_boxes)
                    batch_y[i, j, :] = [left, top]
                else:
                    batch_y[i] = np.logical_or(batch_y[i], gt_reform(self.dx, self.dy, self.grid, left, top))
                    if self.neighborhood > 1:
                        batch_y[i] = gt_reorder(batch_y[i], self.grid_dim, neighborhood=self.neighborhood)
        return batch_x, batch_y
