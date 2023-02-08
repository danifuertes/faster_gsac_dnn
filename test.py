import os
import time
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

from utils.utils import load_lines, plot_predictions
from utils.net_utils import create_model
from utils.data_utils import data_gen
from utils.options import get_options


def main(opts):

    # Load file containing the path of the training images
    lines_test = load_lines(opts.test_path, opts.dataset_path)

    # Generate testing data
    test_generator = data_gen(**{
        'lines': lines_test,
        'batch_size': opts.batch_size_test,
        'input_dim': opts.input_dim,
        'max_boxes': opts.max_boxes,
        'test': True,
        'grid_dim': opts.grid_dim,
        'filter_img': opts.filter_img,
        'neighborhood': opts.neighborhood
    })

    # Define grid positions
    opts.grid = test_generator.grid
    opts.num_classifiers = opts.grid.shape[0]

    # Create model
    model = create_model(opts, opts.weights, True)

    # Use ONNX
    if opts.use_onnx:  # pip install tf2onnx | pip install onnxruntime
        import onnx
        import tf2onnx
        import onnxruntime
        onnx_path = opts.weights.replace('.h5', '.onnx')
        if not os.path.isfile(onnx_path):
            model, _ = tf2onnx.convert.from_keras(model, [model.input.type_spec])
            onnx.save(model, onnx_path)
            print('.h5 model converted to .onnx')
        else:
            print(onnx_path, ' already exists. Loading that ONNX model')
        provider = 'CPUExecutionProvider' if opts.use_cpu else 'CUDAExecutionProvider'
        model = onnxruntime.InferenceSession(onnx_path, providers=[provider])
        # Possible providers: CPUExecutionProvider, CUDAExecutionProvider, TensorrtExecutionProvider

    # Progress bar
    progbar = tf.keras.utils.Progbar(target=test_generator.samples)
    steps_done = 0

    # Create folders to save outputs (for mAP calculation)
    time_txt = time.strftime("%Y%m%dT%H%M%S")
    pred_dir = '{}_{}'.format(opts.dataset_name.replace('/', '_'), time_txt)
    model_dir = '/'.join(opts.weights.split('/')[:-1])
    det_dir = os.path.join(model_dir, 'predictions', pred_dir, 'detection-results')
    gt_dir = os.path.join(model_dir, 'predictions', pred_dir, 'ground-truth')
    img_dir = os.path.join(model_dir, 'predictions', pred_dir, 'images')
    if not os.path.exists(det_dir):
        os.makedirs(det_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(img_dir) and opts.save_imgs:
        os.makedirs(img_dir)

    # Print and save options
    print('\nOptions:')
    save_opts = open(os.path.join(model_dir, 'predictions', pred_dir, 'options_{}.txt'.format(time_txt)), 'w')
    for k, v in vars(opts).items():
        save_opts.write("'{}': {}\n".format(k, v))
        print("'{}': {}".format(k, v))
    print()
    save_opts.close()

    # Iterate over batches
    for batch in test_generator:

        # Make predictions
        if opts.use_onnx:
            b = batch[0] if isinstance(batch[0], list) else [batch[0]]
            feed = dict([(input.name, b[n].astype(np.float32)) for n, input in enumerate(model.get_inputs())])
            pred = model.run(None, feed)[0]
        else:
            pred = model.predict_on_batch(batch[0])
        pred = np.concatenate((pred[..., 1, None], pred[..., 0, None], pred[..., 2, None]), axis=-1)

        # Iterate over images
        for n in range(batch[0].shape[0]):

            # Get ground-truth
            gt = np.array(batch[1][n])
            gt = gt[np.logical_and(gt[:, 0] > 0, gt[:, 1] > 0)]

            # Get path to image
            path = test_generator.batch_lines[n].split(' ')[0]
            path_list = path.replace('images/', '').split('/')
            sequence = path_list[path_list.index('test') + 1] if 'test' in path_list else path_list[-2]

            # Get predictions
            markers = pred[n, :][pred[n, :, 2] != 0]

            # Save predictions for mAP calculation
            results = open(
                os.path.join(det_dir, path.replace("/", "_").replace('.jpg', ".txt").replace('.png', ".txt")), 'w'
            )
            for marker in markers:
                results.write('vehicle {} {} {}\n'.format(marker[2], marker[0], marker[1]))
            results.close()

            # Save gt for mAP calculation
            results = open(
                os.path.join(gt_dir, path.replace("/", "_").replace('.jpg', ".txt").replace('.png', ".txt")), 'w'
            )
            for g in gt:
                results.write('vehicle {} {}\n'.format(g[0], g[1]))
            results.close()

            # Visualization mode (slower)
            if opts.visualize or opts.save_imgs:
                plot_predictions(
                    markers,
                    gt,
                    opts.input_dim[:2],
                    path,
                    img_dir,
                    sequence,
                    visualize=opts.visualize,
                    save_imgs=opts.save_imgs
                )

            # Update progress bar
            steps_done += 1
            progbar.update(steps_done)
    print('Steps done: ', steps_done)


if __name__ == "__main__":
    main(get_options())
