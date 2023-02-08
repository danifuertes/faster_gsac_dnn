import os
import time
import onnx
import tf2onnx
import onnxruntime
import numpy as np
from tqdm import tqdm
DEVICE = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from utils.net_utils import create_model
from utils.data_utils import data_gen
from utils.options import get_options


def main(opts):

    # Generate testing data
    test_generator = data_gen(**{
        'lines': [],
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
    onnx_path = opts.weights.replace('.h5', '.onnx')
    if not os.path.exists(onnx_path):
        print('\nTransforming Keras model to ONNX model...')
        onnx_model, _ = tf2onnx.convert.from_keras(model, [model.input.type_spec])
        onnx.save(onnx_model, onnx_path)
        print('Keras model converted to ONNX model')
    else:
        print(onnx_path, ' already exists. Loading that ONNX model')

    # Load ONNX model. Possible providers: CPUExecutionProvider, CUDAExecutionProvider, TensorrtExecutionProvider
    device = 'CPU' if int(DEVICE) < 0 else 'GPU'
    provider = 'CUDAExecutionProvider' if device == 'GPU' else 'CPUExecutionProvider'
    session = onnxruntime.InferenceSession(onnx_path, providers=[provider])

    # Input image
    x = np.random.randn(1, *opts.input_dim).astype(np.float32)
    loop_count = 1000

    # Keras prediction
    print()
    _ = model.predict_on_batch(np.random.randn(2, *opts.input_dim).astype(np.float32))[0]
    count = []
    for _ in tqdm(range(loop_count)):
        start_time = time.time()
        _ = model.predict_on_batch(x)
        count.append(time.time() - start_time)
    print("Keras inferences (in " + device + ") with %s +- %s second in average" % (np.mean(count), np.std(count)))
    print("\tFPS: %s + %s" % (np.mean(1 / np.array(count)), np.std(1 / np.array(count))))

    # ONNX prediction
    print()
    x = x if isinstance(x, list) else [x]
    feed = dict([(inputs.name, x[n]) for n, inputs in enumerate(session.get_inputs())])
    _ = session.run(None, feed)[0]
    count2 = []
    for _ in tqdm(range(loop_count)):
        start_time = time.time()
        _ = session.run(None, feed)
        count2.append(time.time() - start_time)
    print("ONNX inferences (in " + device + ") with %s +- %s second in average" % (np.mean(count2), np.std(count2)))
    print("\tFPS: %s +- %s" % (np.mean(1 / np.array(count2)), np.std(1 / np.array(count2))))


if __name__ == "__main__":
    main(get_options())
