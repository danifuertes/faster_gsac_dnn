import os
import time
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

from utils.options import get_options
from utils.data_utils import data_gen
from utils.net_utils import create_model
from utils.utils import HistoryCallback, print_opts, plot_hist, load_lines


def gen_callable(_gen):
    def gen():
        for x, y in _gen:
            yield x, y
    return gen


def train_model(opts, train_gen, val_gen, model, initial_epoch):
    """
    Model training.

    # Arguments
       train_gen: Training data generated batch by batch.
       val_gen: Validation data generated batch by batch.
       model: A Model instance.
       initial_epoch: Epoch from which training starts.
    """

    # Compile the model
    losses = tf.keras.losses.BinaryCrossentropy()
    metrics = ['binary_accuracy']
    output_shapes = (tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None]))
    model.compile(
        loss=losses,
        metrics=metrics,
        optimizer=tf.keras.optimizers.Adam(learning_rate=opts.init_lr, clipnorm=1., clipvalue=1.)
    )

    # Save model with the lowest validation loss
    weights_path = os.path.join(opts.model_dir, 'weights_{epoch:03d}.h5')
    write_best_model = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                                                          save_best_only=True, save_weights_only=True)

    # Steps
    steps_per_epoch = int(np.ceil(train_gen.samples / opts.batch_size))
    validation_steps = int(np.ceil(val_gen.samples / opts.batch_size))

    # Other callbacks
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-10)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(opts.model_dir, 'log_dir'), histogram_freq=0)
    hist_callback = HistoryCallback(os.path.join(opts.model_dir, 'log_dir'))
    callbacks = [write_best_model, lr_reducer, tensorboard, hist_callback]

    # TF Dataset for multi GPU training (first epoch sometimes is very slow)
    if opts.multi_gpu:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_gen = tf.data.Dataset.from_generator(
            gen_callable(train_gen),
            output_types=(tf.float64, tf.float64),
            output_shapes=output_shapes
        ).with_options(options).repeat(opts.epochs).prefetch(tf.data.AUTOTUNE)
        val_gen = tf.data.Dataset.from_generator(
            gen_callable(val_gen),
            output_types=(tf.float64, tf.float64),
            output_shapes=output_shapes
        ).with_options(options).repeat(opts.epochs).prefetch(tf.data.AUTOTUNE)

    # Train the model
    try:
        # model.run_eagerly = True
        model.fit(
            train_gen,
            epochs=initial_epoch + opts.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=validation_steps,
            initial_epoch=initial_epoch,
            workers=1 if opts.multi_gpu else 16
        )
    except KeyboardInterrupt:
        model.save(os.path.join(opts.model_dir, 'weights_XXX.h5'))

    # Plot and save history
    plot_hist(opts.model_dir)
    print('Finished')


def main(opts):

    # Create model_dir if not exists and restore_model == False
    time_txt = time.strftime("%Y%m%dT%H%M%S")
    aux = 6 if opts.backbone == 'resnetv1_own' else 9
    aux = 'resnet{}v1own'.format(aux * opts.rn_layers + 2)
    head = 'faster_gsac_dnn' if opts.faster_gsac_dnn else 'gsac_dnn'
    opts.model_dir = os.path.join(
        'models',
        opts.dataset_name.replace('/', '_'),
        '{}_{}_{}'.format(
            opts.backbone if opts.backbone not in ['resnetv1_own', 'resnetv2_own'] else aux,
            head + str(opts.neighborhood) if opts.head == 'gsac_dnn' else opts.head,
            time_txt
        )
    )
    if not opts.restore_model:
        weights_path = None
        init_epoch = 0
    else:
        weights_path = opts.weights if opts.weights.startswith('models') else os.path.join('models', opts.weights)
        init_epoch = opts.init_epoch
        opts.model_dir += '_restored'
    if not os.path.exists(opts.model_dir):
        os.makedirs(opts.model_dir)

    # Print and save options
    print_opts(opts, time_txt)

    # Load file containing the path of the training images
    lines_train = load_lines(opts.train_path, opts.dataset_path)
    num_train = len(lines_train)
    if os.path.isfile(opts.val_path):
        lines_val = load_lines(opts.val_path, opts.dataset_path)
        num_val = len(lines_val)
    else:
        num_val = int(num_train * opts.val_perc)
        num_train -= num_val
        lines_val = lines_train[num_train:]
        lines_train = lines_train[:num_train]
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, opts.batch_size))

    # Generate training data with real-time augmentation
    train_generator = data_gen(**{
        'lines': lines_train,
        'batch_size': opts.batch_size,
        'input_dim': opts.input_dim,
        'data_aug': opts.data_aug,
        'shuffle': True,
        'max_boxes': opts.max_boxes,
        'grid_dim': opts.grid_dim,
        'filter_img': opts.filter_img,
        'neighborhood': opts.neighborhood
    })

    # Generate validation data with real-time augmentation
    val_generator = data_gen(**{
        'lines': lines_val,
        'batch_size': opts.batch_size,
        'input_dim': opts.input_dim,
        'max_boxes': opts.max_boxes,
        'grid_dim': opts.grid_dim,
        'filter_img': opts.filter_img,
        'neighborhood': opts.neighborhood
    })

    # Number of classifiers based on grid size
    opts.grid = train_generator.grid
    opts.num_classifiers = train_generator.grid.shape[0]

    # Define model
    if opts.multi_gpu:
        with tf.distribute.MirroredStrategy().scope():
            model = create_model(opts, weights_path)
    else:
        model = create_model(opts, weights_path)

    # Train model
    train_model(opts, train_generator, val_generator, model, init_epoch)


if __name__ == "__main__":
    main(get_options())
