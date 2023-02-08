import os
import tensorflow as tf

from nets.backbones import *
from nets.gsac_dnn import create_gsac_dnn


def load_weights(model, path):
    """
    Load weights in path on model.
    """
    if path is not None and os.path.isfile(path):
        try:
            model.load_weights(path, by_name=True, skip_mismatch=True)
            print("Loaded model from {}".format(path))
        except:
            print("Impossible to find weight path. Returning untrained model")
    else:
        print("Impossible to find weight path. Returning untrained model")
    return model


def create_model(opts, weights_path, test_mode=False):
    """
    Create model.

    # Arguments
       opts: options.
       num_classes: number of classes to predict.
       weights_path: Path to pre-trained model.
       test_mode: inference phase.

    # Returns
       model: A Model instance.
    """

    # Define input tensor
    inputs = tf.keras.layers.Input(shape=opts.input_dim)

    # Create backbone
    if opts.backbone == 'resnetv1_own':
        depth = opts.rn_layers * 6 + 2  # Computed depth from supplied model parameter n
        feature_maps = resnet_v1(inputs, depth)
    elif opts.backbone == 'resnetv2_own':
        depth = opts.rn_layers * 9 + 2  # Computed depth from supplied model parameter n
        feature_maps = resnet_v2(inputs, depth)
    elif opts.backbone == 'vit_own':
        from nets.vit_backbone import vit
        inputs, feature_maps = vit(
            image_size=opts.input_dim[:2],
            patch_size=16,
            dropout=0.1,
            mlp_dim=3072,
            num_heads=12,
            num_layers=12,
            hidden_size=768
        )
    else:
        feature_maps = keras_backbone(opts.backbone, inputs, opts.input_dim, opts.keras_weights)

    # Create head
    predictions = create_gsac_dnn(
        feature_maps,
        opts.num_classifiers,
        opts.neighborhood,
        grid_dim=opts.grid_dim,
        grid=opts.grid,
        faster=opts.faster_gsac_dnn,
        test_mode=test_mode
    )
    if test_mode:
        predictions = NMS()(predictions, opts.dist_th, opts.score_th, opts.max_boxes)

    # Define model
    model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
    print(model.summary())

    # Load the weights of the model
    model = load_weights(model, weights_path)
    return model


class NMS(tf.keras.layers.Layer):

    def __int__(self):
        super(NMS, self).__init__()

    def call(self, predictions, dist_th=16., score_th=0.3, max_boxes=20):

        batch_size = tf.shape(predictions)[0]
        num_values = tf.shape(predictions)[2]

        # Remove markers whose score is < score_th
        mask = tf.cast(
            tf.tile(
                tf.expand_dims(
                    predictions[..., 2] >= score_th,
                    axis=-1
                ),
                [1, 1, num_values]
            ),
            dtype=predictions.dtype
        )
        predictions = predictions * mask
        predictions = tf.gather_nd(
            predictions,
            tf.argsort(predictions[..., 2], direction='DESCENDING', axis=-1)[..., None],
            batch_dims=1
        )

        def condition(m, p, i):
            return i < max_boxes

        def body(m, p, i):

            # Get best score
            best_scores = tf.math.argmax(p[..., 2], axis=-1)
            idx = tf.stack([tf.range(batch_size, dtype=best_scores.dtype), best_scores], axis=-1)

            current_row = tf.tile((tf.range(max_boxes) == i)[None, :, None], [batch_size, 1, num_values])
            m = tf.where(current_row, tf.gather_nd(p, idx)[:, None], m)

            # Remove scores close to the best score (the best score is also deleted)
            cond = (tf.norm(p[..., :2] - m[:, i, None, :2], axis=-1) < dist_th)[..., None]
            p = tf.where(cond, tf.zeros_like(p), p)
            return m, p, i + 1

        # Apply NMS
        markers, _, _ = tf.while_loop(condition, body, [tf.zeros_like(predictions)[:, :max_boxes], predictions, 0])
        return markers
