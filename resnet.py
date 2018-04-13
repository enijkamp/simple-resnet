import tensorflow as tf

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5
_NUM_IMAGES = {'train': 50000, 'validation': 10000}


###############################################################################

# TODO https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow?answertab=votes#tab-top
# TODO implement training and update means?
def batch_norm_old(x, training):
    epsilon = 1e-3

    gamma = tf.Variable(tf.ones([-1]), validate_shape=False)
    beta = tf.Variable(tf.zeros([-1]), validate_shape=False)

    mu, sigma = tf.nn.moments(x, [0, 1, 2])
    x_hat = (x - mu) / tf.sqrt(sigma + epsilon)
    return gamma * x_hat + beta


def batch_norm(x, training):
    _BATCH_NORM_DECAY = 0.997
    _BATCH_NORM_EPSILON = 1e-5
    return tf.layers.batch_normalization(inputs=x, axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=training)


def fixed_padding(x, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return tf.pad(x, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])


def conv2d_fixed_padding(x, filters, kernel_size, strides):
  if strides > 1:
    x = fixed_padding(x, kernel_size)

  return tf.layers.conv2d(
      inputs=x, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer())


def conv2d_fixed_padding_old(x, filters, kernel_size, strides):
    if strides > 1:
        x = fixed_padding(x, kernel_size)
    padding = ('SAME' if strides == 1 else 'VALID')

    w = tf.Variable(tf.truncated_normal(shape=[-1, kernel_size, kernel_size, filters], stddev=0.1), validate_shape=False)
    return tf.nn.conv2d(x, w, padding=padding, strides=[strides, strides, strides, strides])


def _building_block_v1(x, filters, training, projection_shortcut, strides):
    shortcut = x

    if projection_shortcut is not None:
        shortcut = projection_shortcut(x)
        shortcut = batch_norm(x=shortcut, training=training)

    x = conv2d_fixed_padding(x=x, filters=filters, kernel_size=3, strides=strides)
    x = batch_norm(x, training=training)
    x = tf.nn.relu(x)

    x = conv2d_fixed_padding(x=x, filters=filters, kernel_size=3, strides=1)
    x = batch_norm(x, training=training)
    x += shortcut
    x = tf.nn.relu(x)

    return x


def block_layer(x, filters, block_fn, blocks, strides, training, name):
    def projection_shortcut(x):
        return conv2d_fixed_padding(x=x, filters=filters, kernel_size=1, strides=strides)

    x = block_fn(x, filters, training, projection_shortcut, strides)

    for _ in range(1, blocks):
        x = block_fn(x, filters, training, None, 1)

    return tf.identity(x, name)


###############################################################################


class Model(object):

    def __init__(self, resnet_size, num_classes, num_filters, kernel_size, conv_stride, first_pool_size, first_pool_stride, second_pool_size, second_pool_stride, block_sizes, block_strides, final_size):
        self.resnet_size = resnet_size
        self.block_fn = _building_block_v1
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.second_pool_size = second_pool_size
        self.second_pool_stride = second_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.final_size = final_size

    def __call__(self, x, training):
        x = tf.transpose(x, [0, 3, 1, 2])
        x = conv2d_fixed_padding(x=x, filters=self.num_filters, kernel_size=self.kernel_size, strides=self.conv_stride)
        x = tf.identity(x, 'initial_conv')

        if self.first_pool_size:
            x = tf.nn.max_pool(x, ksize=[1, self.first_pool_size, self.first_pool_size, 1], strides=[1, self.first_pool_stride, self.first_pool_size, 1], padding='SAME')
            x = tf.identity(x, 'initial_max_pool')

        for i, num_blocks in enumerate(self.block_sizes):
            num_filters = self.num_filters * (2**i)
            x = block_layer(x=x, filters=num_filters, block_fn=self.block_fn, blocks=num_blocks, strides=self.block_strides[i], training=training, name='block_layer{}'.format(i + 1))

        x = batch_norm(x=x)
        x = tf.nn.relu(x)

        axes = [2, 3]
        x = tf.reduce_mean(x, axes, keepdims=True)
        x = tf.identity(x, 'final_reduce_mean')

        x = tf.reshape(x, [-1, self.final_size])
        x = tf.layers.dense(inputs=x, units=self.num_classes)
        x = tf.identity(x, 'final_dense')

        return x


class Cifar10Model(Model):
    def __init__(self, resnet_size, num_classes=_NUM_CLASSES):
        assert resnet_size % 6 == 2, 'resnet_size must be 6n + 2'
        num_blocks = (resnet_size - 2) // 6
        super(Cifar10Model, self).__init__(resnet_size=resnet_size, num_classes=num_classes, num_filters=16, kernel_size=3, conv_stride=1, first_pool_size=None, first_pool_stride=None, second_pool_size=8, second_pool_stride=1, block_sizes=[num_blocks] * 3, block_strides=[1, 2, 2], final_size=64)


def learning_rate_with_decay(batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn


def resnet_model_fn(features, labels, mode, model_class, resnet_size, weight_decay, learning_rate_fn, momentum):
    # model
    model = model_class(resnet_size)
    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)
    predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs={'predict': tf.estimator.export.PredictOutput(predictions)})

    # objective
    cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'batch_normalization' not in v.name])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss

    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_fn(global_step)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def cifar10_model_fn(features, labels, mode, params):
    features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])
    learning_rate_fn = learning_rate_with_decay(batch_size=params['batch_size'], batch_denom=128, num_images=_NUM_IMAGES['train'], boundary_epochs=[100, 150, 200], decay_rates=[1, 0.1, 0.01, 0.001])

    weight_decay = 2e-4
    momentum = 0.9

    return resnet_model_fn(features, labels, mode, Cifar10Model, params['resnet_size'], weight_decay, learning_rate_fn, momentum)


###############################################################################


def get_filenames(is_training, data_dir):
    data_dir = data_dir + '/cifar-10-batches-bin'
    assert tf.gfile.Exists(data_dir)
    if is_training:
        return [data_dir + '/data_batch_%d.bin' % i for i in range(1, _NUM_DATA_FILES + 1)]
    else:
        return [data_dir + '/test_batch.bin']


def parse_record(raw_record, is_training):
    record_vector = tf.decode_raw(raw_record, tf.uint8)
    label = tf.cast(record_vector[0], tf.int32)
    label = tf.one_hot(label, _NUM_CLASSES)
    depth_major = tf.reshape(record_vector[1:_RECORD_BYTES], [_NUM_CHANNELS, _HEIGHT, _WIDTH])
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
    image = preprocess_image(image, is_training)
    return image, label


def preprocess_image(image, is_training, seed=1):
    if is_training:
        image = tf.image.resize_image_with_crop_or_pad(image, _HEIGHT + 8, _WIDTH + 8)
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS], seed=seed)
        image = tf.image.random_flip_left_right(image, seed=seed)
    return tf.image.per_image_standardization(image)


def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer, parse_record_fn, num_epochs=1, num_parallel_calls=1, seed=1):
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(lambda value: parse_record_fn(value, is_training), num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def input_function(is_training, data_dir, batch_size, num_epochs=1, num_parallel_calls=1):
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
    num_images = is_training and _NUM_IMAGES['train'] or _NUM_IMAGES['validation']

    return process_record_dataset(dataset, is_training, batch_size, _NUM_IMAGES['train'], parse_record, num_epochs, num_parallel_calls)


###############################################################################


def train_model(flags):
    session_config = tf.ConfigProto(inter_op_parallelism_threads=flags.inter_op_parallelism_threads, intra_op_parallelism_threads=flags.intra_op_parallelism_threads, allow_soft_placement=True)
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9, session_config=session_config)
    classifier = tf.estimator.Estimator(model_fn=cifar10_model_fn, model_dir=flags.model_dir, config=run_config, params={'resnet_size': flags.resnet_size, 'batch_size': flags.batch_size})

    for _ in range(flags.train_epochs // flags.epochs_between_evals):

        tf.logging.info('Starting a training cycle.')

        def input_fn_train():
            return input_function(True, flags.data_dir, flags.batch_size, flags.epochs_between_evals, flags.num_parallel_calls)

        classifier.train(input_fn=input_fn_train, max_steps=flags.max_train_steps)

        tf.logging.info('Starting to evaluate.')

        def input_fn_eval():
            return input_function(False, flags.data_dir, flags.batch_size, 1, flags.num_parallel_calls)

        eval_results = classifier.evaluate(input_fn=input_fn_eval, steps=flags.max_train_steps)

        tf.logging.info(eval_results)


def download_data(flags):
    import os
    import sys
    import tarfile
    from six.moves import urllib

    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    if os.path.exists(flags.data_dir):
        return

    os.makedirs(flags.data_dir)

    filename = url.split('/')[-1]
    filepath = os.path.join(flags.data_dir, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.1f%%' % (filename, 100.0 * count * block_size / total_size))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(flags.data_dir)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.flags.DEFINE_string('data_dir', 'data', '')
    tf.app.flags.DEFINE_string('model_dir', 'models', '')
    tf.app.flags.DEFINE_integer('resnet_size', 32, '')
    tf.app.flags.DEFINE_integer('train_epochs', 250, '')
    tf.app.flags.DEFINE_integer('epochs_between_evals', 10, '')
    tf.app.flags.DEFINE_integer('batch_size', 128, '')
    tf.app.flags.DEFINE_integer('inter_op_parallelism_threads', 0, '')
    tf.app.flags.DEFINE_integer('intra_op_parallelism_threads', 0, '')
    tf.app.flags.DEFINE_integer('num_parallel_calls', 5, '')
    tf.app.flags.DEFINE_boolean('max_train_steps', None, '')

    download_data(tf.app.flags.FLAGS)
    train_model(tf.app.flags.FLAGS)
