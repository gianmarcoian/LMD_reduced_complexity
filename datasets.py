"""Return training and evaluation/test datasets from config files."""
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import os

def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x

def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x

def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC)
    return tf.cast(image, tf.uint8)

def resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio, tf.int32)
    w = tf.round(w * ratio, tf.int32)
    return tf.image.resize(image, [h, w], antialias=True)

def central_crop(image, size):
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)

def filter_digits(example, allowed_digits):
    return tf.reduce_any(tf.equal(example['label'], allowed_digits))

def create_dataset(dataset_builder, split, allowed_digits=None):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
        dataset_builder.download_and_prepare()
        ds = dataset_builder.as_dataset(
            split=split, shuffle_files=True, read_config=read_config)
        
        if allowed_digits is not None:
            allowed_digits_tensor = tf.constant(allowed_digits, dtype=tf.int64)
            ds = ds.filter(lambda x: filter_digits(x, allowed_digits_tensor))
        
        return ds

def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
        config: A ml_collection.ConfigDict parsed from config files.
        uniform_dequantization: If `True`, add uniform dequantization to images.
        evaluation: If `True`, fix number of epochs to 1.

    Returns:
        train_ds, eval_ds, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                         f'the number of devices ({jax.device_count()})')

    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1

    if config.data.dataset == 'MNIST_IN':
        dataset_builder = tfds.builder('mnist')
        train_split_name = 'train'
        eval_split_name = 'test'
        allowed_digits = [0, 1, 2, 3]  # Specifica le cifre che vuoi includere

        train_ds = create_dataset(dataset_builder, train_split_name, allowed_digits)
        eval_ds = create_dataset(dataset_builder, eval_split_name, allowed_digits)
        num_train_examples = sum(1 for _ in train_ds)
        num_eval_examples = sum(1 for _ in eval_ds)
        print(f"Number of training examples: {num_train_examples}")
        print(f"Number of evaluation examples: {num_eval_examples}")
        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.grayscale_to_rgb(img)
            return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    else:
        raise NotImplementedError(
            f'Dataset {config.data.dataset} not yet supported.')

    def preprocess_fn(d):
        """Basic preprocessing function scales data to [0, 1) and randomly flips."""
        img = resize_op(d['image'])
        if config.data.random_flip and not evaluation:
            img = tf.image.random_flip_left_right(img)
        if uniform_dequantization:
            img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

        return dict(image=img, label=d.get('label', None))

    train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(shuffle_buffer_size).batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(prefetch_size)

    eval_ds = eval_ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    eval_ds = eval_ds.batch(batch_size, drop_remainder=True)
    eval_ds = eval_ds.prefetch(prefetch_size)

    return train_ds, eval_ds, dataset_builder
