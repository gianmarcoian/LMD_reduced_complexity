import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds

def save_filtered_mnist_digits(digits, save_path):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    dataset_builder = tfds.builder('mnist')
    dataset_builder.download_and_prepare()
    ds = dataset_builder.as_dataset(split='train')

    for example in ds:
        label = example['label'].numpy()
        if label in digits:
            image = example['image'].numpy()
            #building a subfolder for each class
            digit_path = os.path.join(save_path, str(label)) 
            os.makedirs(digit_path, exist_ok=True)
            image_path = os.path.join(digit_path, f"{len(os.listdir(digit_path))}.png")
            tf.keras.preprocessing.image.save_img(image_path, image)

# for example now I train in_domain from 0 to 3 digits classes
# i'm gonna save, for this reason, this dataset in /data/mnist_in
save_filtered_mnist_digits([0, 1, 2, 3], "./data/mnist_in")

# same for ood
save_filtered_mnist_digits([4, 5, 9], "./data/mnist_out")
