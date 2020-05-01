import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, rgb2hsv
from sklearn.feature_extraction.image import extract_patches_2d
import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from tqdm import tqdm

def S_of_HSV(image):
    image_in_hsv = rgb2hsv(image)
    S_channel = image_in_hsv[:, :, 1]
    return S_channel


def A_of_LAB(image):
    image_in_lab = rgb2lab(image)
    A_channel = image_in_lab[:, :, 1]
    return A_channel


def B_of_RGB(image):
    B_channel = image[:, :, -1]
    return B_channel


def value_scale(array):

    scaled = (array - np.min(array)) / (np.max(array) - np.min(array))
    return scaled


def transform_to_SAB(image):

    S = S_of_HSV(image)
    A = A_of_LAB(image)
    B = B_of_RGB(image)

    S = value_scale(S)
    A = value_scale(A)
    B = value_scale(B)

    transformed = np.dstack((S, A, B))

    return transformed


def read_images(path, image_size, testset_ratio, use_patches, seed):

    dir_list = os.listdir(path)
    testset_size = int(testset_ratio*len(dir_list)) # each directory corresponds to a single sample

    p_size = int(image_size / 2)
    assert p_size * 2 == image_size
    patch_per_image = 2

    train_images = list()
    train_label_masks = list()

    test_images = list()
    test_label_masks = list()

    prng = np.random.RandomState(seed)  # pseudo-random number generator
    prng.shuffle(dir_list)

    it = tqdm(enumerate(dir_list), desc='Preprocessing', total=len(dir_list))
    for ind, directory in it:
        raw_image = imread(os.path.join(path, directory, f'{directory}_Dermoscopic_Image', f'{directory}.bmp'))
        label_mask = imread(os.path.join(path, directory, f'{directory}_lesion', f'{directory}_lesion.bmp'))

        raw_image = resize(raw_image, [image_size, image_size, 3])
        label_mask = resize(label_mask, [image_size, image_size, 1])

        image = transform_to_SAB(raw_image)
        label_mask = label_mask.astype(np.bool).astype(np.uint8)
        if use_patches:
            if ind >= testset_size:
                randint = int(time.time())
                image_patches = extract_patches_2d(image, (p_size, p_size), max_patches=patch_per_image, random_state=randint)
                label_mask_patches = extract_patches_2d(label_mask, (p_size, p_size), max_patches=patch_per_image, random_state=randint)
                label_mask_patches = np.expand_dims(label_mask_patches, -1)

                train_images.append(image_patches)
                train_label_masks.append(label_mask_patches)
            else:
                img2 = image.reshape([1] + list(image.shape))
                label_mask2 = label_mask.reshape([1] + list(label_mask.shape))
                for i in range(2):
                    for j in range(2):

                        test_images.append(img2[:, i * p_size: (i + 1) * p_size, j * p_size: (j + 1) * p_size, :])
                        test_label_masks.append(label_mask2[:, i * p_size: (i + 1) * p_size, j * p_size: (j + 1) * p_size, :])
        else:
            if ind >= testset_size:
                train_images.append(image.reshape([1] + list(image.shape)))
                train_label_masks.append(label_mask.reshape([1] + list(label_mask.shape)))
            else:
                test_images.append(image.reshape([1] + list(image.shape)))
                test_label_masks.append(label_mask.reshape([1] + list(label_mask.shape)))

    train_images = np.concatenate(train_images, 0)
    train_label_masks = np.concatenate(train_label_masks, 0)

    test_images = np.concatenate(test_images, 0)
    test_label_masks = np.concatenate(test_label_masks, 0)
    test_IDs = dir_list[:testset_size]

    return train_images, train_label_masks, test_images, test_label_masks, test_IDs


def build_iterator(images, label_masks, batch_size, shuffle, **kwargs):

    data_gen = ImageDataGenerator(**kwargs)
    iterator = data_gen.flow(images, label_masks, batch_size=batch_size, shuffle=shuffle)

    return iterator
